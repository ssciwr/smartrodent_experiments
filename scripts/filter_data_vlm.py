from pathlib import Path
from typing import ClassVar
import base64
import json
import shutil

import pandas as pd
import requests
import torch
import yaml
from tqdm.auto import tqdm

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "filter_data_vlm_config.yaml"
)


def decided_label(
    image_filter, preprocessed_images: list
) -> tuple[str | None, torch.Tensor]:
    """Return the winning prompt only when CLIP made a non-ambiguous decision."""
    similarity = image_filter.compute_similarity(preprocessed_images)
    decided_idx, _, decided, _ = image_filter.filter_similarities(similarity)
    if len(decided_idx) != 1:
        return None, similarity
    return image_filter.prompts[int(decided.argmax(dim=0).item())], similarity


class Detector:
    """Shared scaffolding for alive/dead/unsure wildlife image classification.

    Subclasses provide the actual inference backend by overriding filter_data().
    """

    def __init__(
        self,
        prompt: str,
        system_prompt: str,
        imgs_root: Path,
        alive_root: Path,
        unsure_root: Path,
        dead_root: Path,
        failure_root: Path,
        image_suffixes: set[str],
    ):
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.imgs_root = Path(imgs_root)
        self.alive_root = Path(alive_root)
        self.unsure_root = Path(unsure_root)
        self.dead_root = Path(dead_root)
        self.failure_root = Path(failure_root)
        self.image_suffixes = set(image_suffixes)

    @classmethod
    def from_config(cls, config_path: str | Path = CONFIG_PATH) -> "Detector":
        with open(config_path) as f:
            config = yaml.safe_load(f)

        imgs_root = Path(config["paths"]["imgs_root"])
        backend = config["backend"]
        common = dict(
            prompt=config["prompt"],
            system_prompt=config["system_prompt"],
            imgs_root=imgs_root,
            image_suffixes=set(config["paths"]["image_suffixes"]),
        )

        if backend == "ollama":
            model_tag = config["ollama"]["model"]
            detector_cls, extra = (
                DetectorOllama,
                dict(url=config["ollama"]["url"], model=config["ollama"]["model"]),
            )
        elif backend == "vllm":
            model_tag = config["vllm"]["model"].split("/")[-1]
            detector_cls, extra = (
                DetectorVLLM,
                dict(
                    model_name=config["vllm"]["model"],
                    gpu_memory_utilization=config["vllm"]["gpu_memory_utilization"],
                    max_model_len=config["vllm"]["max_model_len"],
                    max_new_tokens=config["vllm"]["max_new_tokens"],
                    batch_size=config["vllm"]["batch_size"],
                ),
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}")

        return detector_cls(
            **common,
            alive_root=imgs_root.parent / f"filtered_{model_tag}_kept",
            unsure_root=imgs_root.parent / f"filtered_{model_tag}_undecided",
            dead_root=imgs_root.parent / f"filtered_{model_tag}_rejected",
            failure_root=imgs_root.parent / f"filtere_{model_tag}_failure",
            **extra,
        )

    @property
    def model_tag(self) -> str:
        raise NotImplementedError

    def filter_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def save_results(self, res_df: pd.DataFrame) -> Path:
        out_path = self.imgs_root.parent / f"filter_results_{self.model_tag}.csv"
        res_df.to_csv(out_path)
        return out_path

    def copy_with_structure(self, src: Path, dst_root: Path) -> Path:
        """Copy one image to dst_root while preserving its path below imgs_root."""
        relative_path = src.relative_to(self.imgs_root)
        dst = dst_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    @staticmethod
    def parse_response(raw: str) -> dict:
        """Turn a model's raw JSON text response into the standard result schema."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "label": "failure",
                "visible_animal": None,
                "evidence_alive": [],
                "evidence_dead": [],
                "image_quality": "unknown",
                "needs_human_review": True,
                "raw_response": raw,
                "parse_error": True,
            }

        label = str(data.get("label", "failure")).strip().lower()

        if label not in ["alive", "dead", "unsure"]:
            label = "failure"

        return {
            "label": label,
            "visible_animal": data.get("visible_animal"),
            "evidence_alive": data.get("evidence_alive", []),
            "evidence_dead": data.get("evidence_dead", []),
            "image_quality": data.get("image_quality", "unsure"),
            "needs_human_review": bool(
                data.get("needs_human_review", label == "unsure")
            ),
            "raw_response": raw,
            "parse_error": False,
        }

    def collect_image_paths(self) -> list[Path]:
        species_dirs = sorted(p for p in self.imgs_root.iterdir() if p.is_dir())
        return [
            image_path
            for species_path in species_dirs
            for image_path in sorted(species_path.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in self.image_suffixes
        ]

    @property
    def dest_by_label(self) -> dict[str, Path]:
        return {
            "alive": self.alive_root,
            "dead": self.dead_root,
            "unsure": self.unsure_root,
            "failure": self.failure_root,
        }


class DetectorOllama(Detector):
    """Classifies images one at a time via a local Ollama server. Fallback backend."""

    def __init__(self, *, url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.model = model

    @property
    def model_tag(self) -> str:
        return self.model

    def classify_image(self, path: Path) -> dict:
        img_b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": self.prompt,
            "images": [img_b64],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
            },
        }

        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()

        return self.parse_response(response.json()["response"])

    def filter_data(self) -> pd.DataFrame:
        results = []
        image_paths = self.collect_image_paths()
        dest_by_label = self.dest_by_label

        for image_path in tqdm(image_paths, desc="OLLAMA filtering images"):
            res = self.classify_image(image_path)
            res["species"] = image_path.relative_to(self.imgs_root).parts[0]
            results.append(res)
            self.copy_with_structure(image_path, dest_by_label[res["label"]])

        return pd.DataFrame(results)


class DetectorVLLM(Detector):
    """Classifies images in batches via a local vLLM offline inference engine."""

    RESPONSE_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["alive", "dead", "unsure"]},
            "visible_animal": {"type": "boolean"},
            "evidence_alive": {"type": "array", "items": {"type": "string"}},
            "evidence_dead": {"type": "array", "items": {"type": "string"}},
            "image_quality": {"type": "string", "enum": ["clear", "poor", "unusable"]},
            "needs_human_review": {"type": "boolean"},
        },
        "required": [
            "label",
            "visible_animal",
            "evidence_alive",
            "evidence_dead",
            "image_quality",
            "needs_human_review",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        model_name: str,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 8192,
        max_new_tokens: int = 300,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._llm = None
        self._sampling_params = None

    @property
    def model_tag(self) -> str:
        return self.model_name.split("/")[-1]

    def _ensure_engine(self) -> None:
        if self._llm is not None:
            return

        # Imported lazily so the Ollama fallback still works without vLLM installed/working.
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._llm = LLM(
            model=self.model_name,
            limit_mm_per_prompt={"image": 1},
            # 0.9 (vLLM's default) OOMs on a 24GB card once the desktop/other
            # processes already hold a couple GB of VRAM; 0.7 leaves headroom.
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=self.RESPONSE_JSON_SCHEMA),
        )

    @staticmethod
    def image_to_data_url(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/{suffix};base64,{b64}"

    def build_conversation(self, image_path: Path) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self.image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]

    def classify_images_batch(self, image_paths: list[Path]) -> list[dict]:
        """Run one batched vLLM chat call and return parsed results in input order."""
        self._ensure_engine()
        conversations = [self.build_conversation(p) for p in image_paths]
        outputs = self._llm.chat(
            conversations, sampling_params=self._sampling_params, use_tqdm=False
        )
        return [self.parse_response(output.outputs[0].text) for output in outputs]

    def filter_data(self) -> pd.DataFrame:
        self._ensure_engine()
        image_paths = self.collect_image_paths()
        dest_by_label = self.dest_by_label

        results = []
        with tqdm(total=len(image_paths), desc="vLLM filtering images") as pbar:
            for start in range(0, len(image_paths), self.batch_size):
                chunk = image_paths[start : start + self.batch_size]
                for image_path, res in zip(chunk, self.classify_images_batch(chunk)):
                    res["species"] = image_path.relative_to(self.imgs_root).parts[0]
                    results.append(res)
                    self.copy_with_structure(image_path, dest_by_label[res["label"]])
                pbar.update(len(chunk))

        return pd.DataFrame(results)


if __name__ == "__main__":
    print("cuda? ", torch.cuda.is_available())

    detector = Detector.from_config(CONFIG_PATH)
    res_df = detector.filter_data()
    detector.save_results(res_df)
