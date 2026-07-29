from pathlib import Path
import base64
import json
import shutil
import pandas as pd
import requests
import yaml
from tqdm.auto import tqdm

from .base import Filterable
from .utils import resolve_data_path


class VLMFilter(Filterable):
    """Shared scaffolding for kept/rejected/unsure wildlife image classification.

    Subclasses provide the actual inference backend by overriding filter_data().
    """

    def __init__(
        self,
        prompt: str,
        system_prompt: str,
        imgs_root: Path,
        kept_root: Path,
        unsure_root: Path,
        rejected_root: Path,
        failure_root: Path,
        image_suffixes: set[str],
        species: list[str] | None = None,
        mode: str = "copy",
    ):
        """Initialize the filter.

        Args:
            prompt: User prompt sent to the VLM for each image.
            system_prompt: System prompt sent to the VLM.
            imgs_root: Root directory containing per-species image subfolders.
            kept_root: Destination root for images classified as kept.
            unsure_root: Destination root for images classified as unsure.
            rejected_root: Destination root for images classified as rejected.
            failure_root: Destination root for images whose classification failed.
            image_suffixes: File suffixes (e.g. ``.jpg``) treated as images.
            species: Species names to include. If None, all species subfolders
                under ``imgs_root`` are processed.
            mode: How to place classified images into destination folders,
                either ``"copy"`` or ``"move"``.

        Raises:
            ValueError: If ``mode`` is not ``"copy"`` or ``"move"``.
        """
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.imgs_root = Path(imgs_root)
        self.kept_root = Path(kept_root)
        self.unsure_root = Path(unsure_root)
        self.rejected_root = Path(rejected_root)
        self.failure_root = Path(failure_root)
        self.image_suffixes = set(image_suffixes)
        self.species = species
        print("self.species: ", self.species)
        if self.species is not None:
            self.species = [s.lower() for s in self.species]
        if mode == "copy":
            self.data_func = shutil.copy2
        elif mode == "move":
            self.data_func = shutil.move
        else:
            raise ValueError("Error, mode must be 'move' or 'copy'")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Filter":
        """Build a filter instance from a YAML config file.

        Reads paths, prompts, species, and backend-specific settings from the
        config, creates the output directories (kept/unsure/rejected/failure),
        and copies the config into each for provenance.

        Args:
            config_path: Path to the YAML config file. Must specify a
                ``backend`` of either ``"ollama"`` or ``"vllm"``, along with
                the corresponding backend settings section.

        Returns:
            Filter: A ``FilterOllama`` or ``FilterVLLM`` instance, depending
            on the configured backend.

        Raises:
            ValueError: If ``backend`` is not ``"ollama"`` or ``"vllm"``.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        imgs_root = resolve_data_path(config["paths"]["imgs_root"])
        backend = config["backend"]
        common = dict(
            prompt=config["prompt"],
            system_prompt=config["system_prompt"],
            imgs_root=imgs_root,
            image_suffixes=set(config["paths"]["image_suffixes"]),
            species=config.get("species"),
            mode=config.get("mode", "copy"),
        )

        if backend == "ollama":
            detector_cls, extra = (
                FilterOllama,
                dict(
                    url=config["ollama"]["url"],
                    model=config["ollama"]["model"],
                ),
            )
        elif backend == "vllm":
            detector_cls, extra = (
                FilterVLLM,
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

        kept_root = imgs_root.parent / "filtered_kept"
        unsure_root = imgs_root.parent / "filtered_undecided"
        rejected_root = imgs_root.parent / "filtered_rejected"
        failure_root = imgs_root.parent / "filtered_failure"

        # Drop a copy of the config next to each output so the stage that
        # produced a given folder can always be identified after the fact.
        for root in (kept_root, unsure_root, rejected_root, failure_root):
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, root / config_path.name)

        return detector_cls(
            **common,
            kept_root=kept_root,
            unsure_root=unsure_root,
            rejected_root=rejected_root,
            failure_root=failure_root,
            **extra,
        )

    @property
    def model_tag(self) -> str:
        """str: Short identifier for the backend model, used e.g. in filenames."""
        raise NotImplementedError

    def filter_data(self) -> pd.DataFrame:
        """Classify all collected images and sort them into destination folders.

        Returns:
            pd.DataFrame: One row per processed image with its classification.
        """
        raise NotImplementedError

    def save_results(self, res_df: pd.DataFrame) -> Path:
        """Write filtering results to a CSV next to the images root.

        Args:
            res_df: The results DataFrame returned by :meth:`filter_data`.

        Returns:
            Path: Path to the written ``filter_results.csv`` file.
        """
        out_path = self.imgs_root.parent / "filter_results.csv"
        res_df.to_csv(out_path)
        return out_path

    def copy_with_structure(self, src: Path, dst_root: Path) -> Path:
        """Copy one image to dst_root while preserving its path below imgs_root.

        Args:
            src: Path to the source image, located under ``self.imgs_root``.
            dst_root: Destination root directory to copy/move the image into.

        Returns:
            Path: The destination path the image was written to.
        """
        relative_path = src.relative_to(self.imgs_root)
        dst = dst_root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.data_func(src, dst)
        except shutil.SameFileError:
            pass
        except Exception as e:
            raise e
        return dst

    @staticmethod
    def parse_response(raw: str) -> dict:
        """Turn a model's raw JSON text response into the standard result schema.

        Args:
            raw: Raw text response returned by the VLM backend, expected to
                be a JSON object.

        Returns:
            dict: A result dict with keys ``label``, ``visible_animal``,
            ``evidence_kept``, ``evidence_rejected``, ``image_quality``,
            ``needs_human_review``, ``raw_response``, and ``parse_error``.
            ``label`` falls back to ``"failure"`` if parsing fails or the
            label is not one of ``"kept"``, ``"rejected"``, ``"unsure"``.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "label": "failure",
                "visible_animal": None,
                "evidence_kept": [],
                "evidence_rejected": [],
                "image_quality": "unknown",
                "needs_human_review": True,
                "raw_response": raw,
                "parse_error": True,
            }

        label = str(data.get("label", "failure")).strip().lower()

        if label not in ["kept", "rejected", "unsure"]:
            label = "failure"

        return {
            "label": label,
            "visible_animal": data.get("visible_animal"),
            "evidence_kept": data.get("evidence_kept", []),
            "evidence_rejected": data.get("evidence_rejected", []),
            "image_quality": data.get("image_quality", "unsure"),
            "needs_human_review": bool(
                data.get("needs_human_review", label == "unsure")
            ),
            "raw_response": raw,
            "parse_error": False,
        }

    def collect_image_paths(self) -> list[Path]:
        """Gather image paths from imgs_root, optionally filtered by species.

        Returns:
            list[Path]: Paths of images whose suffix is in
            ``self.image_suffixes``, found in the (optionally species-filtered)
            subdirectories of ``self.imgs_root``.
        """
        species_dirs = sorted(p for p in self.imgs_root.iterdir() if p.is_dir())

        if self.species is not None:
            species_dirs = [s for s in species_dirs if s.name.lower() in self.species]

        image_paths = [
            image_path
            for species_path in species_dirs
            for image_path in sorted(species_path.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in self.image_suffixes
        ]

        return image_paths

    @property
    def dest_by_label(self) -> dict[str, Path]:
        """dict[str, Path]: Mapping from classification label to destination root."""
        return {
            "kept": self.kept_root,
            "rejected": self.rejected_root,
            "unsure": self.unsure_root,
            "failure": self.failure_root,
        }


class FilterOllama(VLMFilter):
    """Classifies images one at a time via a local Ollama server. Fallback backend."""

    def __init__(self, *, url: str, model: str, **kwargs):
        """Initialize the Ollama-backed filter.

        Args:
            url: URL of the Ollama server's generate endpoint.
            model: Name of the Ollama model to use for classification.
            **kwargs: Additional arguments forwarded to :class:`VLMFilter`.
        """
        super().__init__(**kwargs)
        self.url = url
        self.model = model

    @property
    def model_tag(self) -> str:
        """str: The Ollama model name."""
        return self.model

    def classify_image(self, path: Path) -> dict:
        """Classify a single image via the Ollama server.

        Args:
            path: Path to the image file to classify.

        Returns:
            dict: The parsed classification result, as returned by
            :meth:`VLMFilter.parse_response`.
        """
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
        """Classify each collected image one at a time via Ollama.

        Each image is classified, tagged with its species (inferred from its
        parent folder name), and copied/moved into the destination folder for
        its label.

        Returns:
            pd.DataFrame: One row per processed image with its classification
            and species.
        """
        results = []
        image_paths = self.collect_image_paths()
        dest_by_label = self.dest_by_label

        for image_path in tqdm(image_paths, desc="OLLAMA filtering images"):
            res = self.classify_image(image_path)
            res["species"] = image_path.relative_to(self.imgs_root).parts[0]
            results.append(res)
            self.copy_with_structure(image_path, dest_by_label[res["label"]])

        return pd.DataFrame(results)


class FilterVLLM(VLMFilter):
    """Classifies images in batches via a local vLLM offline inference engine."""

    RESPONSE_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["kept", "rejected", "unsure"]},
            "visible_animal": {"type": "boolean"},
            "evidence_kept": {"type": "array", "items": {"type": "string"}},
            "evidence_rejected": {"type": "array", "items": {"type": "string"}},
            "image_quality": {"type": "string", "enum": ["clear", "poor", "unusable"]},
            "needs_human_review": {"type": "boolean"},
        },
        "required": [
            "label",
            "visible_animal",
            "evidence_kept",
            "evidence_rejected",
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
        """Initialize the vLLM-backed filter.

        Args:
            model_name: Name or path of the vLLM model to load.
            gpu_memory_utilization: Fraction of GPU memory vLLM is allowed
                to reserve.
            max_model_len: Maximum context length for the model.
            max_new_tokens: Maximum number of tokens to generate per response.
            batch_size: Number of images to classify per batched chat call.
            **kwargs: Additional arguments forwarded to :class:`VLMFilter`.
        """
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
        """str: The vLLM model name, with any path prefix stripped."""
        return self.model_name.split("/")[-1]

    def _ensure_engine(self) -> None:
        """Lazily construct the vLLM engine and sampling params on first use."""
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
        """Encode an image file as a base64 data URL.

        Args:
            path: Path to the image file.

        Returns:
            str: A ``data:image/<suffix>;base64,...`` URL suitable for use in
            a vLLM chat message.
        """
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/{suffix};base64,{b64}"

    def build_conversation(self, image_path: Path) -> list[dict]:
        """Build the vLLM chat message list for classifying one image.

        Args:
            image_path: Path to the image to include in the conversation.

        Returns:
            list[dict]: A system message followed by a user message
            containing the image (as a data URL) and the classification
            prompt.
        """
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
        """Run one batched vLLM chat call and return parsed results in input order.

        Args:
            image_paths: Paths of the images to classify together in one batch.

        Returns:
            list[dict]: Parsed classification results, in the same order as
            ``image_paths``.
        """
        self._ensure_engine()
        conversations = [self.build_conversation(p) for p in image_paths]
        outputs = self._llm.chat(
            conversations, sampling_params=self._sampling_params, use_tqdm=False
        )
        return [self.parse_response(output.outputs[0].text) for output in outputs]

    def filter_data(self) -> pd.DataFrame:
        """Classify each collected image in batches via vLLM.

        Images are processed in chunks of ``self.batch_size``, tagged with
        their species (inferred from their parent folder name), and
        copied/moved into the destination folder for their label.

        Returns:
            pd.DataFrame: One row per processed image with its classification
            and species.
        """
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
