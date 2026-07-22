from collections import Counter, defaultdict
from pathlib import Path
import shutil
from tqdm.auto import tqdm
import torch
import pandas as pd
import json
import requests
import base64

from smartrodent.dataprocessing import (
    ImageFilterCLIP,
    ImageFilterYoloE,
    ImageFilterBiotroveClip,
)


def copy_with_structure(src: Path, dst_root: Path) -> Path:
    """Copy one image to dst_root while preserving its path below imgs_root."""
    relative_path = src.relative_to(imgs_root)
    dst = dst_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def decided_label(
    image_filter: ImageFilterCLIP, preprocessed_images: list
) -> tuple[str | None, torch.Tensor]:
    """Return the winning prompt only when CLIP made a non-ambiguous decision."""
    similarity = image_filter.compute_similarity(preprocessed_images)
    decided_idx, _, decided, _ = image_filter.filter_similarities(similarity)
    if len(decided_idx) != 1:
        return None, similarity
    return image_filter.prompts[int(decided.argmax(dim=0).item())], similarity


def classify_image(path: str, prompt: str) -> dict:
    img_path = Path(path)
    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw = response.json()["response"]

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
        "needs_human_review": bool(data.get("needs_human_review", label == "unsure")),
        "raw_response": raw,
        "parse_error": False,
    }


def filter_data_ollama(
    prompt: str,
    raw_data_path: str,
    alive_path: str,
    unsure_path: str,
    dead_path: str,
    failure_path: str,
) -> pd.DataFrame:

    results = []
    species_dirs = sorted(p for p in Path(raw_data_path).iterdir() if p.is_dir())
    image_paths = [
        image_path
        for species_path in species_dirs
        for image_path in sorted(species_path.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in image_suffixes
    ]

    for image_path in tqdm(image_paths, desc="OLLAMA filtering images"):
        res = classify_image(image_path, prompt)
        res["species"] = image_path.relative_to(raw_data_path).parts[0]
        results.append(res)

        if res["label"] == "alive":
            copy_with_structure(image_path, alive_path)
        elif res["label"] == "dead":
            copy_with_structure(image_path, dead_path)
        elif res["label"] == "unsure":
            copy_with_structure(image_path, unsure_path)
        elif res["label"] == "failure":
            copy_with_structure(image_path, failure_path)
        else:
            raise ValueError(f"Unknown label {res['label']}")

    return pd.DataFrame(results)


print("cuda? ", torch.cuda.is_available())
# tols = [0.02, 0.05, 0.1]
tols = [
    0.01,
]
# Source directory: one subdirectory per species; each subdirectory name is the label.
imgs_root = Path(
    "/home/hmack/Development/rodent_experiments/datasets/biotrove-central-europe/raw"
)
results = {}
# First pass: animal/rodent-like versus not-an-animal.
not_rodent_prompt = "not an animal at all"
rodent_prompt = "a mouse, rat or other rodent-like animal"

# Second pass: usable live-animal images versus dead/specimen/uncertain images.
healthy_prompt = "a live healthy looking animal in its wild habitat or somewhere in a human dwelling or human environment"
dead_or_unsure_prompt = "a dead animal, roadkill, a skull or bones"

for tol in tols:
    results[tol] = {}

    primary_filter_clip = ImageFilterCLIP(
        model="ViT-L/14@336px",
        prompts=[not_rodent_prompt, rodent_prompt],
        id_tol=tol,
    )
    health_filter_clip = ImageFilterCLIP(
        model="ViT-L/14@336px",
        prompts=[healthy_prompt, dead_or_unsure_prompt],
        id_tol=tol,
    )

    # YOLOE is loaded through Ultralytics from the local weights bundled under
    # src/smartrodent, then configured with the text prompts below via set_classes.

    primary_filter_yoloe = ImageFilterYoloE(
        model="yoloe-26m-seg.pt",
        prompts=[not_rodent_prompt, rodent_prompt],
        id_tol=tol,
        predict_conf=tol,
    )
    health_filter_yoloe = ImageFilterYoloE(
        model="yoloe-26m-seg.pt",
        prompts=[healthy_prompt, dead_or_unsure_prompt],
        id_tol=tol,
        predict_conf=tol,
    )

    biotrove_clip_checkpoint = "biotroveclip-vit-b-16-from-openai-epoch-40.pt"

    primary_filter_biotrove_clip = ImageFilterBiotroveClip(
        model=biotrove_clip_checkpoint,
        prompts=[not_rodent_prompt, rodent_prompt],
        id_tol=tol,
    )
    health_filter_biotrove_clip = ImageFilterBiotroveClip(
        model=biotrove_clip_checkpoint,
        prompts=[healthy_prompt, dead_or_unsure_prompt],
        id_tol=tol,
    )

    for primary_filter, health_filter, name in zip(
        [
            primary_filter_clip,
        ],
        [
            health_filter_clip,
        ],
        [
            "clip",
        ],
    ):
        filtered_root = imgs_root.parent / f"filtered_{name}_kept_{tol}"
        rejected_root = imgs_root.parent / f"filtered_{name}_rejected_{tol}"
        undecided_root = imgs_root.parent / f"filtered_{name}_undecided_{tol}"

        # Leave this False to avoid deleting previous results accidentally. Set to True
        # when rebuilding the filtered directories from scratch.
        clear_existing_outputs = False

        if clear_existing_outputs:
            shutil.rmtree(filtered_root, ignore_errors=True)
            shutil.rmtree(rejected_root, ignore_errors=True)
            shutil.rmtree(undecided_root, ignore_errors=True)

        filtered_root.mkdir(parents=True, exist_ok=True)
        rejected_root.mkdir(parents=True, exist_ok=True)
        undecided_root.mkdir(parents=True, exist_ok=True)

        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        species_dirs = sorted(p for p in imgs_root.iterdir() if p.is_dir())
        image_paths = [
            image_path
            for species_dir in species_dirs
            for image_path in sorted(species_dir.rglob("*"))
            if image_path.is_file() and image_path.suffix.lower() in image_suffixes
        ]

        summary = Counter()
        by_species = defaultdict(Counter)
        errors = []

        for image_path in tqdm(image_paths, desc=f"{name} filtering images"):
            species = image_path.relative_to(imgs_root).parts[0]
            try:
                _, preprocessed = primary_filter.preprocess_images([image_path])
                primary_label, _ = decided_label(primary_filter, preprocessed)

                if primary_label is None:
                    copy_with_structure(image_path, undecided_root)
                    reason = "rodent_unsure"
                elif primary_label != rodent_prompt:
                    copy_with_structure(image_path, rejected_root)
                    reason = "not_rodent"
                else:
                    health_label, _ = decided_label(health_filter, preprocessed)
                    if health_label is None:
                        copy_with_structure(image_path, undecided_root)
                        reason = "health_unsure"
                    elif health_label == healthy_prompt:
                        copy_with_structure(image_path, filtered_root)
                        reason = "alive"
                    else:
                        copy_with_structure(image_path, rejected_root)
                        reason = "dead"

                summary[reason] += 1
                by_species[species][reason] += 1
            except Exception as exc:
                copy_with_structure(image_path, undecided_root)
                summary["error"] += 1
                by_species[species]["error"] += 1
                errors.append((str(image_path), repr(exc)))

        print("Input:", imgs_root)
        print("tol  : ", tol)
        print("name : ", name)
        print(" Kept live rodent images:", filtered_root)
        print(" Rejected/not-rodent/dead images:", rejected_root)
        print(" Undecided/error images:", undecided_root)
        print(" Total images:", len(image_paths))
        print(" Summary:", dict(summary))
        print(" By species:")
        for species, counts in sorted(by_species.items()):
            print(species, dict(counts))

        results[tol][name] = by_species

        if errors:
            print(f"Encountered {len(errors)} errors; first 10:")
            for path, error in errors[:10]:
                print(path, error)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.6:27b"

PROMPT = """
   You are classifying wildlife images.

   Return ONLY valid JSON with this exact schema:

   {
     "label": "alive | dead | unsure",
     "visible_animal": true,
     "evidence_alive": ["string"],
     "evidence_dead": ["string"],
     "image_quality": "clear | poor | unusable",
     "needs_human_review": true
   }

   Rules:
   - label must be exactly one of: "alive", "dead", "unsure".
   - Use "dead" only for clear evidence of death: carcass, roadkill, skull, bones,
 preserved specimen, visibly dead body.
   - Use "alive" only if a live animal is clearly visible.
   - Use "unsure" if no animal is visible, the image is ambiguous, cropped, low
 quality, a drawing, or you cannot confidently decide.
   - needs_human_review should be true for unsure, poor/unusable images, or conflicting
 evidence.
   - Do not include markdown.
   - Do not include text outside the JSON object.
   """

image_suffixes = {".jpg", ".jpeg", ".png"}


imgs_root = Path(
    "/home/hmack/Development/rodent_experiments/datasets/biotrove-central-europe/raw"
)
alive_root = imgs_root.parent / "filtered_qwen3.6_kept"
unsure_root = imgs_root.parent / "filtered_qwen3.6_undecided"
dead_root = imgs_root.parent / "filtered_qwen3.6_rejected"
failure_root = imgs_root.parent / "filtere_qwen3.6_failure"

res_df = filter_data_ollama(
    PROMPT, imgs_root, alive_root, unsure_root, dead_root, failure_root
)

res_df.to_csv(imgs_root.parent / "filter_results_qwen36.csv")
