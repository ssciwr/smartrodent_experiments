import json
import requests
from pathlib import Path
import base64
import pandas as pd
from tqdm import tqdm
import shutil

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


def copy_with_structure(src: Path, dst_root: Path) -> Path:
    """Copy one image to dst_root while preserving its path below imgs_root."""
    relative_path = src.relative_to(imgs_root)
    dst = dst_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


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


if __name__ == "__main__":
    imgs_root = Path(
        "/home/hmack/Development/rodent_experiments/datasets/biotrove-central-europe/imgs"
    )
    alive_root = imgs_root.parent / "filtered_qwen3.6"
    unsure_root = imgs_root.parent / "filtered_qwen3.6_undecided"
    dead_root = imgs_root.parent / "filtered_qwen3.6_rejected"
    failure_root = imgs_root.parent / "filtere_qwen3.6_failure"

    res_df = filter_data_ollama(
        PROMPT, imgs_root, alive_root, unsure_root, dead_root, failure_root
    )

    res_df.to_csv(imgs_root.parent / "filter_results_qwen36.csv")
