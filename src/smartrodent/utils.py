"""Utility helpers for SmartRodent YOLO and SpeciesNet experiments.

This module keeps path normalization, country-code handling, label formatting, and
SpeciesNet visualization/crop helpers out of ``main.py`` so the entry-point file can
focus on orchestrating model runs.
"""

import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COUNTRY_ALIASES = {
    "DE": "DEU",
    "DEU": "DEU",
    "GERMANY": "DEU",
    "LK": "LKA",
    "LKA": "LKA",
    "SRI LANKA": "LKA",
    "SRI_LANKA": "LKA",
    "US": "USA",
    "USA": "USA",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
}


def image_paths(path: str | Path | Iterable[str | Path]) -> list[str]:
    """Normalize an image path, folder, or iterable of paths into absolute strings.

    SpeciesNet accepts lists of file path strings. If a directory is supplied, only
    common image suffixes are included so accidental sidecar files are skipped.
    """
    if isinstance(path, str | Path):
        path = Path(path)
        if path.is_dir():
            return [
                str(img.resolve())
                for img in path.iterdir()
                if img.suffix.lower() in IMAGE_SUFFIXES
            ]
        return [str(path.resolve())]

    return [str(Path(img).resolve()) for img in path]


def path_component(value: str) -> str:
    """Return a readable value that is safe to use as a directory or file name."""
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return value or "unknown"


def extract_crop(
    image: Image.Image,
    detection: dict,
    *,
    output_dir: str | Path,
    source_path: str | Path,
    crop_index: int,
) -> Path | None:
    """Save one normalized SpeciesNet box under its detector label."""
    try:
        xmin, ymin, width, height = map(float, detection["bbox"])
    except (KeyError, TypeError, ValueError):
        return None

    left = max(0, min(image.width, math.floor(xmin * image.width)))
    top = max(0, min(image.height, math.floor(ymin * image.height)))
    right = max(0, min(image.width, math.ceil((xmin + width) * image.width)))
    bottom = max(0, min(image.height, math.ceil((ymin + height) * image.height)))
    if right <= left or bottom <= top:
        return None

    source_path = Path(source_path)
    detection_label = path_component(str(detection.get("label", "unknown")))
    class_dir = Path(output_dir) / detection_label
    class_dir.mkdir(parents=True, exist_ok=True)

    # Include the source path hash so equal image names from separate folders do not
    # overwrite each other when several datasets are compared in one crop directory.
    source_id = hashlib.blake2s(
        str(source_path.resolve()).encode(), digest_size=4
    ).hexdigest()
    filename = f"{path_component(source_path.stem)}_{source_id}_{crop_index:03d}.jpg"
    crop_path = class_dir / filename
    image.crop((left, top, right, bottom)).convert("RGB").save(crop_path)
    # print("  saved image crop to crop_path: ", crop_path)
    return crop_path
