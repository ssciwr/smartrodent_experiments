"""Utility helpers for SmartRodent YOLO and SpeciesNet experiments.

This module keeps path normalization, country-code handling, label formatting, and
SpeciesNet visualization/crop helpers out of ``main.py`` so the entry-point file can
focus on orchestrating model runs.
"""

import hashlib
import math
from pathlib import Path
import re
import importlib
from collections.abc import Iterable, Sequence
from PIL import Image
from typing import Any


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
    pad: float = 0.0,
) -> Path | None:
    """Save one padded normalized SpeciesNet box under its detector label.

    Args:
        image: Source image containing the detection.
        detection: SpeciesNet detection with a normalized ``bbox`` in
            ``[x_min, y_min, width, height]`` format.
        output_dir: Root directory for crop output.
        source_path: Original source image path, used to make stable crop names.
        crop_index: Detection index in the original SpeciesNet prediction list.
        pad: Relative total expansion for the crop. For example, ``pad=0.2`` makes
            the crop 20% wider and 20% taller by adding 10% of the original box size
            on each side. Values are clipped to the image bounds.
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")

    try:
        xmin, ymin, width, height = map(float, detection["bbox"])
    except (KeyError, TypeError, ValueError):
        return None

    # ``pad`` describes the total relative increase in crop size, so split it evenly
    # over both sides of each axis before clipping to the image boundary.
    x_pad = width * pad / 2
    y_pad = height * pad / 2

    left = max(0, min(image.width, math.floor((xmin - x_pad) * image.width)))
    top = max(0, min(image.height, math.floor((ymin - y_pad) * image.height)))
    right = max(0, min(image.width, math.ceil((xmin + width + x_pad) * image.width)))
    bottom = max(
        0, min(image.height, math.ceil((ymin + height + y_pad) * image.height))
    )
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


def import_and_get(importpath: str) -> Any:
    """Import a module and get an object from it.

    Args:
        importpath (str): The import path of the object to get.

    Returns:
        Any: The name as imported from the module.

    Raises:
        KeyError: When the module indicated by the path is not found
        KeyError: When the object name indidcated by the path is not found in the module
    """
    parts = importpath.split(".")
    module_name = ".".join(parts[:-1])
    object_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise KeyError(f"Importing module {module_name} unsuccessful") from e
    try:
        return getattr(module, object_name)
    except Exception as e:
        raise KeyError(f"Could not load name {object_name} from {module_name}") from e


def assign_at_path(cfg: dict, path: Sequence[Any], value: Any) -> None:
    """Assign a value to a key in a nested dictionary 'dict'. The path to follow through this nested structure is given by 'path'.

    Args:
        cfg (dict): The configuration dictionary to modify.
        path (Sequence[Any]): The path to the key to modify as a list of nodes to traverse.
        value (Any): The value to assign to the key.
    """
    for p in path[:-1]:
        cfg = cfg[p]
    cfg[path[-1]] = value


def get_at_path(cfg: dict, path: Sequence[Any], default: Any = None) -> Any:
    """Get the value at a key in a nested dictionary. The path to follow through this nested structure is given by 'path'.

    Args:
        cfg (dict): The configuration dictionary to modify.
        path (Sequence[Any]): The path to the key to get as a list of nodes to traverse.

    Returns:
        Any: The value at the specified key, or None if not found.
    """
    for p in path[:-1]:
        cfg = cfg[p]

    return cfg.get(path[-1], default)
