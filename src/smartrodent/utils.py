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
from PIL import ImageDraw
from PIL import ImageFont
from speciesnet import draw_bboxes


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


def short_speciesnet_label(prediction: str | None) -> str:
    """Return the readable label from a SpeciesNet taxonomy prediction string.

    SpeciesNet labels look like ``uuid;class;order;family;genus;species;common name``.
    For quick console output and preview overlays, the common name at the end is the
    easiest part to read.
    """
    if not prediction:
        return "unknown"
    label = prediction.split(";")[-1]
    return label or prediction


def top_speciesnet_species(
    item: dict,
    limit: int = 2,
) -> list[tuple[str, float | None]]:
    """Return the top species-level classifier labels and scores."""
    classifications = item.get("classifications", {})
    classes = classifications.get("classes", [])
    scores = classifications.get("scores", [])
    species = []

    for label, score in zip(classes, scores, strict=False):
        parts = label.split(";")
        if len(parts) != 7:  # speciesnet taxonomy label depth = 7
            continue
        genus, species_name, common_name = parts[4], parts[5], parts[6]
        if genus and species_name:
            species.append((common_name or f"{genus} {species_name}", score))
        if len(species) == limit:
            break

    return species


def normalize_country(country: str | None) -> str | None:
    """Return the ISO-3166 alpha-3 country code SpeciesNet expects."""
    if country is None:
        return None

    normalized = COUNTRY_ALIASES.get(country.strip().upper())
    if normalized is not None:
        return normalized

    if len(country) == 3 and country.isalpha():
        return country.upper()

    raise ValueError(
        f"SpeciesNet geofencing expects an ISO-3166 alpha-3 country code, got "
        f"{country!r}. Use values like 'DEU' for Germany or 'LKA' for Sri Lanka."
    )


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
    return crop_path


def save_speciesnet_crops(
    predictions: dict,
    output_dir: str | Path = "runs/speciesnet/crops",
) -> list[Path]:
    """Extract every SpeciesNet detection into detector-label directories."""
    crop_paths = []
    for item in predictions["predictions"]:
        image_path = Path(item["filepath"])
        with Image.open(image_path) as image:
            for crop_index, detection in enumerate(item.get("detections", [])):
                crop_path = extract_crop(
                    image,
                    detection,
                    output_dir=output_dir,
                    source_path=image_path,
                    crop_index=crop_index,
                )
                if crop_path is not None:
                    crop_paths.append(crop_path)

    return crop_paths


def save_speciesnet_previews(
    predictions: dict,
    output_dir: str | Path = "runs/speciesnet/boxed",
) -> None:
    """Save boxed preview images from SpeciesNet prediction JSON.

    SpeciesNet's standard output is JSON. Each prediction can contain a ``detections``
    list from MegaDetector, with normalized bounding boxes. ``draw_bboxes`` converts
    those detections into visible boxes, and this helper adds the final ensemble label
    at the top-left of each image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for item in predictions["predictions"]:
        image_path = Path(item["filepath"])
        image = Image.open(image_path).convert("RGB")

        # SpeciesNet detections are already in the format expected by draw_bboxes.
        image = draw_bboxes(image, item.get("detections", [])).convert("RGB")

        label = short_speciesnet_label(item.get("prediction"))
        score = item.get("prediction_score")

        text = f"{label}: {score:.2f}" if score is not None else label
        top_species = top_speciesnet_species(item)
        species_lines = [
            f"{idx}. {name}: {species_score:.4f}"
            if species_score is not None
            else f"{idx}. {name}"
            for idx, (name, species_score) in enumerate(top_species, start=1)
        ]

        # Add small readable label strips above the boxes.
        draw = ImageDraw.Draw(image)

        text_box = draw.textbbox((0, 0), text, font=font)

        draw.rectangle(
            (0, 0, min(image.width, text_box[2] + 12), text_box[3] + 12),
            fill=(0, 0, 0),
        )

        draw.text((6, 6), text, fill=(255, 255, 255), font=font)

        if species_lines:
            species_boxes = [
                draw.textbbox((0, 0), species_line, font=font)
                for species_line in species_lines
            ]
            line_height = max(box[3] - box[1] for box in species_boxes)
            line_gap = 3
            padding = 6
            species_top = text_box[3] + 12
            species_bottom = (
                species_top
                + 2 * padding
                + len(species_lines) * line_height
                + (len(species_lines) - 1) * line_gap
            )
            species_width = max(box[2] - box[0] for box in species_boxes)

            draw.rectangle(
                (
                    0,
                    species_top,
                    min(image.width, species_width + 2 * padding),
                    species_bottom,
                ),
                fill=(35, 35, 35),
            )

            species_y = species_top + padding
            for species_line in species_lines:
                draw.text(
                    (padding, species_y),
                    species_line,
                    fill=(255, 255, 255),
                    font=font,
                )
                species_y += line_height + line_gap

        image.save(output_dir / image_path.name)
