"""Small YOLO and SpeciesNet experiments on the smartrodent image set.

The YOLO path saves Ultralytics preview images directly. The SpeciesNet path writes
its standard JSON predictions and then creates separate preview images from the
MegaDetector boxes included in that JSON.
"""

import json
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from speciesnet import DEFAULT_MODEL
from speciesnet import SpeciesNet
from speciesnet import draw_bboxes
from ultralytics import YOLO


IMAGE_DIR = Path(
    "/mnt/dataLinux/machinelearning_data/smartrodent/irodent/rodent/images/"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(path: str | list[Path]):
    """Run YOLO on one image path or a batch of image paths.

    Ultralytics handles saving boxed preview images when ``save=True`` is passed to
    ``model.predict``. Single-image and batch paths are split here only so the batch
    case can use a larger ``batch`` value.
    """
    # Load a COCO-pretrained YOLO model. This file is already present in src/smartrodent.
    model = YOLO("yolo26m.pt")

    # Optional training call kept here as a reminder/example for later experiments.
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    if isinstance(path, str):
        return model.predict(
            source=str(Path(path).resolve()),
            save=True,
            project="runs/rodent_predictions",
            name="boxed",
            exist_ok=True,
            conf=0.1,
        )

    return model.predict(
        source=path,
        batch=32,
        save=True,
        project="runs/rodent_predictions",
        name="boxed",
        exist_ok=True,
        conf=0.1,
    )


def _image_paths(path: str | Path | Iterable[str | Path]) -> list[str]:
    """Normalize an image path, folder, or iterable of paths into absolute strings.

    SpeciesNet accepts lists of file path strings. If a directory is supplied, only
    common image suffixes are included so accidental sidecar files are skipped.
    """
    if isinstance(path, str | Path):
        path = Path(path)
        if path.is_dir():
            return [
                str(img.resolve())
                for img in sorted(path.iterdir())
                if img.suffix.lower() in IMAGE_SUFFIXES
            ]
        return [str(path.resolve())]

    return [str(Path(img).resolve()) for img in path]


def _short_speciesnet_label(prediction: str | None) -> str:
    """Return the readable label from a SpeciesNet taxonomy prediction string.

    SpeciesNet labels look like ``uuid;class;order;family;genus;species;common name``.
    For quick console output and preview overlays, the common name at the end is the
    easiest part to read.
    """
    if not prediction:
        return "unknown"
    label = prediction.split(";")[-1]
    return label or prediction


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

        label = _short_speciesnet_label(item.get("prediction"))
        score = item.get("prediction_score")

        text = f"{label}: {score:.2f}" if score is not None else label

        # Add a small readable label strip above the boxes.
        draw = ImageDraw.Draw(image)

        text_box = draw.textbbox((0, 0), text, font=font)

        draw.rectangle(
            (0, 0, min(image.width, text_box[2] + 12), text_box[3] + 12),
            fill=(0, 0, 0),
        )

        draw.text((6, 6), text, fill=(255, 255, 255), font=font)

        image.save(output_dir / image_path.name)


def run_speciesnet(
    path: str | Path | Iterable[str | Path],
    *,
    output_json: str | Path = "runs/speciesnet/predictions.json",
    preview_dir: str | Path | None = "runs/speciesnet/boxed",
    batch_size: int = 16,
    country: str | None = None,
    admin1_region: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Run the SpeciesNet ensemble and optionally save boxed preview images.

    Args:
        path: One image path, an image directory, or an iterable of image paths.
        output_json: Where to write the SpeciesNet JSON predictions.
        preview_dir: Where to write boxed preview images. Set to ``None`` to skip.
        batch_size: Classifier batch size used by SpeciesNet.
        country: Optional ISO-3166 alpha-3 country code for geofencing, e.g. ``"USA"``.
        admin1_region: Optional first-level region code. For the USA this is a state,
            e.g. ``"CA"``.
        model_name: SpeciesNet model identifier. Defaults to the package default.

    Returns:
        The SpeciesNet predictions dictionary loaded from or returned by SpeciesNet.
    """
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    model = SpeciesNet(model_name)
    predictions = model.predict(
        filepaths=_image_paths(path),
        country=country,
        admin1_region=admin1_region,
        batch_size=batch_size,
        progress_bars=True,
        predictions_json=output_json,
    )

    # SpeciesNet returns None when predictions_json is supplied, because it writes the
    # result directly to disk. Reload it so callers still get a normal Python dict.
    if predictions is None:
        predictions = json.loads(output_json.read_text())

    if preview_dir is not None:
        save_speciesnet_previews(predictions, preview_dir)

    return predictions


if __name__ == "__main__":
    imgs = sorted(IMAGE_DIR.iterdir())[0:2]

    # YOLO writes boxed preview images under runs/rodent_predictions/boxed.
    # res = main(imgs)

    # SpeciesNet writes JSON first, then this script renders boxed preview images.
    speciesnet_results = run_speciesnet(
        imgs,
        output_json="runs/speciesnet/predictions.json",
        preview_dir="runs/speciesnet/boxed",
        batch_size=16,
        country="USA",
    )

    for item in speciesnet_results["predictions"]:
        print(
            f"{Path(item['filepath']).name}: "
            f"{_short_speciesnet_label(item.get('prediction'))} "
            f"({item.get('prediction_score', 0):.2f})"
        )
