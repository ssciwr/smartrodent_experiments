"""Command and library entry points for SmartRodent model experiments.

This file intentionally contains the high-level workflow only: loading models,
choosing inputs and outputs, and calling the helper functions in ``utils.py``.
Keeping the lower-level image/path/label utilities separate makes this module easier
to read and also lets other scripts import those helpers from the package.
"""

import json
from pathlib import Path
from typing import Iterable

from speciesnet import DEFAULT_MODEL
from speciesnet import SpeciesNet
from ultralytics import YOLO

try:
    # Package import path, used when running with ``python -m smartrodent.main`` or
    # importing ``smartrodent`` from another script.
    from .utils import image_paths
    from .utils import normalize_country
    from .utils import save_speciesnet_crops
    from .utils import save_speciesnet_previews
    from .utils import short_speciesnet_label
except ImportError:  # pragma: no cover - convenience fallback for direct script runs.
    # Direct script path, used when running ``python main.py`` from this directory.
    from utils import image_paths
    from utils import normalize_country
    from utils import save_speciesnet_crops
    from utils import save_speciesnet_previews
    from utils import short_speciesnet_label


def main(
    path: str | Path | list[Path | str],
    batchsize: int,
    project: Path | str = "runs/yolo26",
    crop: bool = False,
):
    """Run YOLO on one image path or a batch of image paths.

    Ultralytics handles saving boxed preview images when ``save=True`` is passed to
    ``model.predict``. Single-image and batch paths are split here only so the batch
    case can use a larger ``batch`` value.
    """
    # Load the local COCO-pretrained YOLO model. Resolve the weights relative to this
    # file so imports from outside the package still find the bundled model file.
    model = YOLO(Path(__file__).with_name("yolo26m.pt"))

    # Optional training call kept here as a reminder/example for later experiments.
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    if isinstance(path, str | Path):
        # A single image does not need an explicit batch size. Resolve the path so
        # Ultralytics receives an absolute filename regardless of the caller's cwd.
        return model.predict(
            source=str(Path(path).resolve()),
            save=True,
            project=project,
            name="boxed",
            exist_ok=True,
            conf=0.01,
            save_crop=crop,
        )

    # For a list of images, pass the caller's batch size through to Ultralytics so
    # larger datasets can be processed more efficiently.
    return model.predict(
        source=path,
        batch=batchsize,
        save=True,
        project=project,
        name="boxed",
        exist_ok=True,
        conf=0.1,
        save_crop=crop,
    )


def run_speciesnet(
    path: str | Path | Iterable[str | Path],
    *,
    output_json: str | Path = "runs/speciesnet/predictions.json",
    preview_dir: str | Path | None = "runs/speciesnet/boxed",
    crop_dir: str | Path | None = "runs/speciesnet/crops",
    batch_size: int = 16,
    country: str | None = None,
    admin1_region: str | None = None,
    model_name: str = DEFAULT_MODEL,
    resume: bool = False,
) -> dict:
    """Run the SpeciesNet ensemble and optionally save boxed preview images.

    Args:
        path: One image path, an image directory, or an iterable of image paths.
        output_json: Where to write the SpeciesNet JSON predictions.
        preview_dir: Where to write boxed preview images. Set to ``None`` to skip.
        crop_dir: Where to save detected objects, grouped by box-level detector label.
            Set to ``None`` to skip crop extraction.
        batch_size: Classifier batch size used by SpeciesNet.
        country: Optional country for geofencing. ISO-3166 alpha-3 codes are passed
            through, and common names used in this project, such as "Germany" and
            "Sri Lanka", are normalized.
        admin1_region: Optional first-level region code. For the USA this is a state,
            e.g. ``"CA"``.
        model_name: SpeciesNet model identifier. Defaults to the package default.
        resume: Reuse complete predictions already present in output_json. Leave
            this disabled when changing country/geofence settings.

    Returns:
        The SpeciesNet predictions dictionary loaded from or returned by SpeciesNet.
    """
    # SpeciesNet writes JSON predictions to disk, so create the output folder before
    # the model starts. This also makes failures easier to diagnose because the target
    # path is explicit.
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # SpeciesNet geofencing expects ISO alpha-3 country codes. The utility accepts a
    # few friendly aliases (for example, "Germany" -> "DEU").
    country = normalize_country(country)

    # If resume is disabled, remove stale predictions. Otherwise SpeciesNet may skip
    # work by reusing previous JSON that was generated with different settings.
    if output_json.exists() and not resume:
        output_json.unlink()

    # Enable all SpeciesNet components: detector, classifier, and ensemble. Geofence
    # is useful for this dataset because location can reduce implausible species.
    model = SpeciesNet(model_name, geofence=True, components="all")
    predictions = model.predict(
        filepaths=image_paths(path),
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

    # Preview images are optional because they add extra image I/O. They are useful
    # for quick visual inspection of MegaDetector boxes and final SpeciesNet labels.
    if preview_dir is not None:
        save_speciesnet_previews(predictions, preview_dir)

    # Crops are also optional. When enabled, detections are saved into folders named
    # after detector labels so they can be inspected or used for later experiments.
    if crop_dir is not None:
        save_speciesnet_crops(predictions, crop_dir)

    return predictions


if __name__ == "__main__":
    # Example experiment block. The functions above can be imported by notebooks or
    # other scripts, while this block lets the file be run directly for ad-hoc tests.
    # IMAGE_DIR = Path(
    # "/mnt/dataLinux/machinelearning_data/smartrodent/irodent/rodent/images/"
    # )

    IMAGE_DIR = Path(
        "/home/hmack/Development/rodent_experiments/datasets/biotrove-sri-lanka/imgs/Rattus rattus"
    )
    out = Path("./runs/yolo/sri-lanka/Rattus rattus")
    imgs = sorted(IMAGE_DIR.iterdir())
    out.mkdir(parents=True, exist_ok=True)

    # YOLO writes boxed preview images under the configured project/name directory.
    for img in imgs:
        res = main(img, 1, crop=True)

    # SpeciesNet writes JSON first, then this script renders boxed preview images.
    out = Path("./runs/speciesnet/sri-lanka/Rattus rattus")

    speciesnet_results = run_speciesnet(
        imgs,
        output_json=out / "predictions.json",
        preview_dir=out / "boxed",
        batch_size=16,
        country="LKA",
        model_name="kaggle:google/speciesnet/pyTorch/v4.0.3b/1",
        crop_dir=out / "crops",
    )

    for item in speciesnet_results["predictions"]:
        print(
            f"{Path(item['filepath']).name}: "
            f"{short_speciesnet_label(item.get('prediction'))} "
            f"({item.get('prediction_score', 0):.2f})"
        )
