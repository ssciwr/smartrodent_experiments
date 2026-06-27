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
from ultralytics import YOLO, YOLOE
from speciesnet.detector import SpeciesNetDetector
import string
import open_clip
from PIL import Image
import torch

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


def detect(
    path: str | Path | list[Path | str],
    batchsize: int,
    project: Path | str = "runs/yolo26",
    name=" boxed",
    crop: bool = False,
    conf=0.1,
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
            name=name,
            exist_ok=True,
            conf=conf,
            save_crop=crop,
        )

    # For a list of images, pass the caller's batch size through to Ultralytics so
    # larger datasets can be processed more efficiently.
    return model.predict(
        source=path,
        batch=batchsize,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
        conf=conf,
        save_crop=crop,
    )


def run_speciesnet(
    path: str | Path | Iterable[str | Path],
    *,
    output_json: str | Path = "runs/segment/speciesnet/predictions.json",
    preview_dir: str | Path | None = "runs/segment/speciesnet/boxed",
    crop_dir: str | Path | None = "runs/segment/speciesnet/crops",
    batch_size: int = 16,
    country: str | None = None,
    admin1_region: str | None = None,
    model_name: str = DEFAULT_MODEL,
    resume: bool = False,
    conf: float = 0.1,
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
        conf: confidence value for detections
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
    model.detector.DETECTION_THRESHOLD = conf
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


def detect_biotrove_clip(
    path: str | Path | list[Path | str],
    batchsize: int,
    project: Path | str = "runs/biotrove-clip",
    crop: bool = False,
    conf=0.1,
    model_checkpoint: str = "biotroveclip-vit-b-16-from-openai-epoch-40.pt",
    classes: list[str] | None = None,
    prompt_template: str = "This is a photo of {}",
    model_dir: str | Path | None = None,
) -> dict:
    """Run BioTrove-CLIP as a zero-shot whole-image classifier.

    This mirrors ``ImageFilter`` in ``dataprocessing.py``: each image is loaded with
    Pillow, converted to RGB, passed through the OpenCLIP preprocessing transform,
    and compared against tokenized text prompts. BioTrove-CLIP does not emit boxes,
    so the return value uses the same dict shape that ``write_detections_json``
    already handles for SpeciesNet: ``{"predictions": [...]}`` with
    ``filepath``, ``prediction`` and ``prediction_score`` fields.
    """
    if batchsize <= 0:
        raise ValueError(f"batchsize must be positive, got {batchsize}")
    if crop:
        raise ValueError("BioTrove-CLIP classifies whole images and cannot save crops.")

    classes = classes or ["rodent", "non-rodent"]
    paths = image_paths(path)

    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)

    model_dir = Path("../../models/BioTrove-CLIP").resolve()
    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = model_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"BioTrove-CLIP checkpoint not found: {checkpoint_path}"
        )
    model_dir = f"local-dir:{str(model_dir)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        str(model_dir),
        pretrained=str(checkpoint_path),
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(str(model_dir))
    model.eval()

    text_tokens = tokenizer([prompt_template.format(label) for label in classes]).to(
        device
    )
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    predictions = []
    # range(..., batchsize) plus slicing includes the last incomplete batch.
    for start in range(0, len(paths), batchsize):
        batch_paths = paths[start : start + batchsize]
        batch_images = []
        valid_paths = []

        for filename in batch_paths:
            try:
                image = Image.open(filename).convert("RGB")
                batch_images.append(preprocess(image))
                valid_paths.append(filename)
            except Exception as exc:
                print(f"Skipping unreadable image {filename}: {exc}")

        if not batch_images:
            continue

        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(
                dim=-1
            )  # what other metrics could there be here? not so sure...

        for filename, scores in zip(valid_paths, probs.cpu(), strict=False):
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            prediction = classes[best_idx] if best_score >= conf else None
            predictions.append(
                {
                    "filepath": filename,
                    "prediction": prediction,
                    "prediction_score": best_score if prediction is not None else None,
                    "classifications": {
                        "classes": classes,
                        "scores": [float(score) for score in scores],
                    },
                }
            )

    result = {"predictions": predictions}
    (project / "predictions.json").write_text(json.dumps(result, indent=2))
    return result


def detect_YOLOE(
    path: str | Path | list[Path | str],
    batchsize: int,
    project: Path | str = "runs/yolo26",
    crop: bool = False,
    conf=0.1,
    task: str = "detect",
    classes: list[str] | None = ["rodent", "non-rodent"],
):
    """Run YOLO on one image path or a batch of image paths.

    Ultralytics handles saving boxed preview images when ``save=True`` is passed to
    ``model.predict``. Single-image and batch paths are split here only so the batch
    case can use a larger ``batch`` value.
    """
    # Load the local COCO-pretrained YOLO model. Resolve the weights relative to this
    # file so imports from outside the package still find the bundled model file.
    model = YOLOE(
        "yoloe-26m-seg.pt",
        task=task,
    )

    if classes:
        model.set_classes(
            classes,
            model.get_text_pe(classes),
        )

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
            conf=conf,
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
        conf=conf,
        save_crop=crop,
    )


def write_detections_json(results, json_path: Path | str) -> None:
    """Append detection records to a JSON file from any supported model.

    Accepts an Ultralytics Results list (YOLO26 or YOLOE) or a SpeciesNet predictions
    dict. Existing entries are preserved so the file accumulates across per-image calls.
    Each entry is keyed by filename and contains a list of {class, conf} dicts sorted
    by confidence descending.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(json_path.read_text()) if json_path.exists() else {}

    if isinstance(results, dict) and "predictions" in results:
        # SpeciesNet: {"predictions": [{filepath, prediction, prediction_score, ...}]}
        for item in results["predictions"]:
            filename = Path(item["filepath"]).name
            label = short_speciesnet_label(item.get("prediction"))
            score = item.get("prediction_score")
            if label == "unknown" or score is None:
                records[filename] = []
            else:
                records[filename] = [{"class": label, "conf": round(float(score), 3)}]
    else:
        # Ultralytics: list of Results objects (YOLO26 or YOLOE)
        for r in results:
            boxes = r.boxes
            if not boxes or len(boxes) == 0:
                records[Path(r.path).name] = []
            else:
                records[Path(r.path).name] = [
                    {"class": r.names[int(cls)], "conf": round(float(conf), 3)}
                    for cls, conf in sorted(
                        zip(boxes.cls, boxes.conf), key=lambda x: -x[1]
                    )
                ]

    json_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    # Example experiment block. The functions above can be imported by notebooks or
    # other scripts, while this block lets the file be run directly for ad-hoc tests.
    # IMAGE_DIR = Path(
    # smartrodent/irodent/rodent/images/"
    # )
    IMAGE_DIR = Path(
        "/home/hmack/Development/rodent_experiments/datasets/biotrove-central-europe/filtered"
    )

    for conf in [0.01, 0.05, 0.1, 0.2]:
        confstr = str(conf).translate(str.maketrans("", "", string.punctuation))

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/yolo26/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     results = []
        #     for img in imgs:
        #         res = detect(
        #             img,
        #             1,
        #             crop=True,
        #             project=out,
        #             conf=conf,
        #         )
        #         results.append(res)

        #     for res in results:
        #         write_detections_json(res, out / "detections.json")

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/yoloe_animalornot/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     results = []
        #     for img in imgs:
        #         res = detect_YOLOE(
        #             img,
        #             1,
        #             crop=True,
        #             project=out,
        #             classes=[
        #                 "This is a photo of an animal like a mouse, rat, cat, fox, or hamster",
        #                 "This is not a photo an animal at all, but a photo of something else entirely like trash, a car, a plastic box or a piece of wood or a leaf",
        #             ],
        #             conf=conf,
        #         )
        #         results.append(res)

        #     for res in results:
        #         write_detections_json(res, out / "detections.json")

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/yoloe_rodentornot/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     results = []
        #     for img in imgs:
        #         res = detect_YOLOE(
        #             img,
        #             1,
        #             crop=True,
        #             project=out,
        #             classes=[
        #                 "This is a photo of a rodent like a rat, mouse or vole, or similar small mammal like a shrew",
        #                 "This is a photo of not a rodent at all, but some other animal like a snake, bird or human ",
        #             ],
        #             conf=conf,
        #         )
        #         results.append(res)

        #     for res in results:
        #         write_detections_json(res, out / "detections.json")

        for imgpath in IMAGE_DIR.iterdir():
            name = imgpath.name
            out = Path(
                f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/speciesnet/central-europe/{name}"
            )
            imgs = sorted(imgpath.iterdir())
            out.mkdir(parents=True, exist_ok=True)

            speciesnet_results = run_speciesnet(
                imgs,
                output_json=out / "predictions.json",
                preview_dir=out / "boxed",
                batch_size=16,
                country="DEU",
                model_name="kaggle:google/speciesnet/pyTorch/v4.0.3b/1",
                crop_dir=out / "crops",
                conf=conf,
            )

            write_detections_json(speciesnet_results, out / "detections.json")

            for item in speciesnet_results["predictions"]:
                print(
                    f"{Path(item['filepath']).name}: "
                    f"{short_speciesnet_label(item.get('prediction'))} "
                    f"({item.get('prediction_score', 0):.2f})"
                )

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/biotrove-clip_rodent_or_not/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     biotrove_clip_results = detect_biotrove_clip(
        #         imgs,
        #         1,
        #         "runs/biotrove-clip",
        #         crop=False,
        #         conf=conf,
        #         model_checkpoint="biotroveclip-vit-b-16-from-openai-epoch-40.pt",
        #         classes=[
        #             "a rodent like a rat, mouse or vole, or similar small mammal like a shrew",
        #             "not a rodent at all, but some other animal like a snake, bird or human ",
        #         ],
        #         prompt_template="This is a photo of {}",
        #     )

        #     write_detections_json(biotrove_clip_results, out / "detections.json")

        #     for item in biotrove_clip_results["predictions"]:
        #         if item is None:
        #             continue
        #         print(
        #             f"{Path(item['filepath']).name}: "
        #             f"{short_speciesnet_label(item.get('prediction'))} "
        #             f"({item.get('prediction_score', 0):.2f})"
        #         )

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/biotrove-clip_animalornot/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     biotrove_clip_results = detect_biotrove_clip(
        #         imgs,
        #         1,
        #         "runs/biotrove-clip",
        #         crop=False,
        #         conf=conf,
        #         model_checkpoint="biotroveclip-vit-b-16-from-openai-epoch-40.pt",
        #         classes=[
        #             "an animal like a mouse, rat, cat, fox, or hamster",
        #             "not an animal at all, but a photo of something else entirely like trash, a car, a plastic box or a piece of wood or a leaf",
        #         ],
        #         prompt_template="This is a photo of {}",
        #     )

        #     write_detections_json(biotrove_clip_results, out / "detections.json")

        #     for item in biotrove_clip_results["predictions"]:
        #         if item is None:
        #             continue
        #         print(
        #             f"{Path(item['filepath']).name}: "
        #             f"{short_speciesnet_label(item.get('prediction'))} "
        #             f"({item.get('prediction_score', 0):.2f})"
        #         )

        # classes_central_europe_species = [
        #     "Rattus norvegicus",
        #     "Rattus rattus",
        #     "Mus musculus",
        #     "Myodes glareolus",
        #     "Apodemus agrarius",
        #     "Apodemus flavicollis",
        #     "Apodemus sylvaticus",
        #     "Microtus arvalis",
        #     "Microtus agrestis",
        #     "Crocidura leucodon",
        # ]

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/biotrove-clip_species_names/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     biotrove_clip_results = detect_biotrove_clip(
        #         imgs,
        #         1,
        #         "runs/biotrove-clip",
        #         crop=False,
        #         conf=conf,
        #         model_checkpoint="biotroveclip-vit-b-16-from-openai-epoch-40.pt",
        #         classes=classes_central_europe_species,
        #         prompt_template="This is a photo of {}",
        #     )

        #     write_detections_json(biotrove_clip_results, out / "detections.json")

        #     for item in biotrove_clip_results["predictions"]:
        #         if item is None:
        #             continue
        #         print(
        #             f"{Path(item['filepath']).name}: "
        #             f"{short_speciesnet_label(item.get('prediction'))} "
        #             f"({item.get('prediction_score', 0):.2f})"
        #         )

        # classes_central_europe_common = [
        #     "brown rat",
        #     "black rat",
        #     "house mouse",
        #     "bank vole",
        #     "striped field mouse",
        #     "yellow-necked mouse",
        #     "wood mouse",
        #     "common vole",
        #     "field vole",
        #     "bicolored white-toothed shrew",
        # ]

        # for imgpath in IMAGE_DIR.iterdir():
        #     name = imgpath.name
        #     out = Path(
        #         f"/home/hmack/Development/rodent_experiments/runs/detect{confstr}/biotrove-clip_common_names/central-europe/{name}"
        #     )
        #     imgs = sorted(imgpath.iterdir())
        #     out.mkdir(parents=True, exist_ok=True)

        #     biotrove_clip_results = detect_biotrove_clip(
        #         imgs,
        #         1,
        #         "runs/biotrove-clip",
        #         crop=False,
        #         conf=conf,
        #         model_checkpoint="biotroveclip-vit-b-16-from-openai-epoch-40.pt",
        #         classes=classes_central_europe_common,
        #         prompt_template="This is a photo of {}",
        #     )

        #     write_detections_json(biotrove_clip_results, out / "detections.json")

        #     for item in biotrove_clip_results["predictions"]:
        #         if item is None:
        #             continue
        #         print(
        #             f"{Path(item['filepath']).name}: "
        #             f"{short_speciesnet_label(item.get('prediction'))} "
        #             f"({item.get('prediction_score', 0):.2f})"
        #         )
