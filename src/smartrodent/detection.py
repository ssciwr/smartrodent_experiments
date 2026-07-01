"""Command and library entry points for SmartRodent model experiments.

This file intentionally contains the high-level workflow only: loading models,
choosing inputs and outputs, and calling the helper functions in ``utils.py``.
Keeping the lower-level image/path/label utilities separate makes this module easier
to read and also lets other scripts import those helpers from the package.
"""

import json
from pathlib import Path

from speciesnet import DEFAULT_MODEL
from speciesnet import SpeciesNet
from ultralytics import YOLO, YOLOE
import string
import open_clip
from PIL import Image
import torch
from tqdm import tqdm

from abc import abstractmethod, ABC

try:
    # Package import path, used when running with ``python -m smartrodent.detection`` or
    # importing ``smartrodent`` from another script.
    from .utils import image_paths, COUNTRY_ALIASES, extract_crop
except ImportError:  # pragma: no cover - convenience fallback for direct script runs.
    from utils import image_paths, COUNTRY_ALIASES, extract_crop


from PIL import ImageDraw
from PIL import ImageFont
from speciesnet import draw_bboxes


class DetectorBase(ABC):
    """Common interface and shared helpers for all detector backends.

    Subclasses wrap very different model APIs (Ultralytics YOLO, SpeciesNet, and
    BioTrove-CLIP) behind one ``detect(path, out)`` method. The base class stores
    the experiment settings that are common across backends and provides helpers for
    writing a normalized ``detections.json`` file.
    """

    def __init__(
        self,
        name: str,
        crop: bool,
        batchsize: int,
        img_outname: str,
        conf: float,
        # YOLO specific
        project: str | None = None,
        # speciesnet specific
        crop_dir: str | Path | None = None,
        country: str | None = None,
        admin1_region: str | None = None,
        model_name: str = DEFAULT_MODEL,
        resume: bool = False,
        # yoloe
        task: str = "detect",
        classes: list[str] | None = None,
        # biotrove-clip
        # model_checkpoint: str = "biotroveclip-vit-b-16-from-openai-epoch-40.pt",
        prompt_template: str = "This is a photo of {}",
        model_dir: str | Path | None = None,
    ):
        """Store detector configuration for a single experiment run.

        Parameters are intentionally broad because the experiment JSON feeds several
        detector types through the same constructor. Each subclass only uses the
        fields relevant to its backend; unused fields are kept here so configs can be
        swapped between experiments without changing the orchestration code.
        """
        # all
        self.name = name
        self.crop = crop
        self.batchsize = batchsize
        self.img_outname = img_outname
        self.conf = conf
        self.model = None
        self.model_name = model_name

        # yolo
        self.project = project

        # speciesnet
        self.crop_dir = crop_dir
        self.country = country
        self.admin1_region = admin1_region
        self.resume = resume

        # yoloe
        self.task = task
        self.classes = classes  # also used for biotrove-clip

        # biotrove-clip
        self.prompt_template = prompt_template
        self.model_dir = model_dir

    def write_detections_json(self, results, json_path: Path | str) -> None:
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
                label = self.shorten_label(item.get("prediction"))
                score = item.get("prediction_score")
                if label == "unknown" or score is None:
                    records[filename] = []
                else:
                    records[filename] = [
                        {"class": label, "conf": round(float(score), 3)}
                    ]
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

    def shorten_label(self, item):
        """Return the display label used in ``detections.json``.

        Most detectors already emit short human-readable labels, so the base
        implementation returns the value unchanged. SpeciesNet overrides this because
        it emits semicolon-delimited taxonomy strings.
        """
        return item

    def resolve_local_model(self, model_name: str | Path) -> str:
        """Resolve bundled model weights relative to this file when present."""
        model_path = Path(model_name)
        if model_path.is_absolute():
            return str(model_path)

        bundled_path = Path(__file__).with_name(str(model_name))
        return str(bundled_path) if bundled_path.exists() else str(model_name)

    def restore_result_paths(self, results, source_paths):
        """Restore original filenames on Ultralytics batch results.

        Ultralytics converts list-of-path inputs into PIL images internally. That can
        drop PIL ``filename`` metadata and make result paths fall back to image0.jpg,
        image1.jpg, etc. The result order matches the input order, so put the original
        source paths back before writing detections.json.
        """
        if isinstance(source_paths, str | Path):
            return results

        for result, source_path in zip(results, source_paths, strict=False):
            result.path = str(Path(source_path).resolve())

        return results

    def path_batches(self, path: str | Path | list[Path | str]):
        """Yield single-image input or bounded path batches for model inference.

        Ultralytics treats a Python list of paths as in-memory images and can ignore
        the separate ``batch=`` setting for that loader. Chunking before ``predict``
        keeps the real GPU batch bounded by ``self.batchsize``.
        """
        if isinstance(path, str | Path):
            yield path
            return

        if self.batchsize <= 0:
            raise ValueError(f"batchsize must be positive, got {self.batchsize}")

        for start in range(0, len(path), self.batchsize):
            yield path[start : start + self.batchsize]

    @abstractmethod
    def detect(
        self, path: str | Path | list[Path | str], out: Path, *args, **kwargs
    ) -> list:
        """Run the detector on one image, a directory, or a batch of image paths.

        Subclasses should save their model-native outputs under ``out`` when useful,
        call ``write_detections_json`` to update the normalized summary, and return
        the backend's native result object for ad-hoc inspection.
        """
        pass


class YOLO_Detector(DetectorBase):
    """Ultralytics YOLO26 detector wrapper for boxed object detections.

    This backend uses the bundled YOLO26 weights by default, lets Ultralytics save
    preview images/crops, and writes a compact per-image class/confidence summary.
    """

    def detect(self, path, out):
        """Run YOLO26 prediction and write normalized detections.

        Single-image inputs are resolved to absolute paths. Batch inputs are passed as
        a list with ``self.batchsize`` so GPU inference can be faster; after inference
        we restore original filenames on Results objects before serializing JSON.
        """

        self.model = YOLO(self.resolve_local_model(self.model_name), task=self.task)

        all_results = []
        for source_batch in self.path_batches(path):
            # Chunk list inputs ourselves. Ultralytics' list loader treats all list
            # items as one in-memory batch, so passing the full group here can OOM.
            res = self.model.predict(
                source=source_batch,
                batch=len(source_batch),
                save=True,
                project=self.project,
                name=self.name,
                exist_ok=True,
                conf=self.conf,
                save_crop=self.crop,
            )

            res = self.restore_result_paths(res, source_batch)
            all_results.extend(res)
        self.write_detections_json(all_results, out / "detections.json")

        return all_results


class YOLOE_Detector(DetectorBase):
    """Ultralytics YOLOE detector wrapper with optional text prompt classes.

    YOLOE can be run with natural-language class prompts through ``self.classes``.
    The rest of the output handling mirrors ``YOLO_Detector`` so experiments can
    compare YOLO26 and YOLOE using the same ``detections.json`` format.
    """

    def detect(self, path, out):
        """Run YOLO on one image path or a batch of image paths.

        Ultralytics handles saving boxed preview images when ``save=True`` is passed to
        ``model.predict``. Single-image and batch paths are split here only so the batch
        case can use a larger ``batch`` value.
        """

        # Load the local YOLOE model. Resolve the weights relative to this file so
        # imports from outside the package still find the bundled model file.
        model = YOLOE(
            self.resolve_local_model(self.model_name),
            task=self.task,
        )

        if self.classes:
            model.set_classes(
                self.classes,
                model.get_text_pe(self.classes),
            )

        # Optional training call kept here as a reminder/example for later experiments.
        # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        all_results = []
        for source_batch in self.path_batches(path):
            if isinstance(source_batch, str | Path):
                # A single image does not need an explicit batch size. Resolve the path so
                # Ultralytics receives an absolute filename regardless of the caller's cwd.
                res = model.predict(
                    source=str(Path(source_batch).resolve()),
                    save=True,
                    project=self.project,
                    name="boxed",
                    exist_ok=True,
                    conf=self.conf,
                    save_crop=self.crop,
                )

            else:
                # Chunk list inputs ourselves. Ultralytics' list loader treats all list
                # items as one in-memory batch, so passing the full group here can OOM.
                res = model.predict(
                    source=source_batch,
                    batch=len(source_batch),
                    save=True,
                    project=self.project,
                    name="boxed",
                    exist_ok=True,
                    conf=self.conf,
                    save_crop=self.crop,
                )

            res = self.restore_result_paths(res, source_batch)
            all_results.extend(res)
        self.write_detections_json(all_results, out / "detections.json")

        return all_results


class SpeciesNet_Detector(DetectorBase):
    """SpeciesNet detector/classifier/ensemble wrapper.

    SpeciesNet produces richer JSON than YOLO: MegaDetector boxes, classifier scores,
    ensemble predictions, and optional geofencing. This class keeps that native JSON
    as ``predictions.json`` while also creating the shared ``detections.json`` summary
    and optional boxed preview/crop folders.
    """

    def detect(self, path, out):
        """Run SpeciesNet on the supplied images and write previews/crops/JSON.

        ``path`` is normalized through ``image_paths`` because SpeciesNet expects a
        list of file path strings. When ``predictions_json`` is supplied SpeciesNet may
        return ``None`` after writing to disk, so this method reloads the JSON before
        downstream preview and summary generation.
        """
        # SpeciesNet writes JSON predictions to disk, so create the output folder before
        # the model starts. This also makes failures easier to diagnose because the target
        # path is explicit.
        output_json = Path(out) / "predictions.json"
        preview_dir = Path(out) / "boxed"
        crop_dir = Path(out) / "crops"
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # SpeciesNet geofencing expects ISO alpha-3 country codes. The utility accepts a
        # few friendly aliases (for example, "Germany" -> "DEU").
        country = self.normalize_country(self.country)

        # If resume is disabled, remove stale predictions. Otherwise SpeciesNet may skip
        # work by reusing previous JSON that was generated with different settings.
        if output_json.exists() and not self.resume:
            output_json.unlink()

        # Enable all SpeciesNet components: detector, classifier, and ensemble. Geofence
        # is useful for this dataset because location can reduce implausible species.
        self.model = SpeciesNet(self.model_name, geofence=True, components="all")
        self.model.detector.DETECTION_THRESHOLD = self.conf
        predictions = self.model.predict(
            filepaths=image_paths(path),
            country=country,
            admin1_region=self.admin1_region,
            batch_size=self.batchsize,
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
            self.save_speciesnet_previews(predictions, preview_dir)

        # Crops are also optional. When enabled, detections are saved into folders named
        # after detector labels so they can be inspected or used for later experiments.
        if crop_dir is not None:
            self.save_speciesnet_crops(predictions, crop_dir)

        self.write_detections_json(predictions, out / "detections.json")

        return predictions

    def shorten_label(self, prediction: str | None) -> str:
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
        self,
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

    def normalize_country(self, country: str | None) -> str | None:
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

    def save_speciesnet_crops(
        self,
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
        self,
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

            label = self.shorten_label(item.get("prediction"))
            score = item.get("prediction_score")

            text = f"{label}: {score:.2f}" if score is not None else label
            top_species = self.top_speciesnet_species(item)
            species_lines = [
                (
                    f"{idx}. {name}: {species_score:.4f}"
                    if species_score is not None
                    else f"{idx}. {name}"
                )
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


class BioTroveClip_Detector(DetectorBase):
    """BioTrove-CLIP zero-shot whole-image classifier wrapper.

    Unlike YOLO and SpeciesNet, BioTrove-CLIP does not produce bounding boxes. It
    embeds each image and compares it to text prompts from ``self.classes``. Results
    are written in a SpeciesNet-like ``{"predictions": [...]}`` shape so the shared
    JSON writer can still summarize per-image labels and confidences.
    """

    def detect(self, path, out) -> list:
        """Classify images in batches with BioTrove-CLIP text prompts.

        Images that cannot be opened are skipped with a console message. The best
        prompt is accepted only when its softmax score is at least ``self.conf``;
        otherwise the prediction is recorded as empty/unknown in ``detections.json``.
        """

        if self.batchsize <= 0:
            raise ValueError(f"batchsize must be positive, got {self.batchsize}")
        if self.crop:
            raise ValueError(
                "BioTrove-CLIP classifies whole images and cannot save crops."
            )

        if self.classes is None or len(self.classes) == 0:
            raise ValueError("Class prompts must be given for biotrove clip model")

        classes = self.classes
        paths = image_paths(path)

        project = Path(self.project)
        project.mkdir(parents=True, exist_ok=True)

        model_dir = Path("../../models/BioTrove-CLIP").resolve()
        checkpoint_path = Path(self.model_name)
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

        text_tokens = tokenizer(
            [self.prompt_template.format(label) for label in classes]
        ).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        predictions = []
        # range(..., batchsize) plus slicing includes the last incomplete batch.
        for start in range(0, len(paths), self.batchsize):
            batch_paths = paths[start : start + self.batchsize]
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
                prediction = classes[best_idx] if best_score >= self.conf else None
                predictions.append(
                    {
                        "filepath": filename,
                        "prediction": prediction,
                        "prediction_score": (
                            best_score if prediction is not None else None
                        ),
                        "classifications": {
                            "classes": classes,
                            "scores": [float(score) for score in scores],
                        },
                    }
                )

        result = {"predictions": predictions}
        (project / "predictions.json").write_text(json.dumps(result, indent=2))

        self.write_detections_json(result, out / "detections.json")

        return result


## Experimentation driver functions
def conf_to_string(conf: float) -> str:
    """Convert a confidence value into the run-folder suffix used by experiments.

    For example, ``0.05`` becomes ``"005"`` so outputs land under folders such as
    ``runs/detect005/...``. This preserves the historical directory layout.
    """
    return str(conf).translate(str.maketrans("", "", string.punctuation))


def resolve_lookups(value, class_sets: dict[str, list[str]]):
    """Replace {"lookup": "name"} markers with lists from config.class_sets."""
    if isinstance(value, dict):
        if set(value) == {"lookup"}:
            lookup_name = value["lookup"]
            try:
                return class_sets[lookup_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown class set lookup {lookup_name!r}. "
                    f"Available class sets: {sorted(class_sets)}"
                ) from exc
        return {key: resolve_lookups(item, class_sets) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve_lookups(item, class_sets) for item in value]

    return value


def load_experiment_config(config_path: Path) -> dict:
    """Load detector experiment settings and resolve class-set references.

    The JSON file can keep long class lists under ``class_sets`` and refer to them
    from an experiment with ``{"lookup": "class_set_name"}``. This function expands
    those references before the main loop instantiates detector classes.
    """
    with config_path.open() as config_file:
        config = json.load(config_file)

    class_sets = config.get("class_sets", {})
    config["experiments"] = resolve_lookups(config["experiments"], class_sets)
    return config


def image_groups(root: Path):
    """Yield named image batches from immediate subdirectories of ``root``.

    The current dataset layout groups images by folder. Each yielded tuple is
    ``(folder_name, sorted_image_paths)`` and empty/non-directory entries are ignored.
    """
    for imgpath in sorted(root.iterdir()):
        if not imgpath.is_dir():
            continue
        imgs = sorted(imgpath.iterdir())
        if imgs:
            yield imgpath.name, imgs


def run_detector_on_group(
    detector_cls: type[DetectorBase],
    experiment_name: str,
    detector_kwargs: dict,
    conf: float,
    group_name: str,
    imgs: list[Path],
):
    """Instantiate one detector and run it for one confidence/group combination.

    Output paths follow ``RUNS_DIR/detect{conf}/{experiment}/{dataset}/{group}``.
    ``detector_kwargs`` is expected to be a per-call copy because ``batchsize`` is
    popped out before forwarding the remaining backend-specific settings. The detector
    class owns model loading inside its ``detect`` method.
    """
    out = (
        RUNS_DIR
        / f"detect{conf_to_string(conf)}"
        / experiment_name
        / DATASET_NAME
        / group_name
    )
    out.mkdir(parents=True, exist_ok=True)

    detector = detector_cls(
        name="boxed",
        batchsize=detector_kwargs.pop("batchsize", BATCHSIZE),
        img_outname="boxed",
        conf=conf,
        project=out,
        **detector_kwargs,
    )
    return detector.detect(imgs, out)


if __name__ == "__main__":
    # Example experiment block. Detector/run settings live in JSON so new experiments
    # can be added without editing this orchestration logic.
    CONFIG_PATH = (
        Path(__file__).resolve().parents[2] / "configs" / "detector_experiments.json"
    )

    DETECTOR_CLASSES = {
        "YOLO_Detector": YOLO_Detector,
        "YOLOE_Detector": YOLOE_Detector,
        "SpeciesNet_Detector": SpeciesNet_Detector,
        "BioTroveClip_Detector": BioTroveClip_Detector,
    }

    config = load_experiment_config(CONFIG_PATH)
    IMAGE_DIR = Path(config["image_dir"])
    RUNS_DIR = Path(config["runs_dir"])
    DATASET_NAME = config["dataset_name"]
    BATCHSIZE = config["batchsize"]
    confs = config["confs"]
    experiments = config["experiments"]

    for experiment in experiments:
        experiment_name = experiment["name"]
        try:
            detector_cls = DETECTOR_CLASSES[experiment["detector"]]
        except KeyError as exc:
            raise KeyError(
                f"Unknown detector class {experiment['detector']!r}. "
                f"Available detectors: {sorted(DETECTOR_CLASSES)}"
            ) from exc
        base_kwargs = experiment["kwargs"]

        for conf in tqdm(confs, desc=experiment_name):
            for group_name, imgs in image_groups(IMAGE_DIR):
                # Copy kwargs because each detector construction pops generic settings
                # out of the backend-specific dictionary.
                run_detector_on_group(
                    detector_cls=detector_cls,
                    experiment_name=experiment_name,
                    detector_kwargs=dict(base_kwargs),
                    conf=conf,
                    group_name=group_name,
                    imgs=imgs,
                )
