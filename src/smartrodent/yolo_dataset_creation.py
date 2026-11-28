"""Create YOLO detector and classifier datasets from split records and SpeciesNet boxes.

A ``YoloDatasetCreator`` combines three parts: the split records of an input root
(``load_split_records``), a detector that finds the animals (``SpeciesNetDetector``),
and an output layout (``DetectorLayout`` or ``ClassifierLayout``).
"""

import json
import os
import re
import sys
from abc import abstractmethod
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypedDict, runtime_checkable

import pandas as pd
import torch
import yaml
from PIL import Image
from pydantic import BeforeValidator, ConfigDict, Field, validate_call, with_config
from pydantic.dataclasses import dataclass
from speciesnet import DEFAULT_MODEL, SpeciesNet
from torchvision.ops import box_convert, nms
from tqdm import tqdm

from .components import ComponentSpec
from .config_loading import load_section
from .training_validation import (
    ClassNames,
    Count,
    MappingInput,
    Nonblank,
    PathSetting,
    Probability,
    parse_settings,
)

type DetectorLabels = tuple[Nonblank, ...]


# --- records -------------------------------------------------------------------------


def load_split_records(
    path_to_image_data: str | Path, class_names: Sequence[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Read the per-species records of an input root, keeping their splits as given.

    Source row order, additional columns and oversampled (repeated) rows are kept.

    Args:
        path_to_image_data: Input root containing one directory per species, each
            with a ``records.csv`` that has ``image_path`` and ``dataset_split``.
        class_names: Species directories in class-index order; None selects all
            of them in sorted order. Unselected directories are not read.

    Returns:
        Species mapped to their records, in class-index order. ``dataset_split``
        uses ``val`` for validation, and ``image_path`` holds absolute paths;
        relative paths are resolved against the parent of the input root.

    Raises:
        ValueError: The input root, the class selection or a records file is
            invalid, or an image is assigned to more than one split.
        FileNotFoundError: A records file or a referenced image is missing.
    """
    root = Path(path_to_image_data).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory {root} does not exist")
    records_by_species = {
        species: _load_species_records(root, species)
        for species in _select_classes(root, class_names)
    }
    _reject_split_conflicts(records_by_species)
    return records_by_species


def _select_classes(root: Path, class_names: Sequence[str] | None) -> list[str]:
    """Discover classes or validate an explicit ordered directory selection."""
    available = {path.name for path in root.iterdir() if path.is_dir()}
    selected = sorted(available) if class_names is None else list(class_names)
    if not selected:
        raise ValueError("No class directories found")
    unknown = set(selected) - available
    if unknown:
        raise ValueError(f"Unknown class directories: {sorted(unknown)}")
    return selected


def _load_species_records(root: Path, species: str) -> pd.DataFrame:
    """Read a species frame, keeping row order, multiplicity, and extra columns."""
    records_path = root / species / "records.csv"
    if not records_path.is_file():
        raise FileNotFoundError(f"Missing species records: {records_path}")
    try:
        records = pd.read_csv(records_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{species}: records.csv has no column header") from error

    missing = {"image_path", "dataset_split"} - set(records.columns)
    if missing:
        raise ValueError(f"{species}: missing required columns {sorted(missing)}")
    records["dataset_split"] = records["dataset_split"].replace({"validation": "val"})
    if not records["dataset_split"].isin(["train", "val", "test"]).all():
        raise ValueError(
            f"{species}: dataset_split must be train, validation/val, or test"
        )
    if (
        not records["image_path"]
        .map(lambda path: isinstance(path, str) and bool(path.strip()))
        .all()
    ):
        raise ValueError(f"{species}: image_path must contain nonempty path strings")
    records["image_path"] = records["image_path"].map(
        lambda path: _resolve_image_path(path, root.parent)
    )
    return records


def _resolve_image_path(image_path: str, dataset_root: Path) -> str:
    """Resolve shared-root source paths explicitly and reject missing images."""
    path = Path(image_path)
    if not path.is_absolute():
        path = dataset_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image_path does not refer to a file: {path}")
    return str(path)


def _reject_split_conflicts(records_by_species: dict[str, pd.DataFrame]) -> None:
    """Reject images assigned across splits while preserving oversampled rows."""
    assignments = pd.concat(
        [
            records[["image_path", "dataset_split"]]
            for records in records_by_species.values()
        ],
        ignore_index=True,
    )
    splits_per_image = assignments.groupby("image_path")["dataset_split"].nunique()
    conflicting = splits_per_image[splits_per_image > 1]
    if not conflicting.empty:
        raise ValueError(
            f"Images assigned to multiple splits: {conflicting.index.tolist()}"
        )


# --- detection -----------------------------------------------------------------------


@runtime_checkable
class Detector(Protocol):
    """Finds the boxes to keep in source images."""

    @abstractmethod
    def detect(self, image_paths: Sequence[str | Path]) -> dict[str, list[dict]]:
        """Return the accepted detections of each image.

        Args:
            image_paths: Source images, inferred together.

        Returns:
            Full source-path strings mapped to their accepted detections. Each
            detection has a normalized top-left xywh ``bbox``, ``confidence``,
            ``label``, the model's ``detection_index`` and an in-memory PIL
            ``crop``. Images without accepted detections map to empty lists.
        """
        ...


class SpeciesNetDetector(Detector):
    """SpeciesNet's detector, filtered by confidence and label, then by NMS.

    Weights load on the first nonempty ``detect`` call, so creating the detector
    (and every dataset creator that shares it) stays cheap.

    Args:
        model: An existing SpeciesNet-compatible model (Python only).
        model_name: Identifier or local weights directory to load; None selects
            SpeciesNet's default model.
        confidence_threshold: Inclusive minimum detection confidence.
        allowed_classes: Detector labels to keep; an empty tuple keeps none.
        iou_threshold: Overlap above which a weaker box is suppressed.

    Raises:
        ValueError: Both ``model`` and ``model_name`` are given, or a setting is
            invalid.
    """

    @validate_call
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: Nonblank | None = None,
        confidence_threshold: Probability = 0.1,
        allowed_classes: DetectorLabels = ("animal",),
        iou_threshold: Probability = 0.45,
    ):
        if model is not None and model_name is not None:
            raise ValueError("Supply either model or model_name, not both")
        self.model = model
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes
        self.iou_threshold = iou_threshold

    def detect(self, image_paths: Sequence[str | Path]) -> dict[str, list[dict]]:
        """Run SpeciesNet in one call and extract the accepted crops, writing nothing.

        Args:
            image_paths: Source images; the caller bounds how many are passed.

        Returns:
            See ``Detector.detect``. Crops are not JSON-serializable; exclude
            them when serializing.

        Raises:
            ValueError: The model does not return one prediction per image.
        """
        filepaths = [str(Path(path).resolve()) for path in image_paths]
        if not filepaths:
            return {}
        results = {}
        for prediction in self._predict(filepaths):
            source_path = str(Path(prediction["filepath"]).resolve())
            accepted = self._accepted(prediction["detections"])
            results[source_path] = self._crops(source_path, accepted)
        return results

    def _predict(self, filepaths: list[str]) -> list[dict]:
        """Run one detector-only SDK call and verify one prediction per image."""
        if self.model is None:
            self.model = SpeciesNet(
                self.model_name if self.model_name is not None else DEFAULT_MODEL,
                components="detector",
            )
        # The SDK drops boxes below its class-level threshold before returning.
        # Disable that cutoff so every box keeps its original index for filtering
        # and NMS here, and restore it even on failure for other model users.
        detector_type = type(self.model.detector)
        previous_threshold = detector_type.DETECTION_THRESHOLD
        try:
            detector_type.DETECTION_THRESHOLD = 0.0
            predictions = self.model.detect(filepaths=filepaths)["predictions"]
        finally:
            detector_type.DETECTION_THRESHOLD = previous_threshold

        predicted_paths = [
            str(Path(item["filepath"]).resolve()) for item in predictions
        ]
        if sorted(predicted_paths) != sorted(filepaths):
            raise ValueError("SpeciesNet must return one prediction per input image")
        return predictions

    def _accepted(self, detections: list[dict]) -> list[tuple[int, dict]]:
        """Filter by confidence and label, then NMS, keeping the original indices."""
        # Original indices matter: crop filenames include the detection index.
        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection["conf"] >= self.confidence_threshold
            and detection["label"] in self.allowed_classes
        ]
        if not candidates:
            return []
        boxes = box_convert(
            torch.tensor([detection["bbox"] for _, detection in candidates]),
            in_fmt="xywh",
            out_fmt="xyxy",
        )
        scores = torch.tensor([detection["conf"] for _, detection in candidates])
        # Greedy NMS: kept boxes come back in decreasing confidence order.
        kept = nms(boxes, scores, self.iou_threshold)
        return [candidates[position] for position in kept.tolist()]

    @staticmethod
    def _crops(source_path: str, accepted: list[tuple[int, dict]]) -> list[dict]:
        """Build metadata and independent crops for an image's accepted detections."""
        if not accepted:
            return []

        with Image.open(source_path) as source:
            image = source.convert("RGB")

        records = []
        for index, detection in accepted:
            x, y, width, height = detection["bbox"]
            # PIL crops use pixel xyxy coordinates, not normalized xywh.
            crop = image.crop(
                (
                    x * image.width,
                    y * image.height,
                    (x + width) * image.width,
                    (y + height) * image.height,
                )
            )
            records.append(
                {
                    "bbox": list(detection["bbox"]),
                    "confidence": detection["conf"],
                    "label": detection["label"],
                    "detection_index": index,
                    "crop": crop,
                }
            )
        return records


# --- layouts -------------------------------------------------------------------------


@runtime_checkable
class DatasetLayout(Protocol):
    """How a dataset's files are arranged and written."""

    @abstractmethod
    def create_directories(self, root: Path, class_names: Sequence[str]) -> None:
        """Create the split directories, including empty ones."""
        ...

    @abstractmethod
    def write(
        self,
        root: Path,
        species: str,
        class_index: int,
        row: dict,
        stem: str,
        detections: list[dict],
    ) -> list[dict]:
        """Write one source occurrence and return its outputs' relative paths."""
        ...

    @abstractmethod
    def training_paths(self) -> dict[str, str]:
        """Return the train, val and test directories that ``data.yaml`` names."""
        ...


class DetectorLayout(DatasetLayout):
    """Source images symlinked into ``images/<split>`` with YOLO box labels."""

    def create_directories(self, root: Path, class_names: Sequence[str]) -> None:
        """Create parallel images and labels directories for every split."""
        for split in ["train", "val", "test"]:
            for directory in ["images", "labels"]:
                (root / directory / split).mkdir(parents=True)

    def write(
        self,
        root: Path,
        species: str,
        class_index: int,
        row: dict,
        stem: str,
        detections: list[dict],
    ) -> list[dict]:
        """Symlink a full source image and write its normalized YOLO box labels."""
        source = Path(row["image_path"])
        split = row["dataset_split"]
        image_path = Path("images") / split / f"{stem}{source.suffix}"
        label_path = Path("labels") / split / f"{stem}.txt"
        destination = root / image_path
        # Filesystem links are relative to their own directory, unlike CSV paths.
        destination.symlink_to(os.path.relpath(source, destination.parent))
        labels = []
        for detection in detections:
            x, y, width, height = detection["bbox"]
            # SpeciesNet uses normalized top-left xywh; YOLO expects center xywh.
            labels.append([class_index, x + width / 2, y + height / 2, width, height])
        (root / label_path).write_text(
            "".join(" ".join(map(str, label)) + "\n" for label in labels)
        )
        return [{"image_path": str(image_path), "label_path": str(label_path)}]

    def training_paths(self) -> dict[str, str]:
        """Return the detector image directories used by Ultralytics."""
        return {split: f"images/{split}" for split in ["train", "val", "test"]}


class ClassifierLayout(DatasetLayout):
    """One saved crop per accepted detection, in ``<split>/<class>``."""

    def create_directories(self, root: Path, class_names: Sequence[str]) -> None:
        """Create each class directory in each split, including empty ones."""
        for split in ["train", "val", "test"]:
            for species in class_names:
                (root / split / species).mkdir(parents=True)

    def write(
        self,
        root: Path,
        species: str,
        class_index: int,
        row: dict,
        stem: str,
        detections: list[dict],
    ) -> list[dict]:
        """Save independent, lossless crops and retain their detection provenance."""
        outputs = []
        for detection in detections:
            index = detection["detection_index"]
            image_path = (
                Path(row["dataset_split"]) / species / f"{stem}_det{index:03d}.png"
            )
            # PNG avoids introducing another lossy compression pass into crops.
            # PIL carries the source JPEG's ICC profile into the crop, and
            # malformed camera/phone profiles make libpng warn on every decode
            # during training. Dropping the profile instead of converting to sRGB
            # keeps the stored pixel values unchanged, which is also what OpenCV
            # (no colour management) has always fed to the model.
            detection["crop"].save(root / image_path, icc_profile=None)
            outputs.append({"image_path": str(image_path), "detection_index": index})
        return outputs

    def training_paths(self) -> dict[str, str]:
        """Return the classifier split directories used by Ultralytics."""
        return {split: split for split in ["train", "val", "test"]}


# --- creation ------------------------------------------------------------------------


def _path_component(value: str) -> str:
    """Return a readable value that is safe to use as a directory or file name."""
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return value or "unknown"


class YoloDatasetCreator:
    """Write one YOLO dataset from an input root's split records.

    Records are read and checked when the creator is built; inference and all
    writing wait for ``create``. Existing split assignments and row multiplicity
    are preserved.

    Args:
        path_to_image_data: Input root with per-species ``records.csv`` files.
        dataset_output_path: Destination of the dataset; must not exist yet.
        layout: How the dataset's files are arranged.
        detector: Finds the boxes; may be shared by several creators.
        class_names: Species in class-index order; None selects all in sorted order.
        batch_size: Source rows per inference call, which bounds the crops in memory.

    Raises:
        ValueError: A setting, the class selection or the records are invalid.
        FileNotFoundError: Records or source images are missing.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        path_to_image_data: PathSetting,
        dataset_output_path: PathSetting,
        *,
        layout: DatasetLayout,
        detector: Detector,
        class_names: ClassNames | None = None,
        batch_size: Count = 32,
    ):
        self.path_to_image_data = Path(path_to_image_data).resolve()
        self.dataset_output_path = Path(dataset_output_path).resolve()
        self.records_by_species = load_split_records(
            self.path_to_image_data, class_names
        )
        self.class_names = list(self.records_by_species)
        self.layout = layout
        self.detector = detector
        self.batch_size = batch_size

    def create(self) -> Path:
        """Infer the source records and write the dataset.

        Images with no accepted detections are skipped, with the reason in
        ``metadata.json``. The metadata keeps source columns, detection details,
        and output paths relative to the dataset root, never the crops.

        Returns:
            The dataset's root directory.

        Raises:
            FileExistsError: The destination already exists.
            OSError: Source images cannot be read or outputs cannot be written.
            ValueError: The detector returns invalid predictions or the output
                directory overlaps the input.
        """
        root = self.dataset_output_path
        if root.is_relative_to(self.path_to_image_data) or (
            self.path_to_image_data.is_relative_to(root)
        ):
            raise ValueError("Dataset output must not overlap the input directory")
        root.mkdir(parents=True, exist_ok=False)
        self.layout.create_directories(root, self.class_names)
        metadata = [
            record
            for species, records in self.records_by_species.items()
            for record in self._create_species(species, records)
        ]
        data = {
            "path": str(root),
            **self.layout.training_paths(),
            "names": dict(enumerate(self.class_names)),
        }
        (root / "data.yaml").write_text(yaml.safe_dump(data))
        (root / "metadata.json").write_text(
            json.dumps({"records": metadata}, allow_nan=False, indent=2)
        )
        return root

    def _create_species(self, species: str, records: pd.DataFrame) -> list[dict]:
        """Infer bounded row batches and write each occurrence in source-row order."""
        metadata = []
        # Bound the number of live crops, not just SDK calls. Inferring the whole
        # dataset first would retain every crop in memory. Duplicate sources within
        # a batch share inference, but each oversampled row still produces an output.
        for start in tqdm(
            range(0, len(records), self.batch_size),
            desc=f"Create species records for {species}",
        ):
            frame = records.iloc[start : start + self.batch_size]
            detections_by_path = self.detector.detect(
                frame["image_path"].drop_duplicates().tolist()
            )
            # Convert missing values to JSON null without a pandas JSON round-trip,
            # which would round high-precision metadata. Keep the source frame intact.
            rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
            for offset, row in enumerate(rows):
                metadata.append(
                    self._write_source_record(
                        species,
                        start + offset,
                        row,
                        detections_by_path[row["image_path"]],
                    )
                )
        return metadata

    def _write_source_record(
        self, species: str, source_row: int, row: dict, detections: list[dict]
    ) -> dict:
        """Write one input occurrence and record its outputs or explicit skip reason."""
        metadata = {
            **row,
            "species": species,
            "source_row": source_row,
            "detections": [
                {key: value for key, value in detection.items() if key != "crop"}
                for detection in detections
            ],
        }
        if not detections:
            metadata.update(
                status="skipped", skip_reason="no_accepted_detections", outputs=[]
            )
        else:
            outputs = self.layout.write(
                self.dataset_output_path,
                species,
                self.class_names.index(species),
                row,
                self._output_stem(species, row["image_path"], source_row),
                detections,
            )
            metadata.update(status="written", skip_reason=None, outputs=outputs)
        return metadata

    @staticmethod
    def _output_stem(species: str, image_path: str, source_row: int) -> str:
        """Name sources independently of output root, without basename collisions."""
        # Full path plus class distinguishes same-named sources, and row position
        # preserves intentional oversampling. Limit the readable stem so long
        # source filenames still fit normal filesystem component-length limits.
        identity = sha256(json.dumps([species, image_path]).encode()).hexdigest()
        readable_stem = _path_component(Path(image_path).stem)[:80]
        return f"{readable_stem}_{identity}_row{source_row:08d}"


# --- configuration -------------------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
class _DatasetPaths(TypedDict):
    """Input records and output root of one dataset."""

    path_to_image_data: PathSetting
    dataset_output_path: PathSetting


def _build_detector(spec) -> "Detector":
    """Build a configured detector when the configuration is parsed.

    Building is cheap (weights load on first use), and a bad name or parameter
    then fails before any records are read.
    """
    spec = parse_settings(MappingInput[ComponentSpec], spec)
    if "model" in spec.parameters:
        raise ValueError("model is a Python-only setting; configure model_name")
    return spec.build(sys.modules[__name__], Detector)


@dataclass(frozen=True, config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DatasetCreationPlan:
    """Datasets created with one shared detector, layout and class selection.

    Args:
        detector: ``{name, parameters}`` of the detector class (in this module, or
            ``module:attr``), built when the plan is parsed; one instance serves
            every dataset, so weights load once. ``model`` is Python-only.
        datasets: Input and output directories of each dataset, in creation order.
        layout: ``detector`` (symlinked images and box labels) or ``classifier``
            (saved crops per class).
        class_names: Species in class-index order; None selects all in sorted order.
        batch_size: Source rows per inference call.
    """

    detector: Annotated[Detector, BeforeValidator(_build_detector)]
    datasets: Annotated[tuple[MappingInput[_DatasetPaths], ...], Field(min_length=1)]
    layout: Literal["detector", "classifier"]
    class_names: ClassNames | None = None
    batch_size: Count = 32

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetCreationPlan":
        """Read a dataset-creation configuration.

        Args:
            path: YAML file; paths inside are relative to the working directory.

        Returns:
            The validated plan with its detector built; no records are read and
            no weights are loaded yet.

        Raises:
            ValueError: A setting, including the detector, is invalid.
        """
        return parse_settings(cls, load_section(path))

    def creators(self) -> list[YoloDatasetCreator]:
        """Build one creator per dataset; each reads and checks its records, nothing else.

        Returns:
            Creators in ``datasets`` order, sharing one detector with unloaded weights.

        Raises:
            ValueError: Any dataset's records are invalid.
        """
        layouts = {"detector": DetectorLayout, "classifier": ClassifierLayout}
        layout = layouts[self.layout]()
        return [
            YoloDatasetCreator(
                **paths,
                layout=layout,
                detector=self.detector,
                class_names=self.class_names,
                batch_size=self.batch_size,
            )
            for paths in self.datasets
        ]

    def create(self) -> list[Path]:
        """Create every dataset after preparing all creators.

        Every creator reads and checks its records first, so an invalid later
        dataset is reported before the first one is written.

        Returns:
            Root directory of each created dataset.
        """
        return [creator.create() for creator in self.creators()]
