"""Build YOLO detection and classification datasets from SpeciesNet output."""

from collections import defaultdict
from copy import deepcopy
import json
from math import ceil, floor
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from .dataset_creation_base import YoloDatasetCreatorBase


def _path_component(value: str) -> str:
    """Return a value safe for use as one filesystem path component."""
    normalized = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return normalized or "unknown"


class YoloDetectorDatasetCreatorFromSpeciesnet(YoloDatasetCreatorBase):
    """Convert SpeciesNet predictions into an Ultralytics detection dataset."""

    def __init__(
        self,
        path_to_image_data: str | Path,
        path_to_labels: str | Path,
        dataset_output_path: str | Path,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.45,
        labels_to_filter: list[str] | None = None,
        background_image_dir: str | Path | None = None,
        background_class_names: list[str] | tuple[str, ...] = ("empty",),
    ) -> None:
        """Initialize detection dataset creation.

        Args:
            path_to_image_data: Root containing one source-image directory per class.
            path_to_labels: Root containing per-class ``predictions.json`` files.
            dataset_output_path: Destination for the generated YOLO dataset.
            class_names: Ordered source class names. Background names are excluded
                from detector metadata.
            train_val_test_split: Train, validation, and test fractions.
            rng_seed: Seed used for deterministic splits.
            confidence_threshold: Minimum accepted SpeciesNet confidence.
            iou_threshold: Maximum Intersection over Union (IoU) allowed between
                two retained boxes. During non-maximum suppression (NMS), a
                lower-confidence box is discarded when its overlap with an already
                retained box is greater than this value. IoU ranges from 0 for no
                overlap to 1 for identical boxes.
            labels_to_filter: SpeciesNet detector labels to retain.
            background_image_dir: Optional directory of negative images.
            background_class_names: Names treated as background rather than detector
                classes.
        """
        self.background_class_names = set(background_class_names)
        detection_classes = [
            name for name in class_names if name not in self.background_class_names
        ]
        super().__init__(
            path_to_image_data=path_to_image_data,
            path_to_labels=path_to_labels,
            dataset_output_path=dataset_output_path,
            class_names=detection_classes,
            train_val_test_split=train_val_test_split,
            rng_seed=rng_seed,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        self.allowed_classes = set(labels_to_filter or ["animal"])
        self.background_image_dir = (
            Path(background_image_dir) if background_image_dir is not None else None
        )
        (self.dataset_output_path / "images").mkdir(exist_ok=True)
        (self.dataset_output_path / "labels").mkdir(exist_ok=True)
        self.labels = self._load_speciesnet_predictions()

    def _load_speciesnet_predictions(self) -> dict[str, list[dict[str, Any]]]:
        """Merge predictions from immediate class directories.

        Returns:
            A dictionary containing one combined ``predictions`` list.

        Raises:
            FileNotFoundError: If no predictions are found.
            ValueError: If a prediction file does not contain a list.
        """
        predictions: list[dict[str, Any]] = []
        for class_dir in sorted(path for path in self.path_to_labels.iterdir() if path.is_dir()):
            predictions_path = class_dir / "predictions.json"
            if not predictions_path.exists():
                continue
            contents = json.loads(predictions_path.read_text())
            class_predictions = contents.get("predictions")
            if not isinstance(class_predictions, list):
                raise ValueError(f"Invalid predictions list in {predictions_path}")
            predictions.extend(class_predictions)

        if not predictions:
            raise FileNotFoundError(
                "No SpeciesNet predictions found under species folders in "
                f"{self.path_to_labels}"
            )
        return {"predictions": predictions}

    @staticmethod
    def _intersection_over_union(first: list[float], second: list[float]) -> float:
        """Calculate overlap between two normalized SpeciesNet boxes.

        SpeciesNet represents each box as ``[x_min, y_min, width, height]``. IoU is
        the area shared by both boxes divided by their combined area. It is 0 when
        boxes do not overlap and 1 when they are identical.
        """
        # Convert width/height into the bottom-right edge coordinates needed to find
        # the rectangle shared by both boxes. Coordinates remain normalized, but the
        # ratio is the same as it would be in image pixels.
        first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
        second_x2, second_y2 = second[0] + second[2], second[1] + second[3]

        # The overlap starts at the later top-left edge and ends at the earlier
        # bottom-right edge. Clamping to zero handles disjoint boxes.
        intersection_width = max(
            0.0, min(first_x2, second_x2) - max(first[0], second[0])
        )
        intersection_height = max(
            0.0, min(first_y2, second_y2) - max(first[1], second[1])
        )
        intersection = intersection_width * intersection_height

        # Adding both areas counts their overlap twice, so subtract it once. A
        # zero-area union is treated as no overlap instead of dividing by zero.
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return intersection / union if union > 0 else 0.0

    def _filter_label_items(self, detections: list[dict]) -> list[tuple[int, dict]]:
        """Apply label, confidence, and NMS filters while retaining source indices.

        NMS removes likely duplicate detections of one animal. Candidates are checked
        from highest to lowest confidence. A candidate survives only when its IoU with
        every previously retained, higher-confidence box is at most
        ``self.iou_threshold``.
        """
        candidates = [
            (index, deepcopy(item))
            for index, item in enumerate(detections)
            if item.get("conf", item.get("confidence", 0.0))
            >= self.confidence_threshold
            and item.get("label", item.get("class")) in self.allowed_classes
        ]
        # Confidence ordering is what makes this non-maximum suppression: when two
        # boxes overlap too much, the strongest prediction is encountered and kept
        # first, while the weaker prediction is rejected below.
        candidates.sort(
            key=lambda pair: pair[1].get("conf", pair[1].get("confidence", 0.0)),
            reverse=True,
        )

        retained: list[tuple[int, dict]] = []
        for candidate in candidates:
            bbox = candidate[1].get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("SpeciesNet detections require a four-value bbox")
            # Keep boxes that are sufficiently distinct from every stronger box.
            # For example, with a threshold of 0.2, overlap greater than 20% of the
            # boxes' combined area causes this lower-confidence candidate to be
            # treated as a duplicate and omitted.
            if all(
                self._intersection_over_union(bbox, kept[1]["bbox"])
                <= self.iou_threshold
                for kept in retained
            ):
                retained.append(candidate)
        return retained

    def _filter_labels(self, detections: list[dict]) -> list[dict]:
        """Return accepted detection dictionaries without source indices."""
        return [item for _, item in self._filter_label_items(detections)]

    @staticmethod
    def _is_primary_observation(path: Path) -> bool:
        """Return whether a filename is the first observation in a sequence."""
        components = path.name.split("_")
        return len(components) == 1 or components[1].startswith("0")

    def _filter_by_observation(self, labels: dict) -> dict:
        """Keep only first observations to reduce near-duplicate leakage."""
        return {
            "predictions": [
                item
                for item in labels["predictions"]
                if self._is_primary_observation(Path(item["filepath"]))
            ]
        }

    def _preprocess_labels(self, raw_labels: dict) -> dict[str, list[dict]]:
        """Convert accepted SpeciesNet boxes into normalized YOLO boxes."""
        labels_by_image: dict[str, list[dict]] = {}
        for prediction in raw_labels["predictions"]:
            source_path = Path(prediction["filepath"])
            species = source_path.parent.name
            if species in self.background_class_names:
                continue
            converted = []
            for item in self._filter_labels(prediction.get("detections", [])):
                x_min, y_min, width, height = item["bbox"]
                converted.append(
                    {
                        "bbox": [
                            x_min + width / 2,
                            y_min + height / 2,
                            width,
                            height,
                        ],
                        "label": species,
                    }
                )
            labels_by_image[source_path.name] = converted
        return labels_by_image

    def _background_image_paths(self) -> list[Path]:
        """Return configured background images."""
        if self.background_image_dir is None:
            return []
        if not self.background_image_dir.is_dir():
            raise ValueError(
                f"Background image directory {self.background_image_dir} does not exist"
            )
        return sorted(
            path
            for path in self.background_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.img_types
        )

    def _split_paths(self, paths: list[Path]) -> dict[str, list[Path]]:
        """Shuffle and split paths according to configured fractions."""
        paths = list(paths)
        self.rng.shuffle(paths)
        train_count = int(ceil(len(paths) * self.train_frac))
        validation_count = int(floor(len(paths) * self.val_frac))
        return {
            "train": paths[:train_count],
            "val": paths[train_count : train_count + validation_count],
            "test": paths[train_count + validation_count :],
        }

    def _split_train_val_test(
        self, processed_labels: dict[str, list[dict]]
    ) -> dict[str, list[Path]]:
        """Copy retained source and background images into detector splits."""
        assignments: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        retained_species = {
            image_name: {item["label"] for item in items}
            for image_name, items in processed_labels.items()
            if items
        }

        for split in assignments:
            (self.dataset_output_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        for class_dir in sorted(
            path for path in self.path_to_image_data.iterdir() if path.is_dir()
        ):
            if class_dir.name in self.background_class_names:
                continue
            source_paths = [
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.img_types
                and class_dir.name in retained_species.get(path.name, set())
            ]
            for split, split_paths in self._split_paths(source_paths).items():
                for source in split_paths:
                    destination = self.dataset_output_path / "images" / split / source.name
                    shutil.copy2(source, destination)
                    assignments[split].append(destination)

        for split, split_paths in self._split_paths(self._background_image_paths()).items():
            for source in split_paths:
                destination = self.dataset_output_path / "images" / split / source.name
                shutil.copy2(source, destination)
                assignments[split].append(destination)
        return assignments

    def _write_labels(
        self,
        assignments: dict[str, list[Path]],
        preprocessed_labels: dict[str, list[dict]],
    ) -> Path:
        """Write YOLO labels and dataset metadata."""
        class_indices = {name: index for index, name in self.classes.items()}
        for split, image_paths in assignments.items():
            for image_path in image_paths:
                rows = []
                for item in preprocessed_labels.get(image_path.name, []):
                    if item["label"] not in class_indices:
                        raise ValueError(f"Unknown class in predictions: {item['label']}")
                    rows.append([class_indices[item["label"]], *item["bbox"]])
                label_path = self.dataset_output_path / "labels" / split / f"{image_path.stem}.txt"
                label_path.write_text(
                    "".join(" ".join(map(str, row)) + "\n" for row in rows)
                )

        metadata = {
            "path": str(self.dataset_output_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.classes,
        }
        (self.dataset_output_path / "data.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False)
        )
        return self.dataset_output_path

    def __call__(self) -> Path:
        """Generate the detection dataset and return its output directory."""
        labels = self._filter_by_observation(self.labels)
        preprocessed = self._preprocess_labels(labels)
        assignments = self._split_train_val_test(preprocessed)
        return self._write_labels(assignments, preprocessed)


class YoloClassifierDatasetCreatorFromSpeciesnet(
    YoloDetectorDatasetCreatorFromSpeciesnet
):
    """Build an Ultralytics classification dataset from SpeciesNet crops."""

    def __init__(
        self,
        path_to_image_data: str | Path,
        dataset_output_path: str | Path,
        class_names: list[str],
        path_to_labels: str | Path | None = None,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.45,
        labels_to_filter: list[str] | None = None,
        image_directory: str = "crops",
        background_image_dir: str | Path | None = None,
        background_class_name: str = "empty",
    ) -> None:
        """Initialize classification dataset creation.

        Args:
            path_to_image_data: SpeciesNet output root with one directory per class.
            dataset_output_path: Destination classification dataset root.
            class_names: Classification directory names to include.
            path_to_labels: Optional separate prediction root.
            train_val_test_split: Train, validation, and test fractions.
            rng_seed: Seed used for deterministic source-image splitting.
            confidence_threshold: Minimum accepted SpeciesNet confidence.
            iou_threshold: Maximum Intersection over Union (IoU) allowed between
                retained crop detections. A lower-confidence detection is discarded
                as a likely duplicate when its overlap exceeds this value.
            labels_to_filter: SpeciesNet detector labels to retain.
            image_directory: Crop directory below each class directory.
            background_image_dir: Optional directory containing background images.
            background_class_name: Class name assigned to background images.
        """
        labels_root = path_to_labels or path_to_image_data
        YoloDatasetCreatorBase.__init__(
            self,
            path_to_image_data=path_to_image_data,
            path_to_labels=labels_root,
            dataset_output_path=dataset_output_path,
            class_names=class_names,
            train_val_test_split=train_val_test_split,
            rng_seed=rng_seed,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        self.allowed_classes = set(labels_to_filter or ["animal"])
        self.image_directory = image_directory
        self.background_image_dir = (
            Path(background_image_dir) if background_image_dir is not None else None
        )
        self.background_class_name = background_class_name
        self.crop_records: dict[str, list[dict]] = {}
        self.missing_crops: list[dict] = []
        self.labels = self._load_speciesnet_predictions()

    def _find_crop_path(
        self,
        species: str,
        source_path: Path,
        detection: dict,
        detection_index: int,
    ) -> Path | None:
        """Find the crop corresponding to an accepted source detection."""
        label = _path_component(
            str(detection.get("label", detection.get("class", "unknown")))
        )
        crop_dir = self.path_to_image_data / species / self.image_directory / label
        if not crop_dir.is_dir():
            return None
        pattern = f"{_path_component(source_path.stem)}_*_{detection_index:03d}.*"
        return next(
            (
                path
                for path in sorted(crop_dir.glob(pattern))
                if path.suffix.lower() in self.img_types
            ),
            None,
        )

    def _preprocess_labels(self, raw_labels: dict) -> dict[str, list[dict]]:
        """Build accepted crop records grouped by classification class."""
        records: dict[str, list[dict]] = {name: [] for name in self.class_names}
        for prediction in raw_labels["predictions"]:
            source_path = Path(prediction["filepath"])
            species = source_path.parent.name
            if species not in records:
                continue
            for detection_index, item in self._filter_label_items(
                prediction.get("detections", [])
            ):
                crop_path = self._find_crop_path(
                    species, source_path, item, detection_index
                )
                if crop_path is None:
                    self.missing_crops.append(
                        {
                            "species": species,
                            "source_path": str(source_path),
                            "detection_index": detection_index,
                            "label": item.get("label", item.get("class", "unknown")),
                        }
                    )
                    continue
                records[species].append(
                    {
                        "species": species,
                        "source_path": source_path,
                        "crop_path": crop_path,
                    }
                )

        if self.background_class_name in records:
            records[self.background_class_name].extend(
                {
                    "species": self.background_class_name,
                    "source_path": path,
                    "crop_path": path,
                }
                for path in self._background_image_paths()
            )
        self.crop_records = records
        return records

    def _filter_by_observation(self, labels: dict | None = None) -> dict[str, list[dict]]:
        """Keep crop records associated with first observations."""
        return {
            species: [
                record
                for record in records
                if self._is_primary_observation(record["source_path"])
            ]
            for species, records in self.crop_records.items()
        }

    def _split_train_val_test(
        self, processed_labels: dict[str, list[dict]] | None = None
    ) -> dict[str, list[dict]]:
        """Split records by source image so crops cannot leak across splits."""
        records_by_species = processed_labels or self.crop_records
        assignments: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for split in assignments:
            for species in self.class_names:
                (self.dataset_output_path / split / species).mkdir(
                    parents=True, exist_ok=True
                )

        for records in records_by_species.values():
            by_source: dict[Path, list[dict]] = defaultdict(list)
            for record in records:
                by_source[record["source_path"]].append(record)
            split_sources = self._split_paths(list(by_source))
            for split, sources in split_sources.items():
                for source in sources:
                    assignments[split].extend(by_source[source])
        return assignments

    def _write_labels(
        self,
        assignments: dict[str, list[dict]],
        preprocessed_labels: dict[str, list[dict]],
    ) -> Path:
        """Copy crops into split/class directories and write metadata."""
        for split, records in assignments.items():
            for record in records:
                destination = (
                    self.dataset_output_path
                    / split
                    / record["species"]
                    / record["crop_path"].name
                )
                shutil.copy2(record["crop_path"], destination)

        metadata = {
            "path": str(self.dataset_output_path),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.classes,
        }
        (self.dataset_output_path / "data.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False)
        )
        return self.dataset_output_path

    def __call__(self) -> Path:
        """Generate the classification dataset and return its output directory."""
        self._preprocess_labels(self.labels)
        self.crop_records = self._filter_by_observation()
        assignments = self._split_train_val_test(self.crop_records)
        return self._write_labels(assignments, self.crop_records)
