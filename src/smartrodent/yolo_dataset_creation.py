"""Prepare record-based YOLO datasets and provide SpeciesNet batch inference."""

from collections.abc import Sequence
from abc import abstractmethod
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Self

from PIL import Image
import pandas as pd
import torch
import yaml
from ultralytics.utils.metrics import box_iou
from speciesnet import DEFAULT_MODEL, SpeciesNet

from .base import YoloDatasetCreatorBase
from .utils import path_component


class _SpeciesNetDatasetMixin:
    """Shared SpeciesNet inference and detection filtering.

    Call ``initialize_speciesnet`` before live inference. Explicit setup avoids
    adding a cooperative constructor to the dataset creators' inheritance chain.
    """

    def initialize_speciesnet(
        self,
        *,
        model: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        """Configure the instance's reusable detector without loading weights.

        Args:
            model: Optional injected SpeciesNet-compatible detector instance.
            model_name: Identifier or local weights directory used for lazy loading.
                None selects SpeciesNet's default model on first nonempty inference.

        Raises:
            ValueError: Both an existing model and a model identifier are supplied.
        """
        if model is not None and model_name is not None:
            raise ValueError("Supply either model or model_name, not both")
        self.speciesnet_model = model
        self.speciesnet_model_name = model_name

    def _get_speciesnet_model(self) -> Any:
        """Load the configured detector once, or return the existing instance."""
        if self.speciesnet_model is None:
            self.speciesnet_model = SpeciesNet(
                (
                    self.speciesnet_model_name
                    if self.speciesnet_model_name is not None
                    else DEFAULT_MODEL
                ),
                components="detector",
            )
        return self.speciesnet_model

    def infer_batch(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 32,
        confidence_threshold: float = 0.1,
        allowed_classes: Sequence[str] = ("animal",),
        iou_threshold: float = 0.45,
    ) -> dict[str, dict]:
        """Run SpeciesNet detection and extract accepted crops without disk writes.

        Args:
            image_paths: Source images to process together.
            batch_size: Maximum source images passed to each detector invocation.
            confidence_threshold: Inclusive minimum detection confidence.
            allowed_classes: Detector labels to retain.
            iou_threshold: Maximum overlap before a weaker box is suppressed.

        Returns:
            Full source-path strings mapped to dictionaries with a ``detections``
            list. Each detection contains a normalized xywh ``bbox``, ``confidence``,
            ``label``, original ``detection_index``, and an independent PIL ``crop``.
            Crops are in-memory images, not JSON-serializable metadata; exclude them
            when serializing records. Images without accepted boxes have empty lists.

        Raises:
            ValueError: Settings are invalid or model results do not match inputs.
            RuntimeError: ``initialize_speciesnet`` has not been called.
        """
        if not hasattr(self, "speciesnet_model"):
            raise RuntimeError("Call initialize_speciesnet before infer_batch")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        filepaths = [str(Path(path).resolve()) for path in image_paths]
        if not filepaths:
            return {}
        predictions = self._run_speciesnet_batches(filepaths, batch_size)

        results = {}
        for prediction in predictions:
            source_path = str(Path(prediction["filepath"]).resolve())
            accepted = self._filter_label_items(
                prediction["detections"],
                allowed_classes=allowed_classes,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
            results[source_path] = {
                "detections": self._extract_crop_records(source_path, accepted)
            }
        return results

    def _run_speciesnet_batches(
        self, filepaths: list[str], batch_size: int
    ) -> list[dict]:
        """Run bounded SDK calls and verify one prediction per source image."""
        model = self._get_speciesnet_model()
        # Filter once in the mixin, rather than letting the SDK discard boxes first.
        # This preserves indices before confidence filtering and NMS.
        detector_type = type(model.detector)
        previous_threshold = detector_type.DETECTION_THRESHOLD
        predictions = []
        try:
            # The SDK reads the class attribute, not an instance override. Restore
            # it even on failure so subsequent model users retain their settings.
            detector_type.DETECTION_THRESHOLD = 0.0
            # predict() needs classifier/ensemble components. The detector-only
            # detect() API has no batch_size argument, so bound its input here.
            for start in range(0, len(filepaths), batch_size):
                batch = model.detect(filepaths=filepaths[start : start + batch_size])
                predictions.extend(batch["predictions"])
        finally:
            detector_type.DETECTION_THRESHOLD = previous_threshold

        predicted_paths = [
            str(Path(item["filepath"]).resolve()) for item in predictions
        ]
        if set(predicted_paths) != set(filepaths) or len(predicted_paths) != len(
            filepaths
        ):
            raise ValueError("SpeciesNet must return one prediction per input image")
        return predictions

    @staticmethod
    def _extract_crop_records(
        source_path: str, accepted: list[tuple[int, dict]]
    ) -> list[dict]:
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
                    "bbox": detection["bbox"],
                    "confidence": detection.get("conf", detection.get("confidence")),
                    "label": detection.get("label", detection.get("class")),
                    "detection_index": index,
                    "crop": crop,
                }
            )
        return records

    @staticmethod
    def _filter_label_items(
        detections: list,
        *,
        allowed_classes: Sequence[str],
        confidence_threshold: float,
        iou_threshold: float,
    ) -> list[tuple[int, dict]]:
        """Filter detections by confidence/class and NMS, preserving original indices."""
        # Keep original indices because SpeciesNet crop filenames include the
        # detection index; index tracking must survive filtering and NMS.
        candidates = [
            (idx, deepcopy(detection)) for idx, detection in enumerate(detections)
        ]
        candidates = [
            (idx, detection)
            for idx, detection in candidates
            if detection.get("conf", detection.get("confidence", 0.0))
            >= confidence_threshold
            and detection.get("label", detection.get("class")) in allowed_classes
        ]
        candidates.sort(
            key=lambda item: item[1].get("conf", item[1].get("confidence", 0.0)),
            reverse=True,
        )

        def bbox_xywh_to_xyxy_tensor(bbox: list[float]) -> torch.Tensor:
            """Convert SpeciesNet [x_min, y_min, width, height] to xyxy for IoU."""
            x_min, y_min, width, height = bbox
            return torch.tensor(
                [[x_min, y_min, x_min + width, y_min + height]], dtype=torch.float32
            )

        if not candidates:
            return []

        keep = []
        suppressed = set()
        # Confidence sorting lets each higher-confidence box suppress overlapping
        # lower-confidence candidates, matching standard greedy NMS behavior.
        for i, detection in enumerate(candidates):
            if i in suppressed:
                continue

            keep.append(detection)
            bbox = bbox_xywh_to_xyxy_tensor(detection[1]["bbox"])
            for j in range(i + 1, len(candidates)):
                if j in suppressed:
                    continue

                other_bbox = bbox_xywh_to_xyxy_tensor(candidates[j][1]["bbox"])
                if box_iou(bbox, other_bbox).item() > iou_threshold:
                    suppressed.add(j)

        return keep


class _SpeciesNetDatasetCreatorBase(_SpeciesNetDatasetMixin, YoloDatasetCreatorBase):
    """Combine record loading with explicit, lazy SpeciesNet configuration."""

    def __init__(
        self,
        path_to_image_data: str | Path,
        dataset_output_path: str | Path,
        class_names: list[str] | None = None,
        *,
        model: Any | None = None,
        model_name: str | None = None,
    ):
        """Load source records and configure the reusable baseline detector.

        Args:
            path_to_image_data: Input root containing per-species records.csv files.
            dataset_output_path: Destination for the generated dataset.
            class_names: Optional species selection in class-index order. None
                discovers all species directories in sorted order.
            model: Optional existing SpeciesNet-compatible detector instance.
            model_name: Identifier or local weights directory for lazy model loading.
                None selects SpeciesNet's default model when no model is injected.

        Raises:
            ValueError: Model configuration, class selection, or metadata is invalid.
            FileNotFoundError: Required records or source files are missing.
        """
        # Validate model configuration before reading records. Weight loading stays
        # in the mixin and occurs only on the first nonempty inference request.
        self.initialize_speciesnet(model=model, model_name=model_name)
        super().__init__(path_to_image_data, dataset_output_path, class_names)
        self.metadata_records: list[dict] = []

    @classmethod
    def from_config(cls, config_path: str | Path) -> Self:
        """Build a creator from its class-named mapping under ``data``.

        Args:
            config_path: YAML configuration with a ``data`` mapping keyed by this
                creator's class name. Source/output paths are relative to the
                working directory, not the configuration file's directory.
                Required settings are ``path_to_image_data`` and
                ``dataset_output_path``. Optional settings are ``class_names`` and
                ``model_name``; omitted values retain the constructor defaults.

        Returns:
            A creator of the requested type, with source records loaded and model
            weights still unloaded.

        Raises:
            FileNotFoundError: The configuration or required source files are missing.
            TypeError: YAML sections or setting values have incorrect types.
            ValueError: Required paths are missing, unsupported settings are supplied,
                or class selection, metadata, or model configuration is invalid.
        """
        settings = cls._load_creator_settings(config_path)
        cls._validate_creator_settings(settings)
        return cls(**settings)

    @classmethod
    def _load_creator_settings(cls, config_path: str | Path) -> dict:
        """Read the required creator mapping without borrowing unrelated pipeline code."""
        path = Path(config_path).expanduser()
        configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(configuration, dict):
            raise TypeError("The configuration must be a mapping")
        data = configuration.get("data")
        if not isinstance(data, dict):
            raise TypeError("The 'data' configuration must be a mapping")
        settings = data.get(cls.__name__)
        if not isinstance(settings, dict):
            raise TypeError(
                f"The 'data.{cls.__name__}' configuration must be a mapping"
            )
        return settings

    @staticmethod
    def _validate_creator_settings(settings: dict) -> None:
        """Reject ambiguous YAML values and Python-only model injection before setup."""
        required = {"path_to_image_data", "dataset_output_path"}
        allowed = required | {"class_names", "model_name"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                f"Unsupported dataset creator settings: {sorted(map(str, unknown))}"
            )
        missing = required - set(settings)
        if missing:
            raise ValueError(
                f"Missing required dataset creator settings: {sorted(missing)}"
            )
        for name in sorted(required):
            value = settings[name]
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a path string")
            if not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")

        class_names = settings.get("class_names")
        if class_names is not None:
            if not isinstance(class_names, list) or any(
                not isinstance(name, str) for name in class_names
            ):
                raise TypeError("class_names must be a list of strings or null")
        model_name = settings.get("model_name")
        if model_name is not None:
            if not isinstance(model_name, str):
                raise TypeError("model_name must be a string or null")
            if not model_name.strip():
                raise ValueError("model_name must be a nonempty string or null")

    def create(self) -> Path:
        """Infer source records and write the configured YOLO dataset.

        Existing split assignments and row multiplicity are preserved. Images with
        no accepted detections are skipped, with the reason in ``metadata.json``.
        The metadata retains source columns, detection details, and output paths
        relative to the dataset root. Crops are never included in serialized data.
        Creation uses the mixin defaults: animal labels, confidence >= 0.1, and
        IoU threshold 0.45, with at most 32 source rows held per inference batch.

        Returns:
            The generated dataset's root directory.

        Raises:
            FileExistsError: The destination already exists.
            OSError: Source images cannot be read or output files cannot be written.
            ValueError: The model returns invalid predictions or the output directory
                overlaps the input.
        """
        if self.dataset_output_path.is_relative_to(
            self.path_to_image_data
        ) or self.path_to_image_data.is_relative_to(self.dataset_output_path):
            raise ValueError("Dataset output must not overlap the input directory")
        self.dataset_output_path.mkdir(parents=True, exist_ok=False)
        self._create_directories()
        self.metadata_records = []
        for species, records in self.records_by_species.items():
            self._create_species_records(species, records)
        self._write_dataset_metadata()
        return self.dataset_output_path

    def _create_species_records(self, species: str, records: pd.DataFrame) -> None:
        """Infer bounded row batches and write each occurrence in source-row order."""
        # Bound the number of live crops, not just SDK calls. Inferring the whole
        # dataset first would retain every crop in memory. Duplicate sources within
        # a batch share inference, but each oversampled row still produces an output.
        batch_size = 32
        for start in range(0, len(records), batch_size):
            frame = records.iloc[start : start + batch_size]
            predictions = self.infer_batch(
                frame["image_path"].drop_duplicates().tolist()
            )
            # Convert missing values to JSON null without a pandas JSON round-trip,
            # which would round high-precision metadata. Keep the source frame intact.
            rows = (
                frame.astype(object)
                .where(frame.notna(), None)
                .to_dict(orient="records")
            )
            for offset, row in enumerate(rows):
                source_row = start + offset
                detections = predictions[row["image_path"]]["detections"]
                metadata = self._write_source_record(
                    species, source_row, row, detections
                )
                self.metadata_records.append(metadata)

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
            stem = self._output_stem(species, row["image_path"], source_row)
            outputs = self._write_outputs(species, row, stem, detections)
            metadata.update(status="written", skip_reason=None, outputs=outputs)
        return metadata

    @staticmethod
    def _output_stem(species: str, image_path: str, source_row: int) -> str:
        """Name sources independently of output root, without basename collisions."""
        # Full path plus class distinguishes same-named sources, and row position
        # preserves intentional oversampling. Limit the readable stem so long
        # source filenames still fit normal filesystem component-length limits.
        identity = sha256(json.dumps([species, image_path]).encode()).hexdigest()
        readable_stem = path_component(Path(image_path).stem)[:80]
        return f"{readable_stem}_{identity}_row{source_row:08d}"

    def _write_dataset_metadata(self) -> None:
        """Persist training configuration and JSON-safe source/output provenance."""
        data = {
            "path": str(self.dataset_output_path),
            **self._training_paths(),
            "names": self.classes,
        }
        (self.dataset_output_path / "data.yaml").write_text(yaml.safe_dump(data))
        (self.dataset_output_path / "metadata.json").write_text(
            json.dumps({"records": self.metadata_records}, allow_nan=False, indent=2)
        )

    @abstractmethod
    def _create_directories(self) -> None:
        """Create the format-specific split directory layout."""
        raise NotImplementedError

    @abstractmethod
    def _write_outputs(
        self, species: str, row: dict, stem: str, detections: list[dict]
    ) -> list[dict]:
        """Write one source occurrence and return relative output-path metadata."""
        raise NotImplementedError

    @abstractmethod
    def _training_paths(self) -> dict[str, str]:
        """Return the format's training, validation, and test directory paths."""
        raise NotImplementedError


class YoloDetectorDatasetCreatorFromSpeciesnet(_SpeciesNetDatasetCreatorBase):
    """Create a YOLO detector dataset using source-image symlinks and inferred boxes.

    Uses the shared input-root/output-root constructor and optional ``class_names``
    selection. Input records already determine all split assignments.
    """

    def _create_directories(self) -> None:
        """Create parallel images and labels directories for every existing split."""
        for split in ["train", "val", "test"]:
            for directory in ["images", "labels"]:
                (self.dataset_output_path / directory / split).mkdir(parents=True)

    def _write_outputs(
        self, species: str, row: dict, stem: str, detections: list[dict]
    ) -> list[dict]:
        """Symlink a full source image and write its normalized YOLO box labels."""
        source = Path(row["image_path"])
        split = row["dataset_split"]
        image_path = Path("images") / split / f"{stem}{source.suffix}"
        label_path = Path("labels") / split / f"{stem}.txt"
        (self.dataset_output_path / image_path).symlink_to(source)
        class_index = self.class_names.index(species)
        labels = []
        for detection in detections:
            x, y, width, height = detection["bbox"]
            # SpeciesNet uses normalized top-left xywh; YOLO expects center xywh.
            labels.append([class_index, x + width / 2, y + height / 2, width, height])
        (self.dataset_output_path / label_path).write_text(
            "".join(" ".join(map(str, label)) + "\n" for label in labels)
        )
        return [{"image_path": str(image_path), "label_path": str(label_path)}]

    def _training_paths(self) -> dict[str, str]:
        """Return the detector image directories used by Ultralytics."""
        return {split: f"images/{split}" for split in ["train", "val", "test"]}


class YoloClassifierDatasetCreatorFromSpeciesnet(_SpeciesNetDatasetCreatorBase):
    """Create a YOLO classifier dataset by saving each accepted detection crop.

    Uses the shared input-root/output-root constructor and optional ``class_names``
    selection. Oversampled rows remain intact during crop output generation.
    """

    def _create_directories(self) -> None:
        """Create each class directory in each existing split, including empty ones."""
        for split in ["train", "val", "test"]:
            for species in self.class_names:
                (self.dataset_output_path / split / species).mkdir(parents=True)

    def _write_outputs(
        self, species: str, row: dict, stem: str, detections: list[dict]
    ) -> list[dict]:
        """Save independent, lossless crops and retain their detection provenance."""
        outputs = []
        for detection in detections:
            index = detection["detection_index"]
            image_path = (
                Path(row["dataset_split"]) / species / f"{stem}_det{index:03d}.png"
            )
            # PNG avoids introducing another lossy compression pass into crops.
            detection["crop"].save(self.dataset_output_path / image_path)
            outputs.append({"image_path": str(image_path), "detection_index": index})
        return outputs

    def _training_paths(self) -> dict[str, str]:
        """Return the classifier split directories used by Ultralytics."""
        return {split: split for split in ["train", "val", "test"]}
