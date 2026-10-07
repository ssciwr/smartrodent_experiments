"""Prepare record-based YOLO datasets and provide SpeciesNet batch inference."""

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from ultralytics.utils.metrics import box_iou
from speciesnet import DEFAULT_MODEL, SpeciesNet

from .base import YoloDatasetCreatorBase


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


class YoloDetectorDatasetCreatorFromSpeciesnet(
    _SpeciesNetDatasetMixin, YoloDatasetCreatorBase
):
    """Prepare species records for future YOLO detector dataset conversion.

    Uses the shared input-root/output-root constructor and optional ``class_names``
    selection. Input records already determine all split assignments.
    """

    def create(self) -> Path:
        """Report that detector inference/output integration is not yet available.

        Raises:
            NotImplementedError: Inference/output integration is pending.
        """
        return super().create()


class YoloClassifierDatasetCreatorFromSpeciesnet(
    _SpeciesNetDatasetMixin, YoloDatasetCreatorBase
):
    """Prepare species records for future YOLO classifier dataset conversion.

    Uses the shared input-root/output-root constructor and optional ``class_names``
    selection. Oversampled rows remain intact for later crop output generation.
    """

    def create(self) -> Path:
        """Report that classifier inference/output integration is not yet available.

        Raises:
            NotImplementedError: Inference/output integration is pending.
        """
        return super().create()
