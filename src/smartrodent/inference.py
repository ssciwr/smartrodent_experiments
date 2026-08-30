"""Run in-memory detector-to-classifier inference with SpeciesNet or YOLO26.

Images use RGB ``PIL.Image.Image`` objects throughout the public API. Detectors
return crops paired with detection confidences, while classifiers return labels
paired with classification confidences.
"""

import math
from typing import Any

from PIL import Image
import speciesnet
import ultralytics


# Public wrapper methods use these aliases to make the two confidence-bearing
# return types easy to distinguish.
DetectionCrop = tuple[Image.Image, float]
Classification = tuple[str, float]


def _validate_max_top_k(max_top_k: int) -> None:
    """Validate a requested top-k result count.

    Args:
        max_top_k: Maximum number of predictions to return.

    Raises:
        ValueError: If ``max_top_k`` is not a positive integer. Booleans are
            rejected explicitly because ``bool`` is a subclass of ``int``.
    """
    if isinstance(max_top_k, bool) or not isinstance(max_top_k, int) or max_top_k <= 0:
        raise ValueError("max_top_k must be a positive integer")


def _crop_normalized_bbox(
    image: Image.Image, bbox: list[float]
) -> Image.Image | None:
    """Crop a normalized ``[x, y, width, height]`` box from an image.

    Args:
        image: Source RGB image.
        bbox: Normalized top-left x/y coordinates followed by width and height.

    Returns:
        The cropped image, or ``None`` if clipping produces an empty box.
    """
    x, y, width, height = bbox

    # Floor the top-left and ceil the bottom-right so fractional edge pixels are
    # retained. Clipping also handles detector boxes that slightly exceed [0, 1].
    left = max(0, min(image.width, math.floor(x * image.width)))
    top = max(0, min(image.height, math.floor(y * image.height)))
    right = max(0, min(image.width, math.ceil(x * image.width + width * image.width)))
    bottom = max(
        0, min(image.height, math.ceil(y * image.height + height * image.height))
    )

    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def load_image(img_path: str) -> Image.Image:
    """Load an image into the common in-memory representation.

    Args:
        img_path: Path to an image readable by Pillow.

    Returns:
        An independent RGB ``PIL.Image.Image`` whose source file is closed.

    Raises:
        OSError: If Pillow cannot open or decode the image.
    """
    # Convert while the source file is open. The returned image owns its pixel
    # data and therefore remains usable after leaving the context manager.
    with Image.open(img_path) as image:
        return image.convert("RGB")


class SpeciesNetDetector:
    """Detect image regions with the detector component of SpeciesNet."""

    def __init__(
        self, model: str, *, kwargs: dict[str, Any] | None = None
    ) -> None:
        """Load a SpeciesNet detector.

        Args:
            model: SpeciesNet model identifier or local model directory.
            kwargs: Wrapper inference options. ``conf`` optionally sets an
                additional minimum confidence applied to SpeciesNet detections.
        """
        # Copy caller-owned configuration so later lookups cannot mutate it.
        self.kwargs = dict(kwargs or {})
        self.model = speciesnet.SpeciesNetDetector(model)

    def detect(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[DetectionCrop]:
        """Detect and crop the most confident image regions.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of crops to return.

        Returns:
            Confidence-ordered ``(crop, detection_confidence)`` pairs. Invalid
            or empty bounding boxes are omitted.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        # Keep one canonical prediction path: detect_json performs inference and
        # this method only adds the non-serializable image crops.
        detections = self.detect_json(image, max_top_k)["detections"]
        crops = []
        for detection in detections:
            crop = _crop_normalized_bbox(image, detection["bbox"])
            if crop is not None:
                crops.append((crop, detection["confidence"]))
        return crops

    def detect_json(
        self, image: Image.Image, max_top_k: int = 1
    ) -> dict[str, Any]:
        """Return the most confident SpeciesNet bounding boxes.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of detections to return.

        Returns:
            A JSON-compatible ``{"detections": [...]}`` dictionary. Each
            detection contains a normalized ``bbox`` in ``[x, y, width,
            height]`` form and a float ``confidence``.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        _validate_max_top_k(max_top_k)

        # SpeciesNet separates preprocessing from prediction. The filepath is
        # reporting metadata only, so an empty value is sufficient in memory.
        prediction = self.model.predict("", self.model.preprocess(image))
        min_confidence = float(self.kwargs.get("conf", 0.0))

        detections = []
        for detection in prediction.get("detections", []):
            confidence = float(detection["conf"])
            bbox = detection.get("bbox")

            # SpeciesNet normally returns valid four-value boxes. Ignore malformed
            # records rather than allowing one record to prevent all crop output.
            if confidence < min_confidence or not isinstance(bbox, (list, tuple)):
                continue
            if len(bbox) != 4:
                continue
            detections.append(
                {
                    "bbox": [float(value) for value in bbox],
                    "confidence": confidence,
                }
            )

        # Sorting here guarantees the same top-k semantics for both backends.
        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return {"detections": detections[:max_top_k]}

    def __call__(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[DetectionCrop]:
        """Call :meth:`detect` with the requested top-k count.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of crops to return.

        Returns:
            Confidence-ordered crop and detection-confidence pairs.
        """
        return self.detect(image, max_top_k)


class SpeciesNetClassifier:
    """Classify image crops with the classifier component of SpeciesNet."""

    def __init__(
        self,
        model: str,
        max_top_k: int = 5,
        *,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Load a SpeciesNet classifier.

        Args:
            model: SpeciesNet model identifier or local model directory.
            max_top_k: Largest per-call top-k request accepted by this wrapper.
                SpeciesNet's public prediction API returns at most five classes.
            kwargs: Arguments forwarded to ``speciesnet.SpeciesNetClassifier``,
                such as ``device`` or ``target_species_txt``. A ``max_top_k``
                entry is accepted for configuration through :class:`Inference`.

        Raises:
            ValueError: If ``max_top_k`` is not between one and five.
        """
        self.kwargs = dict(kwargs or {})

        # Inference passes backend configuration as one dictionary. Pop this
        # wrapper-only option before forwarding SpeciesNet constructor arguments.
        max_top_k = self.kwargs.pop("max_top_k", max_top_k)
        _validate_max_top_k(max_top_k)
        if max_top_k > 5:
            raise ValueError("SpeciesNet returns at most five classifications")
        self.max_top_k = max_top_k
        self.model = speciesnet.SpeciesNetClassifier(model, **self.kwargs)

    def classify(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[Classification]:
        """Classify an image and return label-confidence pairs.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            Confidence-ordered ``(class_name, confidence)`` pairs.

        Raises:
            ValueError: If ``max_top_k`` is invalid or exceeds the configured
                maximum.
        """
        # classify_json owns inference and ordering; this method is only the
        # compact tuple representation used by the pipeline.
        classifications = self.classify_json(image, max_top_k)["classifications"]
        return [
            (classification["class"], classification["confidence"])
            for classification in classifications
        ]

    def classify_json(
        self, image: Image.Image, max_top_k: int = 1
    ) -> dict[str, Any]:
        """Return the most confident SpeciesNet classifications.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            A JSON-compatible ``{"classifications": [...]}`` dictionary. Each
            classification contains ``class`` and ``confidence`` values.

        Raises:
            ValueError: If ``max_top_k`` is invalid or exceeds the configured
                maximum.
        """
        _validate_max_top_k(max_top_k)
        if max_top_k > self.max_top_k:
            raise ValueError(
                f"max_top_k exceeds this classifier's configured maximum "
                f"of {self.max_top_k}"
            )

        prediction = self.model.predict("", self.model.preprocess(image))
        raw_classifications = prediction.get("classifications", {})
        classes = raw_classifications.get("classes", [])
        scores = raw_classifications.get("scores", [])

        # zip(..., strict=False) tolerates a failed or partial backend response;
        # every emitted record still has both a label and confidence.
        classifications = [
            {"class": class_name, "confidence": float(score)}
            for class_name, score in zip(classes, scores, strict=False)
        ]
        classifications.sort(
            key=lambda classification: classification["confidence"], reverse=True
        )
        return {"classifications": classifications[:max_top_k]}

    def __call__(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[Classification]:
        """Call :meth:`classify` with the requested top-k count.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            Confidence-ordered class and classification-confidence pairs.
        """
        return self.classify(image, max_top_k)


class YoloDetector:
    """Detect image regions with a fine-tuned YOLO26 detection model."""

    def __init__(
        self, model: str, *, kwargs: dict[str, Any] | None = None
    ) -> None:
        """Load a YOLO26 detector.

        Args:
            model: Path to fine-tuned YOLO26 detector weights.
            kwargs: Arguments forwarded to ``YOLO.predict``, such as ``conf`` or
                ``device``.
        """
        self.kwargs = dict(kwargs or {})
        # Explicit task selection avoids relying on a custom weight filename.
        self.model = ultralytics.YOLO(model, task="detect")

    def detect(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[DetectionCrop]:
        """Detect and crop the most confident image regions.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of crops to return.

        Returns:
            Confidence-ordered ``(crop, detection_confidence)`` pairs. Invalid
            or empty bounding boxes are omitted.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        detections = self.detect_json(image, max_top_k)["detections"]
        crops = []
        for detection in detections:
            crop = _crop_normalized_bbox(image, detection["bbox"])
            if crop is not None:
                crops.append((crop, detection["confidence"]))
        return crops

    def detect_json(
        self, image: Image.Image, max_top_k: int = 1
    ) -> dict[str, Any]:
        """Return the most confident YOLO26 bounding boxes.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of detections to return.

        Returns:
            A JSON-compatible ``{"detections": [...]}`` dictionary. Each
            detection contains a normalized ``bbox`` in ``[x, y, width,
            height]`` form and a float ``confidence``.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        _validate_max_top_k(max_top_k)
        result = self.model.predict(source=image, **self.kwargs)[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxyn[0].tolist()]

                # Ultralytics exposes normalized corner coordinates. Convert them
                # to the shared top-left/width/height representation.
                detections.append(
                    {
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": float(box.conf.item()),
                    }
                )

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return {"detections": detections[:max_top_k]}

    def __call__(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[DetectionCrop]:
        """Call :meth:`detect` with the requested top-k count.

        Args:
            image: Loaded RGB image.
            max_top_k: Maximum number of crops to return.

        Returns:
            Confidence-ordered crop and detection-confidence pairs.
        """
        return self.detect(image, max_top_k)


class YoloClassifier:
    """Classify image crops with a fine-tuned YOLO26 classification model."""

    def __init__(
        self, model: str, *, kwargs: dict[str, Any] | None = None
    ) -> None:
        """Load a YOLO26 classifier.

        Args:
            model: Path to fine-tuned YOLO26 classifier weights.
            kwargs: Arguments forwarded to ``YOLO.predict``, such as ``device``.
        """
        self.kwargs = dict(kwargs or {})
        # Explicit task selection is necessary for custom-named classifier weights.
        self.model = ultralytics.YOLO(model, task="classify")

    def classify(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[Classification]:
        """Classify an image and return label-confidence pairs.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            Confidence-ordered ``(class_name, confidence)`` pairs.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        classifications = self.classify_json(image, max_top_k)["classifications"]
        return [
            (classification["class"], classification["confidence"])
            for classification in classifications
        ]

    def classify_json(
        self, image: Image.Image, max_top_k: int = 1
    ) -> dict[str, Any]:
        """Return the most confident YOLO26 classifications.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            A JSON-compatible ``{"classifications": [...]}`` dictionary. Each
            classification contains ``class`` and ``confidence`` values. If the
            model produces no classification probabilities, the list is empty.

        Raises:
            ValueError: If ``max_top_k`` is not a positive integer.
        """
        _validate_max_top_k(max_top_k)
        result = self.model.predict(source=image, **self.kwargs)[0]
        if result.probs is None:
            return {"classifications": []}

        # Move scores to CPU before converting values to ordinary Python objects
        # suitable for the JSON-compatible response.
        scores = result.probs.data.detach().cpu()
        top_k = min(max_top_k, len(scores))
        indices = scores.argsort(descending=True)[:top_k].tolist()
        classifications = [
            {
                "class": result.names[int(index)],
                "confidence": float(scores[index].item()),
            }
            for index in indices
        ]
        return {"classifications": classifications}

    def __call__(
        self, image: Image.Image, max_top_k: int = 1
    ) -> list[Classification]:
        """Call :meth:`classify` with the requested top-k count.

        Args:
            image: Loaded RGB image or detector crop.
            max_top_k: Maximum number of classifications to return.

        Returns:
            Confidence-ordered class and classification-confidence pairs.
        """
        return self.classify(image, max_top_k)


class Inference:
    """Apply a selected detector and classifier successively to one image."""

    # Backend registries keep model selection explicit and avoid guessing the
    # backend from fine-tuned weight filenames.
    DETECTORS = {
        "speciesnet": SpeciesNetDetector,
        "yolo26": YoloDetector,
    }
    CLASSIFIERS = {
        "speciesnet": SpeciesNetClassifier,
        "yolo26": YoloClassifier,
    }

    def __init__(
        self,
        detector: str,
        classifier: str,
        detector_kwargs: dict[str, Any] | None = None,
        classifier_kwargs: dict[str, Any] | None = None,
        classes: list[str] | None = None,
        class_mappings: dict[str, str] | None = None,
    ) -> None:
        """Construct a detector-to-classifier pipeline.

        Args:
            detector: Detector backend name: ``"speciesnet"`` or ``"yolo26"``.
            classifier: Classifier backend name: ``"speciesnet"`` or
                ``"yolo26"``.
            detector_kwargs: Detector configuration containing a required
                ``model`` entry plus backend prediction arguments.
            classifier_kwargs: Classifier configuration containing a required
                ``model`` entry plus backend constructor or prediction arguments.
            classes: Optional allowlist applied after class-name mapping.
            class_mappings: Optional mapping from raw model labels to output labels.

        Raises:
            ValueError: If a backend is unsupported or either model entry is
                missing.
        """
        # Work with copies because extracting model paths must not modify caller
        # dictionaries that may also be reused for another pipeline.
        detector_config = dict(detector_kwargs or {})
        classifier_config = dict(classifier_kwargs or {})

        try:
            detector_class = self.DETECTORS[detector.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported detector backend: {detector!r}") from exc
        try:
            classifier_class = self.CLASSIFIERS[classifier.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported classifier backend: {classifier!r}") from exc

        # Keep model identifiers separate from options forwarded by each wrapper.
        try:
            detector_model = detector_config.pop("model")
        except KeyError as exc:
            raise ValueError("detector_kwargs must contain 'model'") from exc
        try:
            classifier_model = classifier_config.pop("model")
        except KeyError as exc:
            raise ValueError("classifier_kwargs must contain 'model'") from exc

        self.detector = detector_class(detector_model, kwargs=detector_config)
        self.classifier = classifier_class(classifier_model, kwargs=classifier_config)
        self.classes = list(classes) if classes is not None else None
        self.class_mappings = dict(class_mappings or {})

    def _detect(
        self, image: Image.Image, max_top_k: int
    ) -> list[DetectionCrop]:
        """Run the configured detector and extract crops.

        Args:
            image: Loaded RGB source image.
            max_top_k: Maximum number of detections to retain.

        Returns:
            Confidence-ordered crop and detection-confidence pairs.
        """
        return self.detector.detect(image, max_top_k)

    def _classify(
        self, crop: Image.Image, max_top_k: int
    ) -> list[dict[str, Any]]:
        """Classify one crop and apply output-label configuration.

        Args:
            crop: RGB crop returned by the detector.
            max_top_k: Maximum number of raw classifier predictions to inspect.

        Returns:
            JSON-compatible class and confidence dictionaries after mapping and
            optional allowlist filtering.
        """
        classifications = []
        for class_name, confidence in self.classifier.classify(crop, max_top_k):
            # Mapping precedes filtering so ``classes`` describes final API labels,
            # not backend-specific taxonomy strings or training labels.
            mapped_class = self.class_mappings.get(class_name, class_name)
            if self.classes is None or mapped_class in self.classes:
                classifications.append(
                    {"class": mapped_class, "confidence": confidence}
                )
        return classifications

    def __call__(
        self,
        image: Image.Image,
        detector_max_top_k: int = 1,
        classifier_max_top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """Run detection followed by classification for each retained crop.

        Args:
            image: Loaded RGB source image.
            detector_max_top_k: Maximum number of detected crops to classify.
            classifier_max_top_k: Maximum classifications returned per crop.

        Returns:
            One dictionary per detected crop. Each dictionary contains the
            detector confidence and its confidence-ordered classifications. An
            image without detections returns an empty list.

        Raises:
            ValueError: If either top-k value is invalid or exceeds a configured
                backend maximum.
        """
        results = []
        for crop, detection_confidence in self._detect(image, detector_max_top_k):
            # Keep detection and classification confidence separate; they describe
            # different model decisions and should not be multiplied implicitly.
            results.append(
                {
                    "detection_confidence": detection_confidence,
                    "classifications": self._classify(
                        crop, classifier_max_top_k
                    ),
                }
            )
        return results
