from pathlib import Path
from typing import Any, cast

import yaml
from PIL import Image
from speciesnet import SpeciesNet
from ultralytics import YOLO


class SpeciesNetYoloInference:
    """Detect animals with SpeciesNet and classify their cropped image regions."""

    def __init__(
        self, speciesnet_model: str | Path, classifier_weights: str | Path
    ) -> None:
        """Initializes the detector and classifier models.

        Args:
            speciesnet_model: SpeciesNet model identifier or local model path.
            classifier_weights: YOLO classification-model weights or path.
        """
        # SpeciesNet's detector gives us MegaDetector-style normalized boxes.
        self.detector: SpeciesNet = SpeciesNet(
            str(speciesnet_model), components="detector"
        )
        self.classify: YOLO = YOLO(str(classifier_weights), task="classify")

    @classmethod
    def from_config(cls, config: str) -> "SpeciesNetYoloInference":
        """Builds an inference instance from a YAML config file.

        The YAML config supplies the ``speciesnet_model`` and
        ``classifier_weights`` keyword arguments passed to :meth:`__init__`.

        Args:
            config: Path to a YAML config file containing the model
                ``speciesnet_model`` and ``classifier_weights`` settings.

        Returns:
            SpeciesNetYoloInference: A configured detector-and-classifier
            instance.

        Raises:
            ValueError: If the supplied config file does not exist.
        """
        if not Path(config).resolve().exists():
            raise ValueError("Error, supplied config does not exist")

        with open(Path(config).resolve(), "r") as f:
            cfg = yaml.safe_load(f)

        return cls(**cfg)

    def _detect(self, image_path: str | Path) -> Image.Image | None:
        """Detects the most confident animal and returns its RGB crop.

        Args:
            image_path: Path to the source image.

        Returns:
            PIL.Image.Image | None: The crop for the highest-confidence detection,
            or ``None`` when no animal is detected.
        """
        # Resolve paths before passing them to SpeciesNet, which receives strings.
        image_path = Path(image_path).resolve()
        detection = cast(
            dict[str, Any],
            self.detector.predict(filepaths=[str(image_path)], batch_size=1),
        )
        detections: list[dict[str, Any]] = detection["predictions"][0].get(
            "detections", []
        )
        if not detections:
            return None

        # Keep the single highest-confidence detection.
        x, y, width, height = max(detections, key=lambda item: item["conf"])["bbox"]
        with Image.open(image_path) as image:
            # YOLO's classifier receives a consistent, three-channel image.
            image = image.convert("RGB")
            return image.crop(
                (
                    x * image.width,
                    y * image.height,
                    (x + width) * image.width,
                    (y + height) * image.height,
                )
            )

    def _classify(self, crop: Image.Image | None) -> dict[str, object]:
        """Classifies a detected animal crop.

        Args:
            crop: RGB animal crop returned by :meth:`_detect`, or ``None``.

        Returns:
            dict: A mapping containing class ``probabilities``, the top ``result``,
            and its ``confidence``. Empty probabilities and ``None`` values are
            returned when there is no detection to classify.
        """
        if crop is None:
            return {"probabilities": {}, "result": None, "confidence": None}

        classification: Any = self.classify(crop, verbose=False)

        # Expose every class score, not only YOLO's top prediction.
        probabilities: dict[str, float] = {
            classification.names[index]: float(probability)
            for index, probability in enumerate(
                classification.probs.data.cpu().tolist()
            )
        }
        top1 = int(classification.probs.top1)
        return {
            "probabilities": probabilities,
            "result": classification.names[top1],
            "confidence": float(classification.probs.top1conf),
        }

    def predict(self, image_path: str | Path) -> dict[str, object]:
        """Runs detection followed by classification for one image.

        Args:
            image_path: Path to the image to analyze.

        Returns:
            dict: Classification probabilities, predicted class, and confidence.
        """
        # Keep the public pipeline explicit: detection produces the classifier input.
        return self._classify(self._detect(image_path))
