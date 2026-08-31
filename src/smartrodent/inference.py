import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from speciesnet import SpeciesNet
from ultralytics import YOLO
from tqdm import tqdm


class SpeciesNetYoloInference:
    """Detect animals with SpeciesNet and classify their cropped image regions."""

    def __init__(
        self,
        speciesnet_detector: str | Path,
        yolo_classifier: str | Path,
        repo_id: str = "MaHaWo/Yolo26Rodent",
    ) -> None:
        """Initializes the detector and classifier models.

        Args:
            speciesnet_detector: SpeciesNet model identifier or local model path.
            yolo_classifier: Classifier filename within ``repo_id``.
            repo_id: Hugging Face repository containing the classifier weights.
        """

        weights_path = hf_hub_download(
            repo_id=str(repo_id), filename=str(yolo_classifier)
        )

        # SpeciesNet's detector gives us MegaDetector-style normalized boxes.
        self.detector: SpeciesNet = SpeciesNet(
            str(speciesnet_detector), components="detector", multiprocessing=False
        )
        self.classify: YOLO = YOLO(str(weights_path), task="classify")

    @classmethod
    def from_config(cls, config: str) -> "SpeciesNetYoloInference":
        """Builds an inference instance from a YAML config file.

        The YAML config maps ``speciesnet_model`` to ``speciesnet_detector`` and
        ``classifier_weights`` to ``yolo_classifier``.

        Args:
            config: Path to a YAML config file containing the model
                ``speciesnet_model`` and ``classifier_weights`` settings.

        Returns:
            SpeciesNetYoloInference: A configured detector-and-classifier
            instance.

        Raises:
            ValueError: If the supplied config file does not exist.
        """
        path = Path(config).resolve()
        if not path.exists():
            raise ValueError(f"Config file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
        missing = [
            key
            for key in ("speciesnet_model", "classifier_weights")
            if not cfg.get(key)
        ]
        if missing:
            raise ValueError(f"Config missing required key(s): {', '.join(missing)}")
        return cls(
            speciesnet_detector=cfg["speciesnet_model"],
            yolo_classifier=cfg["classifier_weights"],
        )

    def _detect(
        self, image_path: str | Path
    ) -> tuple[list[Image.Image] | None, list[tuple[int, int, int, int]] | None]:
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
            self.detector.detect(
                filepaths=[str(image_path)],
            ),
        )
        if not detection.get("predictions"):
            return None, None

        detections: list[dict[str, Any]] = detection["predictions"][0].get(
            "detections", []
        )
        # Skip entries missing confidence or carrying malformed bounding boxes.
        valid_detections = [
            d
            for d in detections
            if "conf" in d
            and isinstance(d.get("bbox"), (list, tuple))
            and len(d["bbox"]) == 4
        ]
        if not valid_detections:
            return None, None

        # Keep the 5 highest-confidence detections
        valid_detections.sort(key=lambda item: item["conf"], reverse=True)

        valid_detections = valid_detections[0 : min(5, len(valid_detections))]
        crops = []
        bboxs = []
        for detect in valid_detections:
            x, y, width, height = detect["bbox"]
            with Image.open(image_path) as image:
                # YOLO's classifier receives a consistent, three-channel image.
                image = image.convert("RGB")
                crops.append(
                    image.crop(
                        (
                            x * image.width,
                            y * image.height,
                            (x + width) * image.width,
                            (y + height) * image.height,
                        )
                    )
                )
            bboxs.append((x, y, width, height))

        return crops, bboxs

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

        # only one element b/c only one input
        classification: Any = next(iter(self.classify(crop, verbose=False)))

        # Expose every class score, not only YOLO's top prediction.
        probabilities: dict[str, float] = {
            classification.names[index]: float(probability)
            for index, probability in enumerate(
                classification.probs.data.cpu().tolist()
            )
        }
        top1 = int(classification.probs.top1)
        top5 = np.array(classification.probs.top5)

        return {
            "probabilities": probabilities,
            "result": classification.names[top1],
            "confidence": float(classification.probs.top1conf),
            "top5": [classification.names[i] for i in top5],
            "top5_confidence": classification.probs.top5conf.cpu().tolist(),
        }

    def predict(self, image_path: str | Path) -> dict[int, dict[str, object]]:
        """Runs detection followed by classification for one image.

        Args:
            image_path: Path to the image to analyze.

        Returns:
            dict: Classification probabilities, predicted class, and confidence.
        """
        # Keep the public pipeline explicit: detection produces the classifier input.

        detections, bboxs = self._detect(image_path)

        if detections is None or bboxs is None:
            return {}

        results = {i: self._classify(d) for i, d in enumerate(detections)}
        for i in range(len(detections)):
            results[i]["bbox"] = bboxs[i]
        return results


def main():
    """Run inference using the supplied YAML configuration file."""
    parser = argparse.ArgumentParser(
        prog="SmartRodentInference",
        description="Run detection->classification inference on camera trap images",
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    inferencepipeline = SpeciesNetYoloInference.from_config(args.config)
    input_path = Path(config["path"]).resolve()
    images = tqdm(list(input_path.iterdir()) if input_path.is_dir() else [input_path])

    results = {
        image.name: inferencepipeline.predict(image)
        for image in images
        if image.suffix.lower() in config["imgs"]
    }

    output_path = Path(config["output"])
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
