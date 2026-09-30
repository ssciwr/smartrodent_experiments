"""Tests for YOLO detection datasets built from SpeciesNet predictions."""

import json
from pathlib import Path

import pytest
import yaml

from smartrodent.dataset_creation import YoloDetectorDatasetCreatorFromSpeciesnet


def write_predictions(root: Path, species: str, predictions: list[dict]) -> None:
    """Write one SpeciesNet prediction file below a species directory."""
    species_root = root / species
    species_root.mkdir(parents=True, exist_ok=True)
    (species_root / "predictions.json").write_text(
        json.dumps({"predictions": predictions})
    )


def detection(
    *,
    confidence: float = 0.9,
    label: str = "animal",
    bbox: list[float] | None = None,
) -> dict:
    """Build a SpeciesNet detection record."""
    return {
        "label": label,
        "conf": confidence,
        "bbox": bbox or [0.1, 0.2, 0.4, 0.2],
    }


def test_detector_filters_by_label_confidence_and_nms(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    (images / "mouse").mkdir(parents=True)
    write_predictions(labels, "mouse", [{"filepath": "mouse/a.jpg", "detections": []}])
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(
        path_to_image_data=images,
        path_to_labels=labels,
        dataset_output_path=tmp_path / "output",
        class_names=["mouse"],
        labels_to_filter=["animal"],
        confidence_threshold=0.5,
        iou_threshold=0.2,
    )
    detections = [
        detection(confidence=0.9),
        detection(confidence=0.8, bbox=[0.12, 0.2, 0.4, 0.2]),
        detection(confidence=0.4, bbox=[0.7, 0.7, 0.1, 0.1]),
        detection(confidence=0.95, label="vehicle", bbox=[0.7, 0.7, 0.1, 0.1]),
    ]

    assert creator._filter_labels(detections) == [detections[0]]


def test_detector_builds_detection_layout_and_background_labels(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    species_dir = images / "mouse"
    species_dir.mkdir(parents=True)
    source = species_dir / "mouse.jpg"
    source.write_bytes(b"mouse")
    background_dir = tmp_path / "background"
    background_dir.mkdir()
    background = background_dir / "empty.jpg"
    background.write_bytes(b"empty")
    write_predictions(
        labels,
        "mouse",
        [{"filepath": str(source), "detections": [detection()]}],
    )
    output = tmp_path / "output"

    result = YoloDetectorDatasetCreatorFromSpeciesnet(
        path_to_image_data=images,
        path_to_labels=labels,
        dataset_output_path=output,
        class_names=["mouse", "empty"],
        labels_to_filter=["animal"],
        background_image_dir=background_dir,
        train_val_test_split=(1.0, 0.0, 0.0),
    )()

    assert result == output
    assert (output / "images/train/mouse.jpg").read_bytes() == b"mouse"
    label = (output / "labels/train/mouse.txt").read_text().strip().split()
    assert label == ["0", "0.30000000000000004", "0.30000000000000004", "0.4", "0.2"]
    assert (output / "images/train/empty.jpg").read_bytes() == b"empty"
    assert (output / "labels/train/empty.txt").read_text() == ""
    metadata = yaml.safe_load((output / "data.yaml").read_text())
    assert metadata["names"] == {0: "mouse"}
    assert {path.name for path in output.iterdir()} == {"images", "labels", "data.yaml"}


def test_detector_reports_absent_predictions(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    with pytest.raises(FileNotFoundError, match="No SpeciesNet predictions"):
        YoloDetectorDatasetCreatorFromSpeciesnet(
            path_to_image_data=images,
            path_to_labels=labels,
            dataset_output_path=tmp_path / "output",
            class_names=["mouse"],
        )
