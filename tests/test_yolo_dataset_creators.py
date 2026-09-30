"""Behavioral tests for YOLO datasets built from SpeciesNet predictions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, strategies as st
import pytest
import yaml

from smartrodent.base import YoloDatasetCreatorBase
from smartrodent.dataprocessing import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)


class ConcreteDatasetCreator(YoloDatasetCreatorBase):
    """Minimal concrete creator used to exercise shared validation."""

    def __call__(self):
        """Return the configured output root."""
        return self.dataset_output_path


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


def test_base_validates_inputs_and_creates_only_dataset_root(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    images.mkdir()
    labels.mkdir()

    creator = ConcreteDatasetCreator(
        path_to_image_data=images,
        path_to_labels=labels,
        dataset_output_path=output,
        class_names=["mouse"],
    )

    assert creator.dataset_output_path == output
    assert output.is_dir()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("missing_argument", ["path_to_image_data", "path_to_labels"])
def test_base_rejects_missing_input_paths(tmp_path, missing_argument):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    arguments = {
        "path_to_image_data": images,
        "path_to_labels": labels,
        "dataset_output_path": tmp_path / "output",
        "class_names": ["mouse"],
    }
    arguments[missing_argument] = tmp_path / "missing"

    with pytest.raises(ValueError, match="does not exist"):
        ConcreteDatasetCreator(**arguments)


@given(
    train=st.integers(min_value=0, max_value=100),
    validation=st.integers(min_value=0, max_value=100),
    test=st.integers(min_value=0, max_value=100),
)
def test_base_rejects_split_fractions_that_do_not_sum_to_one(
    train, validation, test
):
    if train + validation + test == 100:
        return
    with TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConcreteDatasetCreator(
                path_to_image_data=images,
                path_to_labels=labels,
                dataset_output_path=root / "output",
                class_names=["mouse"],
                train_val_test_split=(train / 100, validation / 100, test / 100),
            )


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


def test_classifier_keeps_crops_from_one_source_in_one_split(tmp_path):
    root = tmp_path / "speciesnet"
    species_root = root / "mouse"
    crop_root = species_root / "crops" / "animal"
    crop_root.mkdir(parents=True)
    source_a = species_root / "a.jpg"
    source_b = species_root / "b.jpg"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    for name in ["a_hash_000.jpg", "a_hash_001.jpg", "b_hash_000.jpg"]:
        (crop_root / name).write_bytes(name.encode())
    write_predictions(
        root,
        "mouse",
        [
            {
                "filepath": str(source_a),
                "detections": [
                    detection(bbox=[0.0, 0.0, 0.2, 0.2]),
                    detection(bbox=[0.7, 0.7, 0.2, 0.2]),
                ],
            },
            {"filepath": str(source_b), "detections": [detection()]},
        ],
    )
    output = tmp_path / "output"

    result = YoloClassifierDatasetCreatorFromSpeciesnet(
        path_to_image_data=root,
        dataset_output_path=output,
        class_names=["mouse"],
        train_val_test_split=(0.5, 0.0, 0.5),
        rng_seed=7,
    )()

    assert result == output
    locations = {
        image.name: split
        for split in ("train", "val", "test")
        for image in (output / split / "mouse").glob("*.jpg")
    }
    assert locations["a_hash_000.jpg"] == locations["a_hash_001.jpg"]
    assert set(locations) == {"a_hash_000.jpg", "a_hash_001.jpg", "b_hash_000.jpg"}
    assert not (output / "images").exists()
    assert not (output / "labels").exists()


def test_classifier_records_missing_crops(tmp_path):
    root = tmp_path / "speciesnet"
    species_root = root / "mouse"
    species_root.mkdir(parents=True)
    source = species_root / "mouse.jpg"
    source.write_bytes(b"mouse")
    write_predictions(
        root,
        "mouse",
        [{"filepath": str(source), "detections": [detection()]}],
    )
    creator = YoloClassifierDatasetCreatorFromSpeciesnet(
        path_to_image_data=root,
        dataset_output_path=tmp_path / "output",
        class_names=["mouse"],
    )

    creator()

    assert creator.missing_crops == [
        {
            "species": "mouse",
            "source_path": str(source),
            "detection_index": 0,
            "label": "animal",
        }
    ]
