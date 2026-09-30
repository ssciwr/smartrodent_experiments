"""Tests for YOLO classification datasets built from SpeciesNet crops."""

import json
from pathlib import Path

from smartrodent.dataset_creation import YoloClassifierDatasetCreatorFromSpeciesnet


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
