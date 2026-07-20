import json
from pathlib import Path

import pytest

from conftest import DummyDetector, make_result


def read_json(path):
    return json.loads(path.read_text())


def test_write_ultralytics_results_sorted_and_rounded(detector, tmp_path, yolo_result):
    json_path = tmp_path / "nested" / "detections.json"

    detector.write_detections_json([yolo_result], json_path)

    assert read_json(json_path) == {
        "camera.jpg": [
            {"class": "rat", "conf": 0.988},
            {"class": "mouse", "conf": 0.457},
        ]
    }


def test_write_ultralytics_preserves_existing_and_handles_empty_boxes(
    detector, tmp_path, empty_yolo_result
):
    json_path = tmp_path / "detections.json"
    json_path.write_text(json.dumps({"old.jpg": [{"class": "old", "conf": 1.0}]}))

    detector.write_detections_json([empty_yolo_result], json_path)

    assert read_json(json_path) == {
        "old.jpg": [{"class": "old", "conf": 1.0}],
        "empty.jpg": [],
    }


def test_write_speciesnet_results_known_unknown_and_missing_score(detector, tmp_path):
    json_path = tmp_path / "detections.json"
    results = {
        "predictions": [
            {
                "filepath": "/data/mouse.jpg",
                "prediction": "rodent",
                "prediction_score": 0.6543,
            },
            {
                "filepath": "/data/unknown.jpg",
                "prediction": "unknown",
                "prediction_score": 0.9,
            },
            {"filepath": "/data/no-score.jpg", "prediction": "rat"},
        ]
    }

    detector.write_detections_json(results, json_path)

    assert read_json(json_path) == {
        "mouse.jpg": [{"class": "rodent", "conf": 0.654}],
        "unknown.jpg": [],
        "no-score.jpg": [],
    }


def test_shorten_label_returns_item_unchanged(detector):
    assert detector.shorten_label("rodent") == "rodent"


def test_resolve_local_model_absolute_and_missing_relative(detector, tmp_path):
    absolute = tmp_path / "weights.pt"

    assert detector.resolve_local_model(absolute) == str(absolute)
    assert detector.resolve_local_model("missing.pt") == "missing.pt"


def test_resolve_local_model_finds_bundled_file(detector, tmp_path, monkeypatch):
    fake_base = tmp_path / "base.py"
    bundled = tmp_path / "weights.pt"
    fake_base.write_text("")
    bundled.write_text("weights")

    monkeypatch.setattr("smartrodent.base.__file__", str(fake_base))

    assert detector.resolve_local_model("weights.pt") == str(bundled)


def test_restore_result_paths_updates_lists_only(detector, sample_paths):
    results = [make_result("image0.jpg"), make_result("image1.jpg")]

    restored = detector.restore_result_paths(results, sample_paths[:2])

    assert restored is results
    assert [result.path for result in results] == [
        str(path.resolve()) for path in sample_paths[:2]
    ]

    unchanged = [make_result("single.jpg")]
    assert detector.restore_result_paths(unchanged, sample_paths[0]) is unchanged
    assert unchanged[0].path == "single.jpg"


def test_path_batches_single_path_list_chunks_and_invalid_batchsize(sample_paths):
    detector = DummyDetector(batchsize=2)

    assert list(detector.path_batches(sample_paths[0])) == [sample_paths[0]]
    assert list(detector.path_batches("image.jpg")) == ["image.jpg"]
    assert list(detector.path_batches(sample_paths)) == [
        sample_paths[:2],
        sample_paths[2:],
    ]

    bad_detector = DummyDetector(batchsize=0)
    with pytest.raises(ValueError, match="batchsize must be positive"):
        list(bad_detector.path_batches(sample_paths))
