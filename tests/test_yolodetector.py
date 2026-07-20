import json

import pytest

import smartrodent.detection as detection
from smartrodent.detection import YOLO_Detector


def test_yolo_detector_requires_model_name():
    with pytest.raises(ValueError, match="model_name cannot be None"):
        YOLO_Detector("run", False, 1, "boxed", 0.5)


def test_yolo_detector_stores_configuration():
    detector = YOLO_Detector(
        "run",
        True,
        3,
        "boxed",
        0.25,
        relpad=0.1,
        model_name="weights.pt",
        classes=["mouse"],
        task="detect",
        project="project",
    )

    assert detector.name == "run"
    assert detector.crop is True
    assert detector.batchsize == 3
    assert detector.conf == 0.25
    assert detector.relpad == 0.1
    assert detector.model_name == "weights.pt"
    assert detector.classes == ["mouse"]
    assert detector.task == "detect"
    assert detector.project == "project"


def test_yolo_detector_detect_chunks_predicts_and_writes_json(
    monkeypatch, tmp_path, sample_paths, recording_model
):
    constructor_calls = []

    def fake_yolo(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return recording_model

    monkeypatch.setattr(detection, "YOLO", fake_yolo)
    detector = YOLO_Detector(
        "run",
        True,
        2,
        "boxed",
        0.25,
        model_name="weights.pt",
        task="detect",
        project=str(tmp_path / "project"),
    )

    results = detector.detect(sample_paths, tmp_path / "out")

    assert constructor_calls == [(("weights.pt",), {"task": "detect"})]
    assert len(results) == 3
    assert [call["source"] for call in recording_model.predict_calls] == [
        sample_paths[:2],
        sample_paths[2:],
    ]
    assert [call["batch"] for call in recording_model.predict_calls] == [2, 1]
    for call in recording_model.predict_calls:
        assert call["save"] is True
        assert call["project"] == str(tmp_path / "project")
        assert call["name"] == "run"
        assert call["exist_ok"] is True
        assert call["conf"] == 0.25
        assert call["save_crop"] is True

    assert [result.path for result in results] == [
        str(path.resolve()) for path in sample_paths
    ]
    assert json.loads((tmp_path / "out" / "detections.json").read_text()) == {
        "image_0.jpg": [{"class": "mouse", "conf": 0.8}],
        "image_1.jpg": [{"class": "rat", "conf": 0.8}],
        "image_2.jpg": [{"class": "mouse", "conf": 0.8}],
    }
