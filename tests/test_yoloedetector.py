import json
from pathlib import Path

import pytest

import smartrodent.detection as detection
from smartrodent.detection import YOLOE_Detector


def test_yoloe_detector_requires_model_name():
    with pytest.raises(ValueError, match="model_name cannot be None"):
        YOLOE_Detector("run", False, 1, "boxed", 0.5)


def test_yoloe_detector_stores_configuration():
    detector = YOLOE_Detector(
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


def test_yoloe_detector_single_path_sets_classes_and_predicts_without_batch(
    monkeypatch, tmp_path, sample_paths, recording_model
):
    constructor_calls = []

    def fake_yoloe(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return recording_model

    monkeypatch.setattr(detection, "YOLOE", fake_yoloe)
    detector = YOLOE_Detector(
        "ignored",
        False,
        2,
        "boxed",
        0.35,
        model_name="weights.pt",
        classes=["mouse", "rat"],
        task="detect",
        project=str(tmp_path / "project"),
    )

    results = detector.detect(sample_paths[0], tmp_path / "out")

    assert constructor_calls == [(("weights.pt",), {"task": "detect"})]
    assert recording_model.get_text_pe_calls == [["mouse", "rat"]]
    assert recording_model.set_classes_calls == [(["mouse", "rat"], "text-pe")]
    assert len(results) == 1
    call = recording_model.predict_calls[0]
    assert call["source"] == str(sample_paths[0].resolve())
    assert "batch" not in call
    assert call["save"] is True
    assert call["project"] == str(tmp_path / "project")
    assert call["name"] == "boxed"
    assert call["exist_ok"] is True
    assert call["conf"] == 0.35
    assert call["save_crop"] is False
    assert (tmp_path / "out" / "detections.json").exists()


def test_yoloe_detector_list_input_skips_classes_chunks_and_writes_json(
    monkeypatch, tmp_path, sample_paths, recording_model
):
    monkeypatch.setattr(detection, "YOLOE", lambda *args, **kwargs: recording_model)
    detector = YOLOE_Detector(
        "ignored",
        True,
        2,
        "boxed",
        0.4,
        model_name=Path("weights.pt"),
        classes=None,
        project=str(tmp_path / "project"),
    )

    results = detector.detect(sample_paths, tmp_path / "out")

    assert recording_model.get_text_pe_calls == []
    assert recording_model.set_classes_calls == []
    assert [call["source"] for call in recording_model.predict_calls] == [
        sample_paths[:2],
        sample_paths[2:],
    ]
    assert [call["batch"] for call in recording_model.predict_calls] == [2, 1]
    assert [result.path for result in results] == [
        str(path.resolve()) for path in sample_paths
    ]
    assert json.loads((tmp_path / "out" / "detections.json").read_text()) == {
        "image_0.jpg": [{"class": "mouse", "conf": 0.8}],
        "image_1.jpg": [{"class": "rat", "conf": 0.8}],
        "image_2.jpg": [{"class": "mouse", "conf": 0.8}],
    }
