from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import smartrodent.inference as inference
from smartrodent.inference import SpeciesNetYoloInference


class FakeSpeciesNet:
    """Module-boundary fake for SpeciesNet's predictor."""

    def __init__(self):
        self.result: dict | None = None
        self.filepaths: list[str] | None = None
        self.batch_size: int | None = None

    def predict(self, filepaths=None, batch_size=None):
        self.filepaths = filepaths
        self.batch_size = batch_size
        return self.result


class _FakeProbData:
    def __init__(self, probs):
        self._probs = probs

    def cpu(self):
        return self

    def tolist(self):
        return self._probs


class FakeClassProbs:
    """Shape-matches ``YOLO.probs`` for the classifier result."""

    def __init__(self, probs, top1, top1conf):
        self.data = _FakeProbData(probs)
        self.top1 = top1
        self.top1conf = top1conf


class FakeClassResult:
    """Canned YOLO-classification result exposing names/probs/top1/top1conf."""

    def __init__(self, names, probs, top1, top1conf):
        self.names = names
        self.probs = FakeClassProbs(probs, top1, top1conf)


class FakeYolo:
    """Module-boundary fake for ``YOLO`` used as the classifier."""

    def __init__(self, result=None):
        self.result = result
        self.last_call: tuple | None = None

    def __call__(self, *args, **kwargs):
        self.last_call = (args, kwargs)
        return self.result


@pytest.fixture
def fakes(monkeypatch):
    """Replace ``SpeciesNet``/``YOLO`` with recording fakes at import boundary."""

    species_calls: list[dict[str, object]] = []
    yolo_calls: list[dict[str, object]] = []
    detector = FakeSpeciesNet()
    yolo = FakeYolo()

    def fake_species(model, components=None):
        species_calls.append({"model": model, "components": components})
        return detector

    def fake_yolo_ctor(weights, task=None):
        yolo_calls.append({"weights": weights, "task": task})
        return yolo

    monkeypatch.setattr(inference, "SpeciesNet", fake_species)
    monkeypatch.setattr(inference, "YOLO", fake_yolo_ctor)
    return SimpleNamespace(
        species_calls=species_calls,
        yolo_calls=yolo_calls,
        detector=detector,
        yolo=yolo,
    )


@pytest.fixture
def tiny_image(tmp_path):
    """Writes a real small RGB PNG to disk for detection cropping."""

    path = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), "red").save(path)
    return path


def _write_config(tmp_path, content, name="config.yaml"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _conf_detection(result):
    fakes.result = None if result is None else {"predictions": result}


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_constructs_detector_and_classifier(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")

    assert fakes.species_calls == [{"model": "det.pt", "components": "detector"}]
    assert fakes.yolo_calls == [{"weights": "cls.pt", "task": "classify"}]
    assert inst.detector is fakes.detector
    assert inst.classify is fakes.yolo


def test_init_coerces_path_objects_to_strings(fakes):
    SpeciesNetYoloInference(Path("det.pt"), Path("cls.pt"))

    assert fakes.species_calls == [{"model": "det.pt", "components": "detector"}]
    assert fakes.yolo_calls == [{"weights": "cls.pt", "task": "classify"}]


# --------------------------------------------------------------------------- #
# from_config
# --------------------------------------------------------------------------- #


def test_from_config_missing_file_raises(fakes):
    with pytest.raises(ValueError, match="does not exist"):
        SpeciesNetYoloInference.from_config(str(Path("does_not_exist/nope.yaml")))


def test_from_config_builds_instance(fakes, tmp_path):
    cfg = _write_config(
        tmp_path, "speciesnet_model: md_v5a.pt\nclassifier_weights: yolo.pt\n"
    )

    inst = SpeciesNetYoloInference.from_config(str(cfg))

    assert inst.detector is fakes.detector
    assert inst.classify is fakes.yolo
    assert fakes.species_calls == [{"model": "md_v5a.pt", "components": "detector"}]
    assert fakes.yolo_calls == [{"weights": "yolo.pt", "task": "classify"}]


def test_from_config_empty_yaml_raises_missing_keys(fakes, tmp_path):
    cfg = _write_config(tmp_path, "# only a comment\n\n")

    with pytest.raises(ValueError, match="missing required key"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_missing_one_key_raises(fakes, tmp_path):
    cfg = _write_config(tmp_path, "speciesnet_model: model.pt\n")

    with pytest.raises(ValueError, match="classifier_weights"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_null_values_raise_missing_keys(fakes, tmp_path):
    # Mirrors the committed configs/inference_pipeline.yaml (null values).
    cfg = _write_config(tmp_path, "speciesnet_model:\nclassifier_weights:\n")

    with pytest.raises(ValueError, match="missing required key"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_extra_key_raises(fakes, tmp_path):
    cfg = _write_config(
        tmp_path,
        "speciesnet_model: model.pt\nclassifier_weights: yolo.pt\nunknown: x\n",
    )

    with pytest.raises(ValueError, match="unexpected key"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_non_mapping_raises(fakes, tmp_path):
    cfg = _write_config(tmp_path, "- a\n- b\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        SpeciesNetYoloInference.from_config(str(cfg))


# --------------------------------------------------------------------------- #
# _detect
# --------------------------------------------------------------------------- #


def test_detect_no_predictions_returns_none(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": []}

    assert inst._detect(tiny_image) is None


def test_detect_prediction_without_detections_returns_none(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{}]}

    assert inst._detect(tiny_image) is None


def test_detect_empty_detections_returns_none(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{"detections": []}]}

    assert inst._detect(tiny_image) is None


def test_detect_single_detection_returns_rgb_crop(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [
            {"detections": [{"conf": 0.9, "bbox": [0.25, 0.5, 0.25, 0.25]}]}
        ]
    }

    crop = inst._detect(tiny_image)

    assert crop is not None
    assert crop.mode == "RGB"
    # tiny_image is 100x80; bbox normalized (0.25, 0.5, 0.25, 0.25) => (25, 40, 50, 60).
    assert crop.size == (25, 20)


def test_detect_selects_highest_confidence(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [
            {
                "detections": [
                    {"conf": 0.4, "bbox": [0.0, 0.0, 0.2, 0.2]},
                    {"conf": 0.9, "bbox": [0.5, 0.5, 0.5, 0.5]},
                ]
            }
        ]
    }

    crop = inst._detect(tiny_image)

    assert crop is not None
    assert crop.size == (50, 40)


def test_detect_skips_malformed_detections(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [
            {
                "detections": [
                    {"conf": 0.9, "bbox": [0.5, 0.5, 0.5, 0.5]},
                    {"conf": 0.8},  # missing bbox
                    {"bbox": [0.0, 0.0, 1.0, 1.0]},  # missing conf
                    {"conf": 0.7, "bbox": [0.0, 0.0, 0.1]},  # wrong bbox length
                    {"conf": 0.6, "bbox": "not-a-box"},  # wrong bbox type
                ]
            }
        ]
    }

    crop = inst._detect(tiny_image)

    assert crop is not None
    assert crop.size == (50, 40)


def test_detect_all_malformed_returns_none(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [{"detections": [{"conf": 0.9}, {"bbox": [0, 0, 1, 1]}]}]
    }

    assert inst._detect(tiny_image) is None


def test_detect_resolves_relative_image_path(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{"detections": []}]}

    inst._detect(Path("some/relative.png"))

    assert fakes.detector.filepaths == [str(Path("some/relative.png").resolve())]
    assert fakes.detector.batch_size == 1


# --------------------------------------------------------------------------- #
# _classify
# --------------------------------------------------------------------------- #


def test_classify_none_crop_returns_sentinel(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")

    assert inst._classify(None) == {
        "probabilities": {},
        "result": None,
        "confidence": None,
    }


def test_classify_builds_probabilities_result_and_confidence(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.yolo.result = FakeClassResult(
        ["rodent", "bird", "plant"], [0.1, 0.2, 0.7], top1=2, top1conf=0.7
    )
    crop = Image.new("RGB", (10, 10), "green")

    result = inst._classify(crop)

    assert result == {
        "probabilities": {"rodent": 0.1, "bird": 0.2, "plant": 0.7},
        "result": "plant",
        "confidence": 0.7,
    }
    assert fakes.yolo.last_call == ((crop,), {"verbose": False})


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #


def test_predict_happy_path(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [{"detections": [{"conf": 0.9, "bbox": [0, 0, 1, 1]}]}]
    }
    fakes.yolo.result = FakeClassResult(["animal"], [1.0], top1=0, top1conf=1.0)

    result = inst.predict(tiny_image)

    assert result == {
        "probabilities": {"animal": 1.0},
        "result": "animal",
        "confidence": 1.0,
    }


def test_predict_no_detection_returns_sentinel(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": []}

    assert inst.predict(tiny_image) == {
        "probabilities": {},
        "result": None,
        "confidence": None,
    }
