import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import smartrodent.inference as inference
from smartrodent.inference import SpeciesNetYoloInference


class FakeSpeciesNet:
    """Module-boundary fake for SpeciesNet's detector."""

    def __init__(self):
        self.result: dict | None = None
        self.filepaths: list[str] | None = None

    def detect(self, filepaths=None):
        self.filepaths = filepaths
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

    def __init__(self, probs, top1, top1conf, top5, top5conf):
        self.data = _FakeProbData(probs)
        self.top1 = top1
        self.top1conf = top1conf
        self.top5 = top5
        self.top5conf = _FakeProbData(top5conf)


class FakeClassResult:
    """Canned YOLO classification result exposing top-1 and top-5 data."""

    def __init__(self, names, probs, top1, top1conf, top5, top5conf):
        self.names = names
        self.probs = FakeClassProbs(probs, top1, top1conf, top5, top5conf)


class FakeYolo:
    """Module-boundary fake for ``YOLO`` used as the classifier."""

    def __init__(self, result=None):
        self.result = result
        self.last_call: tuple | None = None

    def __call__(self, *args, **kwargs):
        self.last_call = (args, kwargs)
        return [self.result]


@pytest.fixture
def fakes(monkeypatch):
    """Replace ``SpeciesNet``/``YOLO`` with recording fakes at import boundary."""

    species_calls: list[dict[str, object]] = []
    yolo_calls: list[dict[str, object]] = []
    hub_calls: list[dict[str, str]] = []
    detector = FakeSpeciesNet()
    yolo = FakeYolo()

    def fake_hf_hub_download(repo_id, filename):
        hub_calls.append({"repo_id": repo_id, "filename": filename})
        return str(Path("/fake-huggingface-cache") / filename)

    def fake_species(model, components=None, multiprocessing=None):
        species_calls.append(
            {
                "model": model,
                "components": components,
                "multiprocessing": multiprocessing,
            }
        )
        return detector

    def fake_yolo_ctor(weights, task=None):
        yolo_calls.append({"weights": weights, "task": task})
        return yolo

    monkeypatch.setattr(inference, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(inference, "SpeciesNet", fake_species)
    monkeypatch.setattr(inference, "YOLO", fake_yolo_ctor)
    return SimpleNamespace(
        hub_calls=hub_calls,
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


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_constructs_detector_and_downloaded_classifier(fakes):
    inst = SpeciesNetYoloInference(
        speciesnet_detector="det.pt", yolo_classifier="cls.pt"
    )

    assert fakes.hub_calls == [{"repo_id": "MaHaWo/Yolo26Rodent", "filename": "cls.pt"}]
    assert fakes.species_calls == [
        {
            "model": "det.pt",
            "components": "detector",
            "multiprocessing": False,
        }
    ]
    assert fakes.yolo_calls == [
        {"weights": "/fake-huggingface-cache/cls.pt", "task": "classify"}
    ]
    assert inst.detector is fakes.detector
    assert inst.classify is fakes.yolo


def test_init_coerces_path_objects_to_strings(fakes):
    SpeciesNetYoloInference(
        speciesnet_detector=Path("det.pt"), yolo_classifier=Path("cls.pt")
    )

    assert fakes.hub_calls == [{"repo_id": "MaHaWo/Yolo26Rodent", "filename": "cls.pt"}]
    assert fakes.species_calls == [
        {
            "model": "det.pt",
            "components": "detector",
            "multiprocessing": False,
        }
    ]
    assert fakes.yolo_calls == [
        {"weights": "/fake-huggingface-cache/cls.pt", "task": "classify"}
    ]


def test_init_uses_custom_classifier_repository(fakes):
    SpeciesNetYoloInference("det.pt", "models/cls.pt", repo_id="owner/models")

    assert fakes.hub_calls == [{"repo_id": "owner/models", "filename": "models/cls.pt"}]


# --------------------------------------------------------------------------- #
# from_config
# --------------------------------------------------------------------------- #


def test_from_config_missing_file_raises(fakes):
    with pytest.raises(ValueError, match="does not exist"):
        SpeciesNetYoloInference.from_config(str(Path("does_not_exist/nope.yaml")))


def test_from_config_builds_instance(fakes, tmp_path):
    cfg = _write_config(
        tmp_path,
        "speciesnet_model: md_v5a.pt\n"
        "classifier_weights: yolo.pt\n"
        "input_path: images\n"
        "output_path: results\n"
        "image_extensions: [.png]\n",
    )

    inst = SpeciesNetYoloInference.from_config(str(cfg))

    assert inst.detector is fakes.detector
    assert inst.classify is fakes.yolo
    assert fakes.hub_calls == [
        {"repo_id": "MaHaWo/Yolo26Rodent", "filename": "yolo.pt"}
    ]
    assert fakes.species_calls == [
        {
            "model": "md_v5a.pt",
            "components": "detector",
            "multiprocessing": False,
        }
    ]
    assert fakes.yolo_calls == [
        {"weights": "/fake-huggingface-cache/yolo.pt", "task": "classify"}
    ]


def test_from_config_empty_yaml_raises_missing_keys(fakes, tmp_path):
    cfg = _write_config(tmp_path, "# only a comment\n\n")

    with pytest.raises(ValueError, match="missing required key"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_missing_one_key_raises(fakes, tmp_path):
    cfg = _write_config(tmp_path, "speciesnet_model: model.pt\n")

    with pytest.raises(ValueError, match="classifier_weights"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_null_values_raise_missing_keys(fakes, tmp_path):
    # Model values are required even when runtime settings are present.
    cfg = _write_config(tmp_path, "speciesnet_model:\nclassifier_weights:\n")

    with pytest.raises(ValueError, match="missing required key"):
        SpeciesNetYoloInference.from_config(str(cfg))


def test_from_config_non_mapping_raises(fakes, tmp_path):
    cfg = _write_config(tmp_path, "- a\n- b\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        SpeciesNetYoloInference.from_config(str(cfg))


# --------------------------------------------------------------------------- #
# _detect
# --------------------------------------------------------------------------- #


def test_detect_no_predictions_returns_none_pair(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": []}

    assert inst._detect(tiny_image) == (None, None)


def test_detect_prediction_without_detections_returns_none_pair(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{}]}

    assert inst._detect(tiny_image) == (None, None)


def test_detect_empty_detections_returns_none_pair(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{"detections": []}]}

    assert inst._detect(tiny_image) == (None, None)


def test_detect_single_detection_returns_rgb_crop_and_bbox(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [
            {"detections": [{"conf": 0.9, "bbox": [0.25, 0.5, 0.25, 0.25]}]}
        ]
    }

    crops, bboxs = inst._detect(tiny_image)

    assert crops is not None
    assert crops[0].mode == "RGB"
    # tiny_image is 100x80; bbox normalized (0.25, 0.5, 0.25, 0.25) => (25, 40, 50, 60).
    assert crops[0].size == (25, 20)
    assert bboxs == [(0.25, 0.5, 0.25, 0.25)]


def test_detect_returns_detections_in_descending_confidence_order(fakes, tiny_image):
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

    crops, bboxs = inst._detect(tiny_image)

    assert crops is not None
    assert [crop.size for crop in crops] == [(50, 40), (20, 16)]
    assert bboxs == [(0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 0.2, 0.2)]


def test_detect_limits_results_to_five(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [
            {
                "detections": [
                    {"conf": confidence, "bbox": [0.0, 0.0, 0.1, 0.1]}
                    for confidence in range(6)
                ]
            }
        ]
    }

    crops, bboxs = inst._detect(tiny_image)

    assert crops is not None
    assert len(crops) == 5
    assert bboxs is not None
    assert len(bboxs) == 5


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

    crops, bboxs = inst._detect(tiny_image)

    assert crops is not None
    assert [crop.size for crop in crops] == [(50, 40)]
    assert bboxs == [(0.5, 0.5, 0.5, 0.5)]


def test_detect_all_malformed_returns_none_pair(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {
        "predictions": [{"detections": [{"conf": 0.9}, {"bbox": [0, 0, 1, 1]}]}]
    }

    assert inst._detect(tiny_image) == (None, None)


def test_detect_resolves_relative_image_path(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": [{"detections": []}]}

    inst._detect(Path("some/relative.png"))

    assert fakes.detector.filepaths == [str(Path("some/relative.png").resolve())]


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


def test_classify_builds_top1_and_top5_results(fakes):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.yolo.result = FakeClassResult(
        ["rodent", "bird", "plant"],
        [0.1, 0.2, 0.7],
        top1=2,
        top1conf=0.7,
        top5=[2, 1, 0],
        top5conf=[0.7, 0.2, 0.1],
    )
    crop = Image.new("RGB", (10, 10), "green")

    result = inst._classify(crop)

    assert result == {
        "probabilities": {"rodent": 0.1, "bird": 0.2, "plant": 0.7},
        "result": "plant",
        "confidence": 0.7,
        "top5": ["plant", "bird", "rodent"],
        "top5_confidence": [0.7, 0.2, 0.1],
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
    fakes.yolo.result = FakeClassResult(
        ["animal"],
        [1.0],
        top1=0,
        top1conf=1.0,
        top5=[0],
        top5conf=[1.0],
    )

    result = inst.predict(tiny_image)

    assert result == {
        0: {
            "probabilities": {"animal": 1.0},
            "result": "animal",
            "confidence": 1.0,
            "top5": ["animal"],
            "top5_confidence": [1.0],
            "bbox": (0, 0, 1, 1),
        }
    }


def test_predict_no_detection_returns_empty_mapping(fakes, tiny_image):
    inst = SpeciesNetYoloInference("det.pt", "cls.pt")
    fakes.detector.result = {"predictions": []}

    assert inst.predict(tiny_image) == {}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def test_main_reads_input_output_and_extensions_from_config(monkeypatch, tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "keep.PNG").write_bytes(b"image")
    (input_dir / "ignore.txt").write_text("not an image", encoding="utf-8")
    cfg = _write_config(
        tmp_path,
        "speciesnet_model: model.pt\n"
        "classifier_weights: classifier.pt\n"
        "path: images\n"
        "output: results\n"
        "imgs: [.png]\n",
    )
    pipeline = SimpleNamespace(predict=lambda image: {"source": image.name})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SpeciesNetYoloInference, "from_config", lambda config: pipeline)
    monkeypatch.setattr("sys.argv", ["inference", "--config", str(cfg)])

    inference.main()

    results = json.loads((tmp_path / "results" / "results.json").read_text())
    assert results == {"keep.PNG": {"source": "keep.PNG"}}
