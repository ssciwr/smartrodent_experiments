from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml

from smartrodent.inference import (
    Inference,
    SpeciesNetClassifier,
    SpeciesNetDetector,
    YoloClassifier,
    YoloDetector,
    _crop_normalized_bbox,
    _load_inference_config,
    _validate_max_top_k,
    load_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CONFIGS = [
    "inference_yolo26_speciesnames.yaml",
    "inference_yolo26_genusnames_224.yaml",
    "inference_yolo26_genusnames_640.yaml",
    "inference_yolo26_commonnames.yaml",
]


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeIndexList(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return FakeIndexList(value) if isinstance(index, slice) else value

    def tolist(self):
        return list(self)


class FakeScores:
    def __init__(self, values):
        self.values = values
        self.detach_called = False
        self.cpu_called = False
        self.argsort_calls = []

    def detach(self):
        self.detach_called = True
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def __len__(self):
        return len(self.values)

    def argsort(self, *, descending):
        self.argsort_calls.append(descending)
        return FakeIndexList(
            sorted(range(len(self.values)), key=self.values.__getitem__, reverse=descending)
        )

    def __getitem__(self, index):
        return FakeScalar(self.values[index])


class FakeBox:
    def __init__(self, corners, confidence):
        self.xyxyn = [SimpleNamespace(tolist=lambda: corners)]
        self.conf = FakeScalar(confidence)


@pytest.fixture
def fake_speciesnet(monkeypatch):
    """Replace SpeciesNet loaders with predictable in-memory collaborators."""

    state = SimpleNamespace(
        detector_preprocess_calls=[],
        detector_predict_calls=[],
        classifier_preprocess_calls=[],
        classifier_predict_calls=[],
        detector_prediction={"detections": []},
        classifier_prediction={"classifications": {"classes": [], "scores": []}},
        speciesnet_constructor_calls=[],
        classifier_constructor_calls=[],
    )

    class FakeDetectorModel:
        def preprocess(self, image):
            state.detector_preprocess_calls.append(image)
            return ("detector-preprocessed", image)

        def predict(self, path, preprocessed):
            state.detector_predict_calls.append((path, preprocessed))
            return state.detector_prediction

    class FakeClassifierModel:
        def preprocess(self, image):
            state.classifier_preprocess_calls.append(image)
            return ("classifier-preprocessed", image)

        def predict(self, path, preprocessed):
            state.classifier_predict_calls.append((path, preprocessed))
            return state.classifier_prediction

    class FakeSpeciesNet:
        def __init__(self, model, *, components, geofence):
            state.speciesnet_constructor_calls.append(
                {"model": model, "components": components, "geofence": geofence}
            )
            self.detector = FakeDetectorModel()

    class FakeSpeciesNetClassifier:
        def __init__(self, model, **kwargs):
            state.classifier_constructor_calls.append({"model": model, "kwargs": kwargs})
            self.model = FakeClassifierModel()

        def preprocess(self, image):
            return self.model.preprocess(image)

        def predict(self, path, preprocessed):
            return self.model.predict(path, preprocessed)

    monkeypatch.setattr("smartrodent.inference.speciesnet.SpeciesNet", FakeSpeciesNet)
    monkeypatch.setattr(
        "smartrodent.inference.speciesnet.SpeciesNetClassifier", FakeSpeciesNetClassifier
    )
    return state


@pytest.fixture
def fake_yolo(monkeypatch):
    """Replace ultralytics.YOLO while retaining its observable wrapper protocol."""

    state = SimpleNamespace(constructor_calls=[], prediction_calls=[], results={})

    class FakeYOLO:
        def __init__(self, model, *, task):
            self.model = model
            self.task = task
            state.constructor_calls.append({"model": model, "task": task})

        def predict(self, *, source, **kwargs):
            state.prediction_calls.append(
                {"task": self.task, "source": source, "kwargs": kwargs}
            )
            return [state.results[self.task]]

    monkeypatch.setattr("smartrodent.inference.ultralytics.YOLO", FakeYOLO)
    return state


@pytest.fixture
def image():
    return Image.new("RGB", (100, 50))


def write_inference_config(tmp_path, **overrides):
    config = {
        "detector": "speciesnet",
        "classifier": "speciesnet",
        "detector_kwargs": {"model": "detector-model", "conf": 0.25},
        "classifier_kwargs": {"model": "classifier-model", "max_top_k": 3},
        "classes": ["mouse"],
        "class_mappings": {"house mouse": "mouse"},
    }
    config.update(overrides)
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump({"inference": config}))
    return path


def test_load_image_returns_rgb_image_detached_from_file(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (12, 8), color=128).save(image_path)

    loaded_image = load_image(str(image_path))
    image_path.unlink()

    assert loaded_image.mode == "RGB"
    assert loaded_image.size == (12, 8)
    assert loaded_image.getpixel((0, 0)) == (128, 128, 128)


@pytest.mark.parametrize("max_top_k", [0, -1, 1.5, True, None])
def test_max_top_k_must_be_a_positive_integer(max_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        _validate_max_top_k(max_top_k)


def test_crop_normalized_bbox_uses_image_coordinates(image):
    crop = _crop_normalized_bbox(image, [0.1, 0.2, 0.5, 0.4])

    assert crop is not None
    assert crop.size == (50, 20)


def test_crop_normalized_bbox_clips_to_image_bounds(image):
    crop = _crop_normalized_bbox(image, [-0.1, -0.2, 0.3, 0.5])

    assert crop is not None
    assert crop.size == (20, 15)


def test_crop_normalized_bbox_rejects_empty_crop(image):
    assert _crop_normalized_bbox(image, [0.5, 0.5, 0.0, 0.0]) is None


@pytest.mark.parametrize("config_name", INFERENCE_CONFIGS)
def test_inference_configs_reference_available_classifier_models(config_name):
    config_path = PROJECT_ROOT / "configs" / config_name

    config = _load_inference_config(config_path)

    assert config["detector_kwargs"]["model"] == (
        "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
    )
    assert config["detector_kwargs"]["conf"] == 0.01
    assert Path(config["classifier_kwargs"]["model"]).is_file()
    assert config["classifier_kwargs"]["max_top_k"] == 5
    assert config["classes"]


@pytest.mark.parametrize(
    ("wrapper_class", "component", "expected_task"),
    [
        (SpeciesNetDetector, "detector", None),
        (YoloDetector, "detector", "detect"),
        (SpeciesNetClassifier, "classifier", None),
        (YoloClassifier, "classifier", "classify"),
    ],
)
def test_wrapper_from_config_selects_matching_backend(
    tmp_path, fake_speciesnet, fake_yolo, wrapper_class, component, expected_task
):
    backend = "speciesnet" if wrapper_class.__name__.startswith("SpeciesNet") else "yolo26"
    config_path = write_inference_config(tmp_path, **{component: backend})

    wrapper = wrapper_class.from_config(config_path)

    assert isinstance(wrapper, wrapper_class)
    if expected_task is None:
        if component == "detector":
            assert fake_speciesnet.speciesnet_constructor_calls[0]["model"] == "detector-model"
        else:
            assert fake_speciesnet.classifier_constructor_calls[0]["model"] == "classifier-model"
    else:
        assert fake_yolo.constructor_calls == [
            {
                "model": f"{component}-model",
                "task": expected_task,
            }
        ]


@pytest.mark.parametrize(
    ("wrapper_class", "component", "wrong_backend"),
    [
        (SpeciesNetDetector, "detector", "yolo26"),
        (YoloDetector, "detector", "speciesnet"),
        (SpeciesNetClassifier, "classifier", "yolo26"),
        (YoloClassifier, "classifier", "speciesnet"),
    ],
)
def test_wrapper_from_config_rejects_other_backend(
    tmp_path, wrapper_class, component, wrong_backend
):
    config_path = write_inference_config(tmp_path, **{component: wrong_backend})

    with pytest.raises(ValueError, match="does not select"):
        wrapper_class.from_config(config_path)


def test_speciesnet_detector_copies_kwargs_and_loads_detector(fake_speciesnet):
    kwargs = {"conf": 0.4}

    detector = SpeciesNetDetector("detector-model", kwargs=kwargs)
    kwargs["conf"] = 0.9

    assert detector.kwargs == {"conf": 0.4}
    assert fake_speciesnet.speciesnet_constructor_calls == [
        {"model": "detector-model", "components": "detector", "geofence": False}
    ]


def test_speciesnet_detector_json_filters_sorts_and_limits(fake_speciesnet, image):
    fake_speciesnet.detector_prediction = {
        "detections": [
            {"bbox": [0, 0, 1, 1], "conf": 0.4},
            {"bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.9},
            {"bbox": [0, 0, 0.1, 0.1], "conf": 0.2},
        ]
    }
    detector = SpeciesNetDetector("detector-model", kwargs={"conf": 0.3})

    result = detector.detect_json(image, max_top_k=2)

    assert result == {
        "detections": [
            {"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9},
            {"bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 0.4},
        ]
    }
    assert fake_speciesnet.detector_preprocess_calls == [image]
    assert fake_speciesnet.detector_predict_calls == [
        ("", ("detector-preprocessed", image))
    ]


@pytest.mark.parametrize("max_top_k", [0, True, 1.2])
def test_speciesnet_detector_rejects_invalid_top_k_before_predicting(
    fake_speciesnet, image, max_top_k
):
    detector = SpeciesNetDetector("detector-model")

    with pytest.raises(ValueError, match="positive integer"):
        detector.detect_json(image, max_top_k)

    assert fake_speciesnet.detector_predict_calls == []


def test_speciesnet_detector_detect_crops_valid_boxes_and_call_delegates(
    fake_speciesnet, image
):
    fake_speciesnet.detector_prediction = {
        "detections": [
            {"bbox": [0.1, 0.2, 0.5, 0.4], "conf": 0.9},
            {"bbox": [0.5, 0.5, 0, 0], "conf": 0.8},
        ]
    }
    detector = SpeciesNetDetector("detector-model")

    detections = detector(image, max_top_k=2)

    assert [(crop.size, confidence) for crop, confidence in detections] == [
        ((50, 20), 0.9)
    ]


def test_yolo_detector_copies_kwargs_and_loads_detect_model(fake_yolo):
    kwargs = {"conf": 0.4, "device": "cpu"}

    detector = YoloDetector("detector.pt", kwargs=kwargs)
    kwargs["conf"] = 0.9

    assert detector.kwargs == {"conf": 0.4, "device": "cpu"}
    assert fake_yolo.constructor_calls == [{"model": "detector.pt", "task": "detect"}]


def test_yolo_detector_json_converts_corners_sorts_and_forwards_kwargs(fake_yolo, image):
    fake_yolo.results["detect"] = SimpleNamespace(
        boxes=[
            FakeBox([0.1, 0.2, 0.4, 0.6], 0.4),
            FakeBox([0.2, 0.1, 0.8, 0.9], 0.9),
        ]
    )
    detector = YoloDetector("detector.pt", kwargs={"conf": 0.2, "verbose": False})

    result = detector.detect_json(image, max_top_k=1)

    assert result == {
        "detections": [{"bbox": [0.2, 0.1, 0.6000000000000001, 0.8], "confidence": 0.9}]
    }
    assert fake_yolo.prediction_calls == [
        {
            "task": "detect",
            "source": image,
            "kwargs": {"conf": 0.2, "verbose": False},
        }
    ]


@pytest.mark.parametrize("max_top_k", [0, -2, None])
def test_yolo_detector_rejects_invalid_top_k_before_predicting(fake_yolo, image, max_top_k):
    detector = YoloDetector("detector.pt")

    with pytest.raises(ValueError, match="positive integer"):
        detector.detect_json(image, max_top_k)

    assert fake_yolo.prediction_calls == []


def test_yolo_detector_detect_crops_results_and_omits_empty_boxes(fake_yolo, image):
    fake_yolo.results["detect"] = SimpleNamespace(
        boxes=[FakeBox([0.1, 0.2, 0.6, 0.6], 0.9), FakeBox([0.5, 0.5, 0.5, 0.5], 0.8)]
    )
    detector = YoloDetector("detector.pt")

    detections = detector.detect(image, max_top_k=2)

    assert [(crop.size, confidence) for crop, confidence in detections] == [
        ((50, 20), 0.9)
    ]


@pytest.mark.parametrize("max_top_k", [0, -1, True, 6])
def test_speciesnet_classifier_rejects_invalid_constructor_max_top_k(
    fake_speciesnet, max_top_k
):
    match = "at most five" if max_top_k == 6 else "positive integer"

    with pytest.raises(ValueError, match=match):
        SpeciesNetClassifier("classifier-model", max_top_k=max_top_k)

    assert fake_speciesnet.classifier_constructor_calls == []


def test_speciesnet_classifier_copies_kwargs_and_forwards_them_to_loader(fake_speciesnet):
    kwargs = {"device": "cpu"}

    classifier = SpeciesNetClassifier("classifier-model", max_top_k=3, kwargs=kwargs)
    kwargs["device"] = "cuda"

    assert classifier.kwargs == {"device": "cpu"}
    assert classifier.max_top_k == 3
    assert fake_speciesnet.classifier_constructor_calls == [
        {"model": "classifier-model", "kwargs": {"device": "cpu"}}
    ]


def test_speciesnet_classifier_json_sorts_pairs_and_limits(fake_speciesnet, image):
    fake_speciesnet.classifier_prediction = {
        "classifications": {
            "classes": ["rat", "mouse", "shrew"],
            "scores": [0.2, 0.9, 0.4],
        }
    }
    classifier = SpeciesNetClassifier("classifier-model", max_top_k=2)

    result = classifier.classify_json(image, max_top_k=2)

    assert result == {
        "classifications": [
            {"class": "mouse", "confidence": 0.9},
            {"class": "shrew", "confidence": 0.4},
        ]
    }
    assert fake_speciesnet.classifier_preprocess_calls == [image]
    assert fake_speciesnet.classifier_predict_calls == [
        ("", ("classifier-preprocessed", image))
    ]


@pytest.mark.parametrize("max_top_k", [0, 3])
def test_speciesnet_classifier_rejects_invalid_request_before_predicting(
    fake_speciesnet, image, max_top_k
):
    classifier = SpeciesNetClassifier("classifier-model", max_top_k=2)

    with pytest.raises(ValueError):
        classifier.classify_json(image, max_top_k)

    assert fake_speciesnet.classifier_predict_calls == []


def test_speciesnet_classifier_classify_and_call_return_tuples(fake_speciesnet, image):
    fake_speciesnet.classifier_prediction = {
        "classifications": {"classes": ["mouse"], "scores": [0.9]}
    }
    classifier = SpeciesNetClassifier("classifier-model")

    assert classifier(image) == [("mouse", 0.9)]


def test_yolo_classifier_copies_kwargs_and_loads_classify_model(fake_yolo):
    kwargs = {"imgsz": 224, "verbose": False}

    classifier = YoloClassifier("classifier.pt", max_top_k=3, kwargs=kwargs)
    kwargs["imgsz"] = 640

    assert classifier.kwargs == {"imgsz": 224, "verbose": False}
    assert classifier.max_top_k == 3
    assert fake_yolo.constructor_calls == [{"model": "classifier.pt", "task": "classify"}]


def test_yolo_classifier_json_moves_scores_to_cpu_sorts_and_limits(fake_yolo, image):
    scores = FakeScores([0.2, 0.9, 0.4])
    fake_yolo.results["classify"] = SimpleNamespace(
        probs=SimpleNamespace(data=scores), names={0: "rat", 1: "mouse", 2: "shrew"}
    )
    classifier = YoloClassifier("classifier.pt", max_top_k=3, kwargs={"imgsz": 224})

    result = classifier.classify_json(image, max_top_k=2)

    assert result == {
        "classifications": [
            {"class": "mouse", "confidence": 0.9},
            {"class": "shrew", "confidence": 0.4},
        ]
    }
    assert scores.detach_called and scores.cpu_called
    assert scores.argsort_calls == [True]
    assert fake_yolo.prediction_calls == [
        {"task": "classify", "source": image, "kwargs": {"imgsz": 224}}
    ]


def test_yolo_classifier_caps_result_to_available_scores(fake_yolo, image):
    fake_yolo.results["classify"] = SimpleNamespace(
        probs=SimpleNamespace(data=FakeScores([0.8])), names={0: "mouse"}
    )
    classifier = YoloClassifier("classifier.pt", max_top_k=3)

    assert classifier.classify_json(image, max_top_k=3) == {
        "classifications": [{"class": "mouse", "confidence": 0.8}]
    }


@pytest.mark.parametrize("max_top_k", [0, True, 3])
def test_yolo_classifier_rejects_invalid_or_excessive_request_before_predicting(
    fake_yolo, image, max_top_k
):
    classifier = YoloClassifier("classifier.pt", max_top_k=2)

    with pytest.raises(ValueError):
        classifier.classify_json(image, max_top_k)

    assert fake_yolo.prediction_calls == []


def test_yolo_classifier_call_returns_tuple_classifications(fake_yolo, image):
    fake_yolo.results["classify"] = SimpleNamespace(
        probs=SimpleNamespace(data=FakeScores([0.7])), names={0: "mouse"}
    )
    classifier = YoloClassifier("classifier.pt")

    assert classifier(image) == [("mouse", 0.7)]


def test_inference_constructs_case_insensitive_backends_without_mutating_inputs(
    fake_speciesnet, fake_yolo
):
    detector_kwargs = {"model": "species-detector", "conf": 0.3}
    classifier_kwargs = {"model": "yolo-classifier.pt", "max_top_k": 2, "imgsz": 224}
    classes = ["mouse"]
    mappings = {"house mouse": "mouse"}

    inference = Inference(
        "SPECIESNET",
        "YOLO26",
        detector_kwargs,
        classifier_kwargs,
        classes,
        mappings,
    )
    detector_kwargs["conf"] = 0.9
    classifier_kwargs["imgsz"] = 640
    classes.append("rat")
    mappings["rat"] = "rat"

    assert isinstance(inference.detector, SpeciesNetDetector)
    assert isinstance(inference.classifier, YoloClassifier)
    assert inference.detector.kwargs == {"conf": 0.3}
    assert inference.classifier.kwargs == {"imgsz": 224}
    assert inference.classifier.max_top_k == 2
    assert inference.classes == ["mouse"]
    assert inference.class_mappings == {"house mouse": "mouse"}


@pytest.mark.parametrize(
    ("detector", "classifier", "detector_kwargs", "classifier_kwargs", "match"),
    [
        ("unknown", "yolo26", {"model": "d"}, {"model": "c"}, "Unsupported detector"),
        ("yolo26", "unknown", {"model": "d"}, {"model": "c"}, "Unsupported classifier"),
        ("yolo26", "yolo26", {}, {"model": "c"}, "detector_kwargs"),
        ("yolo26", "yolo26", {"model": "d"}, {}, "classifier_kwargs"),
    ],
)
def test_inference_rejects_invalid_backend_or_missing_model(
    detector, classifier, detector_kwargs, classifier_kwargs, match
):
    with pytest.raises(ValueError, match=match):
        Inference(detector, classifier, detector_kwargs, classifier_kwargs)


def test_inference_from_config_resolves_local_models_and_builds_pipeline(
    tmp_path, fake_yolo
):
    detector_model = tmp_path / "detector.pt"
    classifier_model = tmp_path / "classifier.pt"
    detector_model.touch()
    classifier_model.touch()
    config_path = write_inference_config(
        tmp_path,
        detector="yolo26",
        classifier="yolo26",
        detector_kwargs={"model": detector_model.name, "conf": 0.2},
        classifier_kwargs={"model": classifier_model.name, "max_top_k": 2, "imgsz": 224},
    )

    inference = Inference.from_config(config_path)

    assert isinstance(inference.detector, YoloDetector)
    assert isinstance(inference.classifier, YoloClassifier)
    assert fake_yolo.constructor_calls == [
        {"model": str(detector_model.resolve()), "task": "detect"},
        {"model": str(classifier_model.resolve()), "task": "classify"},
    ]


def test_inference_runs_each_crop_maps_then_filters_classes(monkeypatch, image):
    crops = [Image.new("RGB", (10, 10)), Image.new("RGB", (20, 20))]

    class RecordingDetector:
        def __init__(self, model, *, kwargs):
            assert model == "detector-model"
            assert kwargs == {"conf": 0.1}

        def detect(self, received_image, max_top_k):
            assert received_image is image
            assert max_top_k == 2
            return [(crops[0], 0.8), (crops[1], 0.6)]

    class RecordingClassifier:
        def __init__(self, model, max_top_k, *, kwargs):
            assert model == "classifier-model"
            assert max_top_k == 3
            assert kwargs == {"device": "cpu"}
            self.calls = []

        def classify(self, crop, max_top_k):
            self.calls.append((crop, max_top_k))
            return [("house mouse", 0.9), ("rat", 0.7)]

    monkeypatch.setitem(Inference.DETECTORS, "recording", RecordingDetector)
    monkeypatch.setitem(Inference.CLASSIFIERS, "recording", RecordingClassifier)
    inference = Inference(
        "recording",
        "recording",
        {"model": "detector-model", "conf": 0.1},
        {"model": "classifier-model", "max_top_k": 3, "device": "cpu"},
        classes=["mouse"],
        class_mappings={"house mouse": "mouse"},
    )

    result = inference(image, detector_max_top_k=2, classifier_max_top_k=1)

    assert result == [
        {
            "detection_confidence": 0.8,
            "classifications": [{"class": "mouse", "confidence": 0.9}],
        },
        {
            "detection_confidence": 0.6,
            "classifications": [{"class": "mouse", "confidence": 0.9}],
        },
    ]
    assert inference.classifier.calls == [(crops[0], 1), (crops[1], 1)]


def test_inference_returns_empty_list_without_detections(monkeypatch, image):
    class EmptyDetector:
        def __init__(self, model, *, kwargs):
            pass

        def detect(self, received_image, max_top_k):
            return []

    class UnusedClassifier:
        def __init__(self, model, max_top_k, *, kwargs):
            pass

        def classify(self, crop, max_top_k):
            pytest.fail("classifier must not run when detector returns no crops")

    monkeypatch.setitem(Inference.DETECTORS, "empty", EmptyDetector)
    monkeypatch.setitem(Inference.CLASSIFIERS, "unused", UnusedClassifier)
    inference = Inference(
        "empty", "unused", {"model": "d"}, {"model": "c"}
    )

    assert inference(image) == []
