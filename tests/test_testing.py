"""Behavioral tests for configurable YOLO classifier testing."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import pytest
import torch
import yaml

import smartrodent.testing as testing

CLASS_NAMES = {0: "mouse", 1: "shrew", 2: "vole"}


def metric_configs(num_classes: int = 3) -> dict:
    """Return a small metric config covering scalar, per-class and matrix outputs."""
    return {
        "accuracy": {
            "class": "torchmetrics.classification.MulticlassAccuracy",
            "kwargs": {"num_classes": num_classes, "average": "micro"},
        },
        "macro_f1": {
            "class": "torchmetrics.classification.MulticlassF1Score",
            "kwargs": {"num_classes": num_classes, "average": "macro"},
        },
        "per_class_f1": {
            "class": "torchmetrics.classification.MulticlassF1Score",
            "kwargs": {"num_classes": num_classes, "average": "none"},
        },
        "confusion_matrix": {
            "class": "torchmetrics.classification.MulticlassConfusionMatrix",
            "kwargs": {"num_classes": num_classes},
        },
    }


class FakeYolo:
    """Classifier double: predicts a fixed class per image file name."""

    def __init__(self, model_path, predictions, task="classify"):
        self.model_path = model_path
        self.predictions = predictions
        self.task = task
        self.names = dict(CLASS_NAMES)
        self.predict_calls = []

    def predict(self, source, stream, **kwargs):
        """Record the call and yield one result with class probabilities per image."""
        self.predict_calls.append({"source": source, "stream": stream, **kwargs})
        for path in source:
            probabilities = torch.full((len(self.names),), 0.1)
            probabilities[self.predictions[Path(path).name]] = 0.8
            yield SimpleNamespace(probs=SimpleNamespace(data=probabilities))


def make_dataset(root: Path, samples: list[tuple[int, int]]) -> tuple[Path, dict]:
    """Create class folders with one image per (label, prediction) sample.

    Returns:
        The split directory and the image-name -> predicted-class mapping.
    """
    data_path = root / "test"
    predictions = {}
    for class_name in CLASS_NAMES.values():
        (data_path / class_name).mkdir(parents=True)
    for index, (label, prediction) in enumerate(samples):
        name = f"image_{index}.jpg"
        (data_path / CLASS_NAMES[label] / name).write_bytes(b"")
        predictions[name] = prediction
    return data_path, predictions


def install_fake_yolo(monkeypatch, predictions, task="classify") -> list[FakeYolo]:
    """Replace the Ultralytics model constructor and collect created doubles."""
    models = []

    def factory(model_path):
        model = FakeYolo(model_path, predictions, task)
        models.append(model)
        return model

    monkeypatch.setattr(testing, "YOLO", factory)
    return models


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A model file, a 4-image split (3 correct) and a fake classifier."""
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"")
    samples = [(0, 0), (0, 1), (1, 1), (2, 2)]
    data_path, predictions = make_dataset(tmp_path, samples)
    models = install_fake_yolo(monkeypatch, predictions)
    return SimpleNamespace(
        model_path=model_path,
        data_path=data_path,
        output_path=tmp_path / "results" / "nested",
        models=models,
    )


def make_tester(setup, **overrides) -> testing.YoloTester:
    """Construct a tester from the fixture with optional argument overrides."""
    arguments = {
        "model_path": setup.model_path,
        "data_path": setup.data_path,
        "output_path": setup.output_path,
        "metrics": metric_configs(),
    } | overrides
    return testing.YoloTester(**arguments)


def test_missing_model_file_raises(setup, tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file"):
        make_tester(setup, model_path=tmp_path / "missing.pt")


def test_missing_data_directory_raises(setup, tmp_path):
    with pytest.raises(FileNotFoundError, match="Data directory"):
        make_tester(setup, data_path=tmp_path / "missing")


def test_output_directory_is_created(setup):
    make_tester(setup)
    assert setup.output_path.is_dir()


def test_non_classifier_model_is_rejected(setup, monkeypatch):
    install_fake_yolo(monkeypatch, {}, task="detect")
    with pytest.raises(ValueError, match="Only classification"):
        make_tester(setup)


def test_class_directory_unknown_to_model_is_rejected(setup):
    (setup.data_path / "hedgehog").mkdir()
    with pytest.raises(ValueError, match="hedgehog"):
        make_tester(setup)


def test_split_without_images_is_rejected(setup):
    for image in setup.data_path.rglob("*.jpg"):
        image.unlink()
    (setup.data_path / "mouse" / "notes.txt").write_text("not an image")
    with pytest.raises(ValueError, match="No images"):
        make_tester(setup)


def test_non_image_files_are_ignored(setup):
    (setup.data_path / "mouse" / "records.csv").write_text("a,b\n")
    tester = make_tester(setup)
    assert len(tester.image_paths) == 4


def test_labels_follow_model_names(setup):
    tester = make_tester(setup)
    labels = {
        path.name: label for path, label in zip(tester.image_paths, tester.labels)
    }
    assert labels == {
        "image_0.jpg": 0,
        "image_1.jpg": 0,
        "image_2.jpg": 1,
        "image_3.jpg": 2,
    }


@pytest.mark.parametrize(
    ("metrics", "error", "message"),
    [
        ({}, ValueError, "At least one metric"),
        ({"acc": {"kwargs": {}}}, ValueError, "missing its 'class'"),
        ({"acc": {"class": "torch.nn.Linear"}}, TypeError, "not a torchmetrics"),
        ({"acc": {"class": "torch.no_grad"}}, TypeError, "not a torchmetrics"),
        ({"acc": {"class": "json.dumps"}}, TypeError, "not a torchmetrics"),
        (metric_configs(num_classes=5), ValueError, "num_classes=5"),
    ],
)
def test_invalid_metric_configs_are_rejected(setup, metrics, error, message):
    with pytest.raises(error, match=message):
        make_tester(setup, metrics=metrics)


def test_metric_without_kwargs_is_built():
    collection = testing.build_metric_collection(
        {"mean": {"class": "torchmetrics.MeanMetric"}}, num_classes=3
    )
    assert list(collection.keys()) == ["mean"]


def test_test_writes_and_returns_configured_metrics(setup):
    results = make_tester(setup).test()

    assert results == json.loads((setup.output_path / "metrics.json").read_text())
    assert results["accuracy"] == pytest.approx(0.75)
    assert results["per_class_f1"] == pytest.approx([2 / 3, 2 / 3, 1.0])
    assert results["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_predict_kwargs_are_forwarded_with_streaming(setup):
    make_tester(setup, predict_kwargs={"imgsz": 64, "batch": 2}).test()
    call = setup.models[0].predict_calls[0]
    assert call["stream"] is True
    assert call["imgsz"] == 64 and call["batch"] == 2
    assert len(call["source"]) == 4


def test_repeated_test_runs_do_not_accumulate(setup):
    tester = make_tester(setup)
    assert tester.test() == tester.test()


def test_missing_prediction_results_fail_loudly(setup):
    tester = make_tester(setup)
    tester.model.predict = lambda source, stream, **kwargs: iter([])
    with pytest.raises(ValueError):
        tester.test()


def test_from_config_matches_direct_construction(setup, tmp_path):
    config = {
        "model_path": str(setup.model_path),
        "data_path": str(setup.data_path),
        "output_path": str(setup.output_path),
        "predict_kwargs": {"imgsz": 64},
        "metrics": metric_configs(),
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    from_config = testing.YoloTester.from_config(config_path)
    direct = make_tester(setup, predict_kwargs={"imgsz": 64})

    assert from_config.model_path == direct.model_path
    assert from_config.data_path == direct.data_path
    assert from_config.output_path == direct.output_path
    assert from_config.predict_kwargs == direct.predict_kwargs
    assert from_config.image_paths == direct.image_paths
    assert from_config.test() == direct.test()


# property test
samples_strategy = st.lists(
    st.tuples(st.integers(0, 2), st.integers(0, 2)), min_size=1, max_size=30
)


@settings(max_examples=40, deadline=None)
@given(samples=samples_strategy)
def test_metrics_match_independent_computation(samples):
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as mp:
        root = Path(directory)
        model_path = root / "best.pt"
        model_path.write_bytes(b"")
        data_path, predictions = make_dataset(root, samples)
        install_fake_yolo(mp, predictions)

        results = testing.YoloTester(
            model_path, data_path, root / "out", metric_configs()
        ).test()

    correct = sum(label == prediction for label, prediction in samples)
    assert results["accuracy"] == pytest.approx(correct / len(samples))

    matrix = results["confusion_matrix"]
    assert sum(map(sum, matrix)) == len(samples)
    for label, prediction in set(samples):
        assert matrix[label][prediction] == samples.count((label, prediction))

    # Reference macro F1: torchmetrics averages over classes that occur in either
    # targets or predictions, so classes absent from both are excluded here too.
    f1_scores = []
    for cls in range(3):
        tp = sum(label == cls and prediction == cls for label, prediction in samples)
        fp = sum(label != cls and prediction == cls for label, prediction in samples)
        fn = sum(label == cls and prediction != cls for label, prediction in samples)
        if tp + fp + fn > 0:
            f1_scores.append(2 * tp / (2 * tp + fp + fn))
    assert results["macro_f1"] == pytest.approx(sum(f1_scores) / len(f1_scores))
