"""Behavioral tests for the YOLO training helpers."""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ultralytics.utils.metrics import DetMetrics

from smartrodent import yolo_training as training


class FakeYolo:
    """Record calls made through the small Ultralytics API surface we wrap."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.callbacks = []
        self.train_kwargs = None
        self.tune_kwargs = None
        self.export_kwargs = None

    def add_callback(self, event, callback):
        """Record a callback registration."""
        self.callbacks.append((event, callback))

    def train(self, **kwargs):
        """Record training arguments."""
        self.train_kwargs = kwargs

    def tune(self, **kwargs):
        """Record tuning arguments and return a recognizable result."""
        self.tune_kwargs = kwargs
        return "tuned"

    def export(self, **kwargs):
        """Record export arguments and return a recognizable path."""
        self.export_kwargs = kwargs
        return "model.onnx"


@pytest.fixture
def fake_yolo(monkeypatch):
    """Replace Ultralytics models with deterministic test doubles."""
    models = []

    def factory(model_name, **kwargs):
        model = FakeYolo(model_name, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(training, "YOLO", factory)
    return models


def test_detector_trains_with_resolved_yaml_and_callbacks(tmp_path, fake_yolo):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("names: [rodent]\n")

    def extra_callback(trainer):
        return trainer

    trainer = training.YoloDetectionTrainer(
        dataset,
        model_name="yolo26n.pt",
        epochs=3,
        event_callback=[("on_train_start", extra_callback)],
    )

    assert trainer.train() == {}
    assert fake_yolo[0].train_kwargs == {
        "trainer": None,
        "data": str(data_yaml),
        "epochs": 3,
    }
    assert fake_yolo[0].callbacks == [
        ("on_fit_epoch_end", trainer.collect_metrics),
        ("on_train_start", extra_callback),
    ]


def test_detector_reports_missing_dataset_yaml(tmp_path, fake_yolo):
    with pytest.raises(FileNotFoundError, match="No data.yaml or config.yaml"):
        training.YoloDetectionTrainer(tmp_path).data_yaml()


@given(space=st.dictionaries(st.text(min_size=1), st.lists(st.floats(), max_size=4)))
def test_tune_space_converts_ranges_to_tuples(space):
    with patch.object(training, "YOLO", FakeYolo):
        trainer = training.YoloDetectionTrainer("dataset", tune_kwargs={"space": space})

    assert trainer.tune_space() == {key: tuple(value) for key, value in space.items()}


def test_tune_specific_arguments_override_training_defaults(tmp_path, fake_yolo):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [rodent]\n")
    trainer = training.YoloDetectionTrainer(
        data_yaml,
        epochs=100,
        tune_kwargs={"epochs": 5, "space": {"lr0": [0.001, 0.01]}},
    )

    assert trainer.tune() == "tuned"
    assert fake_yolo[0].tune_kwargs == {
        "data": str(data_yaml),
        "epochs": 5,
        "space": {"lr0": (0.001, 0.01)},
    }


def test_mlflow_uses_configured_uri_and_operation_experiments(
    tmp_path, fake_yolo, monkeypatch
):
    """Training and tuning select their separately configured MLflow experiments."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [rodent]\n")
    mlflow_client = Mock()
    ultralytics_settings = Mock()
    environment = {}
    monkeypatch.setattr(training, "mlflow", mlflow_client)
    monkeypatch.setattr(training, "settings", ultralytics_settings)
    monkeypatch.setattr(training.os, "environ", environment)
    trainer = training.YoloDetectionTrainer(
        data_yaml,
        tune_kwargs={"epochs": 5},
        mlflow_config={
            "tracking_uri": "sqlite:///mlflow.db",
            "train_experiment_name": "detector-training",
            "tune_experiment_name": "detector-tuning",
        },
    )

    trainer.train()

    assert environment == {
        "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
        "MLFLOW_EXPERIMENT_NAME": "detector-training",
    }
    mlflow_client.set_tracking_uri.assert_called_once_with("sqlite:///mlflow.db")
    mlflow_client.set_experiment.assert_called_once_with("detector-training")
    ultralytics_settings.update.assert_called_once_with({"mlflow": True})

    assert trainer.tune() == "tuned"

    assert environment["MLFLOW_EXPERIMENT_NAME"] == "detector-tuning"
    assert mlflow_client.set_tracking_uri.call_args_list == [
        call("sqlite:///mlflow.db"),
        call("sqlite:///mlflow.db"),
    ]
    assert mlflow_client.set_experiment.call_args_list == [
        call("detector-training"),
        call("detector-tuning"),
    ]
    assert ultralytics_settings.update.call_count == 2


def _validated_detection_metrics() -> DetMetrics:
    """Real Ultralytics metrics: rodent found perfectly, snake missed, empty unseen."""
    metrics = DetMetrics(names={0: "empty", 1: "rodent", 2: "snake"})
    for image, true_positives, confidences, classes in [
        ("a.jpg", [[True] * 10, [True] * 5 + [False] * 5], [0.9, 0.8], [1, 1]),
        ("b.jpg", [[False] * 10], [0.3], [2]),
    ]:
        metrics.update_stats(
            {
                "tp": np.array(true_positives),
                "conf": np.array(confidences),
                "pred_cls": np.array(classes),
                "target_cls": np.array(classes),
                "target_img": np.unique(classes),
                "im_name": image,
            }
        )
    metrics.process(plot=False)
    return metrics


def test_detector_collects_per_class_metrics_from_ultralytics(fake_yolo):
    """Per-class values come from Ultralytics; unvalidated classes are explicit None."""
    trainer = training.YoloDetectionTrainer("dataset")
    metrics = _validated_detection_metrics()
    callback_trainer = SimpleNamespace(
        epoch=0,
        epochs=2,
        metrics=metrics.results_dict,
        validator=SimpleNamespace(metrics=metrics),
    )

    trainer.collect_metrics(callback_trainer)
    # Ultralytics' final best-model evaluation reuses the event after the last epoch.
    trainer.collect_metrics(SimpleNamespace(**{**vars(callback_trainer), "epoch": 2}))

    assert list(trainer.history) == [1]
    history = trainer.history[1]
    assert history["classes"] == {
        "empty": {
            "class_index": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "ap50": None,
            "ap50_95": None,
        },
        "rodent": {
            "class_index": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": pytest.approx(1.0),
            "ap50": pytest.approx(0.995),
            "ap50_95": pytest.approx(0.745),
        },
        "snake": {
            "class_index": 2,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ap50": 0.0,
            "ap50_95": 0.0,
        },
    }
    assert history["map50"] == history["iou_ap50"] == pytest.approx(0.4975)
    assert history["map50_95"] == history["iou_map50_95"] == pytest.approx(0.3725)
    assert history["macro_f1"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "change",
    [
        lambda config: config.pop("tune_experiment_name"),
        lambda config: config.update(registered_model_name="detector"),
        lambda config: config.update(tracking_uri=" "),
    ],
    ids=["missing", "unexpected", "blank"],
)
def test_detector_rejects_invalid_mlflow_configuration(fake_yolo, change):
    mlflow_config = {
        "tracking_uri": "sqlite:///mlflow.db",
        "train_experiment_name": "detector-training",
        "tune_experiment_name": "detector-tuning",
    }
    change(mlflow_config)

    with pytest.raises(ValueError):
        training.YoloDetectionTrainer("dataset", mlflow_config=mlflow_config)


def test_detector_loads_config_exports_and_formats_history(tmp_path, fake_yolo):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "config.yaml").write_text("names: [rodent]\n")
    config = tmp_path / "trainer.yaml"
    config.write_text(
        f"""train_dataset: {dataset}
model_name: custom.pt
return_format: dataframe
mlflow:
  tracking_uri: sqlite:///test.db
  train_experiment_name: detector-train
  tune_experiment_name: detector-tune
train_kwargs:
  epochs: 2
export_kwargs:
  format: onnx
"""
    )

    trainer = training.YoloDetectionTrainer.from_config(config)
    assert trainer.mlflow_config == training.MlflowTrackingConfiguration(
        tracking_uri="sqlite:///test.db",
        train_experiment_name="detector-train",
        tune_experiment_name="detector-tune",
    )
    trainer.history = {
        1: {
            "map50": 0.8,
            "classes": {
                "rodent": {"class_index": 0, "precision": 0.75},
            },
        }
    }

    dataframe = trainer.dataframe()
    assert dataframe.to_dict("records") == [
        {
            "epoch": 1,
            "class_name": "rodent",
            "map50": 0.8,
            "class_index": 0,
            "precision": 0.75,
        }
    ]
    assert trainer.export() == "model.onnx"
    assert fake_yolo[0].export_kwargs == {"format": "onnx"}


def test_detector_rejects_unknown_return_format(tmp_path, fake_yolo):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [rodent]\n")
    trainer = training.YoloDetectionTrainer(data_yaml)

    with pytest.raises(ValueError, match="return_format"):
        trainer.train(return_format="records")
