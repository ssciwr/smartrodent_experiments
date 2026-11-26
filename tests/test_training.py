"""Behavioral tests for the YOLO training helpers."""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smartrodent import training


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
    extra_callback = lambda trainer: trainer

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


def test_detector_collects_per_class_metrics(fake_yolo):
    trainer = training.YoloDetectionTrainer("dataset")
    box = SimpleNamespace(
        ap_class_index=[1],
        p=[0.75],
        r=[0.5],
        f1=[0.6],
        ap50=[0.8],
        ap=[0.7],
    )
    callback_trainer = SimpleNamespace(
        epoch=0,
        epochs=2,
        metrics={"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.7},
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=box, names={0: "empty", 1: "rodent"})
        ),
    )

    trainer.collect_metrics(callback_trainer)

    assert trainer.history[1]["classes"]["empty"]["f1"] is None
    assert trainer.history[1]["classes"]["rodent"]["f1"] == pytest.approx(0.6)
    assert trainer.history[1]["macro_f1"] == pytest.approx(0.6)


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
