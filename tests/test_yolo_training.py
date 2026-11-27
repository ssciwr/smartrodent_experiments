"""Behavioral tests for the standalone Ultralytics trainer (detect and classify)."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.utils.metrics import DetMetrics

from smartrodent import yolo_training as training
from smartrodent.training_validation import parse_settings
from smartrodent.yolo_training import (
    ClassifyTask,
    DetectTask,
    UltralyticsSettings,
    UltralyticsTrainer,
    ultralytics_task,
)

MLFLOW = {
    "tracking_uri": "sqlite:///mlflow.db",
    "train_experiment_name": "yolo-training",
    "tune_experiment_name": "yolo-tuning",
}


def settings_values(task="detect", **sections):
    """A valid raw settings mapping; keyword sections replace the defaults wholesale."""
    values = {
        "task": task,
        "model": "yolo26n.pt",
        "dataset": "dataset",
        "train_kwargs": {"project": "runs/train", "optimizer": "AdamW", "epochs": 3},
        "export_kwargs": {"format": "onnx"},
        "tune_kwargs": {"iterations": 2, "space": {"lr0": [1e-5, 1e-2]}},
    }
    values.update(sections)
    return copy.deepcopy(values)


class FakeYolo:
    """Record the Ultralytics calls we wrap and replay scripted validation epochs."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.callbacks = []
        self.scripted_trainers = []
        self.after_epoch = None
        self.train_kwargs = None
        self.tune_kwargs = None
        self.export_kwargs = None

    def add_callback(self, event, callback):
        """Record a callback registration."""
        self.callbacks.append((event, callback))

    def train(self, **kwargs):
        """Record arguments and fire ``on_fit_epoch_end`` once per scripted trainer."""
        self.train_kwargs = kwargs
        for scripted in self.scripted_trainers:
            for event, callback in self.callbacks:
                if event == "on_fit_epoch_end":
                    callback(scripted)
            if self.after_epoch is not None:
                self.after_epoch(scripted)

    def tune(self, **kwargs):
        """Record tuning arguments."""
        self.tune_kwargs = kwargs

    def export(self, **kwargs):
        """Record export arguments and return a recognizable path."""
        self.export_kwargs = kwargs
        return "model.onnx"


@pytest.fixture(autouse=True)
def mlflow_doubles(monkeypatch):
    """Isolate MLflow, Ultralytics' persistent settings and the MLflow environment."""
    client, ultralytics_settings = Mock(), Mock()
    monkeypatch.setattr(training, "mlflow", client)
    monkeypatch.setattr(training, "settings", ultralytics_settings)
    # delenv records nothing for an absent variable, so setenv first: it records the
    # prior state (including absence) and teardown restores it after configure().
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return client, ultralytics_settings


@pytest.fixture(scope="session")
def validator_dir(tmp_path_factory):
    """One throwaway directory so real validators never create run dirs elsewhere."""
    return tmp_path_factory.mktemp("validator")


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


@pytest.fixture
def detect_dataset(tmp_path):
    """A detection dataset root holding only ``data.yaml``."""
    root = tmp_path / "detect_ds"
    root.mkdir()
    (root / "data.yaml").write_text("names: [rodent]\n")
    return root


@pytest.fixture
def classify_dataset(tmp_path):
    """A classification dataset root holding a ``train`` directory."""
    root = tmp_path / "classify_ds"
    (root / "train").mkdir(parents=True)
    return root


def make_trainer(task, dataset, tmp_path, **sections):
    """Build a trainer over a real tmp dataset with outputs under ``tmp_path``."""
    sections.setdefault(
        "train_kwargs",
        {"project": str(tmp_path / "train"), "optimizer": "AdamW", "epochs": 3},
    )
    values = settings_values(task, dataset=str(dataset), **sections)
    return UltralyticsTrainer(parse_settings(UltralyticsSettings, values))


# --------------------------------------------------------------------------- settings


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_valid_settings_parse_for_both_tasks(task):
    """Test 1: a complete configuration of either task parses into typed settings."""
    parsed = parse_settings(UltralyticsSettings, settings_values(task, mlflow=MLFLOW))

    assert parsed.task == task
    assert parsed.train_project == Path("runs/train").absolute()
    assert parsed.mlflow.train_experiment_name == "yolo-training"


def test_settings_load_from_a_yaml_config(tmp_path, fake_yolo, detect_dataset):
    """Test 1: from_config reads the YAML and builds a trainer around the dataset."""
    config = tmp_path / "config.yaml"
    values = settings_values(dataset=str(detect_dataset))
    values["train_kwargs"]["project"] = str(tmp_path / "train")
    config.write_text(yaml.safe_dump(values))

    trainer = UltralyticsTrainer.from_config(config)

    assert trainer.export() == "model.onnx"


INVALID_CHANGES = {
    "unknown-top-key": lambda v: v.update(surprise=1),
    "no-task": lambda v: v.pop("task"),
    "no-model": lambda v: v.pop("model"),
    "no-dataset": lambda v: v.pop("dataset"),
    "no-project": lambda v: v["train_kwargs"].pop("project"),
    "no-optimizer": lambda v: v["train_kwargs"].pop("optimizer"),
    "no-format": lambda v: v["export_kwargs"].pop("format"),
    "bad-task": lambda v: v.update(task="segment"),
    "train-typo-key": lambda v: v["train_kwargs"].update(flipup=0.5),
    "train-bad-type": lambda v: v["train_kwargs"].update(epochs="many"),
    "tune-typo-key": lambda v: v["tune_kwargs"].update(flipup=0.5),
    "tune-bad-type": lambda v: v["tune_kwargs"].update(epochs="many"),
    "space-typo-key": lambda v: v["tune_kwargs"]["space"].update(flipup=[0.0, 0.5]),
    "empty-space": lambda v: v["tune_kwargs"].update(space={}),
    "no-space": lambda v: v["tune_kwargs"].pop("space"),
    "no-iterations": lambda v: v["tune_kwargs"].pop("iterations"),
    "null-bounds": lambda v: v["tune_kwargs"]["space"].update(lr0=None),
    "reversed-bounds": lambda v: v["tune_kwargs"]["space"].update(lr0=[0.1, 0.01]),
    "gain-too-high": lambda v: v["tune_kwargs"]["space"].update(lr0=[0.01, 0.1, 1.5]),
    "gain-negative": lambda v: v["tune_kwargs"]["space"].update(lr0=[0.01, 0.1, -0.1]),
    "bounds-one-element": lambda v: v["tune_kwargs"]["space"].update(lr0=[0.1]),
    "bounds-four-elements": lambda v: v["tune_kwargs"]["space"].update(
        lr0=[0.0, 0.1, 0.5, 0.5]
    ),
    "export-opset": lambda v: v["export_kwargs"].update(opset=12),
    "mlflow-missing": lambda v: v.update(
        mlflow={k: x for k, x in MLFLOW.items() if k != "tune_experiment_name"}
    ),
    "mlflow-unexpected": lambda v: v.update(
        mlflow={**MLFLOW, "registered_model_name": "m"}
    ),
    "mlflow-blank": lambda v: v.update(mlflow={**MLFLOW, "tracking_uri": " "}),
}


@pytest.mark.parametrize("change", INVALID_CHANGES.values(), ids=INVALID_CHANGES.keys())
def test_invalid_settings_are_rejected(change):
    """Test 2: every documented misconfiguration fails at parse time as ValueError."""
    values = settings_values()
    change(values)

    with pytest.raises(ValueError):
        parse_settings(UltralyticsSettings, values)


@pytest.mark.parametrize("section", ["train_kwargs", "tune_kwargs"])
@pytest.mark.parametrize("key", ["data", "model", "task"])
def test_reserved_keys_are_rejected_in_the_kwargs(section, key):
    """Test 2: data/model/task are top-level settings, not train or tune kwargs."""
    values = settings_values()
    values[section][key] = "detect"

    with pytest.raises(ValueError):
        parse_settings(UltralyticsSettings, values)


@given(
    epochs=st.integers(1, 500),
    patience=st.integers(1, 100),
    override_epochs=st.integers(501, 1000),
    iterations=st.integers(1, 50),
    imgsz=st.integers(1, 40).map(lambda n: n * 32),
)
@settings(max_examples=40, deadline=None)
def test_merged_tune_arguments_keep_train_values_and_overrides_win(
    epochs, patience, override_epochs, iterations, imgsz
):
    """Test 3: tune args hold every train key, overrides win, iterations/space added."""
    values = settings_values(
        train_kwargs={
            "project": "p",
            "optimizer": "AdamW",
            "epochs": epochs,
            "patience": patience,
        },
        tune_kwargs={
            "iterations": iterations,
            "epochs": override_epochs,
            "imgsz": imgsz,  # not a train key: still a valid override
            "space": {"lr0": [1e-5, 1e-2]},
        },
    )

    merged = parse_settings(UltralyticsSettings, values).tune_arguments

    assert merged == {
        "project": "p",
        "optimizer": "AdamW",
        "epochs": override_epochs,
        "patience": patience,
        "imgsz": imgsz,
        "iterations": iterations,
        "space": {"lr0": (1e-5, 1e-2)},
    }


@given(epochs=st.integers(1, 500))
@settings(max_examples=40, deadline=None)
def test_override_equal_to_train_value_is_rejected_but_different_is_accepted(epochs):
    """Test 4: an override that merely repeats the train value is rejected."""

    def values(override):
        return settings_values(
            train_kwargs={"project": "p", "optimizer": "AdamW", "epochs": epochs},
            tune_kwargs={
                "iterations": 1,
                "epochs": override,
                "space": {"lr0": [0.0, 1.0]},
            },
        )

    with pytest.raises(ValueError):
        parse_settings(UltralyticsSettings, values(epochs))
    parse_settings(UltralyticsSettings, values(epochs + 1))


finite = st.floats(0.0, 1.0, allow_nan=False)


@given(bounds=st.tuples(finite, finite).map(sorted), gain=st.none() | finite)
@settings(max_examples=40, deadline=None)
def test_list_bounds_become_equal_tuples(bounds, gain):
    """Test 5: YAML lists, with or without gain, become equal tuples."""
    raw = [*bounds] if gain is None else [*bounds, gain]
    values = settings_values(tune_kwargs={"iterations": 1, "space": {"lr0": raw}})

    space = parse_settings(UltralyticsSettings, values).tune_arguments["space"]

    assert space == {"lr0": tuple(raw)}


@given(low=finite, delta=st.floats(1e-6, 1.0, allow_nan=False))
@settings(max_examples=40, deadline=None)
def test_reversed_bounds_are_rejected(low, delta):
    """Test 5: low above high is rejected whatever the magnitudes."""
    values = settings_values(
        tune_kwargs={"iterations": 1, "space": {"lr0": [low + delta, low]}}
    )

    with pytest.raises(ValueError):
        parse_settings(UltralyticsSettings, values)


# ------------------------------------------------------------------ dataset and model


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_yolo_is_built_with_model_and_task_and_trains_on_the_task_data_argument(
    tmp_path, fake_yolo, request, task
):
    """Tests 7 and 9: detect passes <root>/data.yaml, classify passes <root>."""
    dataset = request.getfixturevalue(f"{task}_dataset")
    expected_data = dataset / "data.yaml" if task == "detect" else dataset

    make_trainer(task, dataset, tmp_path).train()

    assert str(fake_yolo[0].model_name) == "yolo26n.pt"
    assert fake_yolo[0].init_kwargs["task"] == task
    assert fake_yolo[0].train_kwargs["data"] == str(expected_data)


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_missing_dataset_content_fails_before_the_model_is_built(
    tmp_path, fake_yolo, task
):
    """Test 8: a missing data.yaml / train directory is a FileNotFoundError up front."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        make_trainer(task, empty, tmp_path)

    assert fake_yolo == []


def test_unknown_task_name_is_rejected():
    """Test 7: the task factory is explicit about names it does not know."""
    assert isinstance(ultralytics_task("detect"), DetectTask)
    assert isinstance(ultralytics_task("classify"), ClassifyTask)
    with pytest.raises(ValueError):
        ultralytics_task("segment")


# ------------------------------------------------------------------ metric builders


def detection_metrics() -> DetMetrics:
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


def as_ultralytics_results(results):
    """Mimic the trainer's metrics: 5-decimal floats, ``fitness`` removed."""
    return {k: round(float(v), 5) for k, v in results.items() if k != "fitness"}


def detect_trainer(save_dir, epoch=0, epochs=2, final=False):
    """A stand-in for the Ultralytics trainer as seen by ``on_fit_epoch_end``."""
    metrics = detection_metrics()
    return SimpleNamespace(
        epoch=epoch,
        epochs=epochs,
        save_dir=save_dir,
        metrics=as_ultralytics_results(metrics.results_dict),
        # The final best-model evaluation runs the validator outside training.
        validator=SimpleNamespace(
            metrics=metrics, names=metrics.names, training=not final
        ),
    )


def classify_trainer(
    validator_dir, y_true, y_pred, names, save_dir=None, epoch=0, epochs=2, final=False
):
    """A trainer whose validator is a real ``ClassificationValidator`` after updates."""
    validator = ClassificationValidator(save_dir=validator_dir, args={"plots": False})
    validator.init_metrics(SimpleNamespace(names=dict(enumerate(names))))
    logits = torch.eye(len(names))[torch.tensor(y_pred, dtype=torch.long)]
    validator.update_metrics(logits, {"cls": torch.tensor(y_true)})
    validator.training = not final
    return SimpleNamespace(
        epoch=epoch,
        epochs=epochs,
        save_dir=save_dir,
        metrics=as_ultralytics_results(validator.get_stats()),
        validator=validator,
    )


def rows_by_class(task, trainer, epoch=1):
    """Long-form epoch rows as a frame indexed by class name."""
    return pd.DataFrame(task.epoch_metrics(trainer).rows(epoch)).set_index("class_name")


# ------------------------------------------------------------------- train / tune


@pytest.mark.parametrize(
    ("epochs", "final_epoch"),
    [
        pytest.param(2, 2, id="full-run"),
        # After early stopping the final evaluation fires before ``epochs`` is reached.
        pytest.param(5, 2, id="early-stopped"),
    ],
)
def test_train_forwards_exactly_data_and_train_kwargs_and_records_epochs(
    tmp_path, fake_yolo, detect_dataset, epochs, final_epoch
):
    """Test 10: train args are untouched; per-epoch CSV grows; final eval adds no rows."""
    train_kwargs = {
        "project": str(tmp_path / "train"),
        "optimizer": "AdamW",
        "epochs": epochs,
    }
    trainer = make_trainer(
        "detect", detect_dataset, tmp_path, train_kwargs=train_kwargs
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "per_class_metrics.csv"
    model = fake_yolo[0]
    model.scripted_trainers = [
        detect_trainer(run_dir, epoch=0, epochs=epochs),
        detect_trainer(run_dir, epoch=1, epochs=epochs),
        detect_trainer(run_dir, epoch=final_epoch, epochs=epochs, final=True),
    ]
    csv_epochs = []
    model.after_epoch = lambda _: csv_epochs.append(
        sorted(pd.read_csv(csv_path)["epoch"].unique())
    )

    frame = trainer.train()

    assert model.train_kwargs == {
        "data": str(detect_dataset / "data.yaml"),
        **train_kwargs,
    }
    assert sorted(frame["epoch"].unique()) == [1, 2]
    assert len(frame) == 6
    assert csv_epochs == [[1], [1, 2], [1, 2]]
    assert len(pd.read_csv(csv_path)) == 6


def test_detect_per_class_metrics_come_from_ultralytics(tmp_path):
    """Test 11: per-class values from DetMetrics; unseen class NaN; macro_f1 0.5."""
    rows = rows_by_class(DetectTask(), detect_trainer(tmp_path))

    assert rows.loc["empty", "class_index"] == 0
    assert (
        rows.loc["empty", ["precision", "recall", "f1", "ap50", "ap50_95"]].isna().all()
    )
    assert rows.loc["rodent", "precision"] == pytest.approx(1.0, abs=1e-5)
    assert rows.loc["rodent", "f1"] == pytest.approx(1.0, abs=1e-5)
    assert rows.loc["rodent", "ap50"] == pytest.approx(0.995, abs=1e-5)
    assert rows.loc["rodent", "ap50_95"] == pytest.approx(0.745, abs=1e-5)
    # Precision of a missed class depends on an Ultralytics tie-break: not asserted.
    assert rows.loc["snake", ["recall", "f1", "ap50", "ap50_95"]].tolist() == [0.0] * 4
    assert rows["map50"].to_numpy() == pytest.approx(0.4975, abs=1e-5)
    assert rows["map50_95"].to_numpy() == pytest.approx(0.3725, abs=1e-5)
    assert rows["macro_f1"].to_numpy() == pytest.approx(0.5)
    assert (rows["epoch"] == 1).all()


def test_classify_known_example_with_undefined_classes(validator_dir):
    """Test 12: known counts; never-predicted -> precision NaN, F1 0; absent -> all NaN."""
    names = ["mouse", "rat", "snake", "empty"]
    trainer = classify_trainer(
        validator_dir, [0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 0, 0, 1], names
    )

    rows = rows_by_class(ClassifyTask(), trainer)

    assert rows.loc["mouse", "precision"] == pytest.approx(0.5)
    assert rows.loc["mouse", "recall"] == pytest.approx(2 / 3)
    assert rows.loc["rat", "precision"] == pytest.approx(1 / 3)
    assert pd.isna(rows.loc["snake", "precision"])
    assert rows.loc["snake", "recall"] == 0.0
    assert rows.loc["snake", "f1"] == 0.0
    assert rows.loc["empty", ["precision", "recall", "f1"]].isna().all()
    expected_macro = (4 / 7 + 2 / 5 + 0.0) / 3  # absent class is excluded
    assert rows["macro_f1"].to_numpy() == pytest.approx(expected_macro)
    assert rows["accuracy_top1"].to_numpy() == pytest.approx(3 / 7, abs=1e-5)
    assert rows["accuracy_top5"].to_numpy() == pytest.approx(1.0, abs=1e-5)


def test_classify_simple_precision_and_recall(validator_dir):
    """Test 12: the two-class example with precision 0.75 and recall 0.6 for mouse."""
    y_true = [0] * 5 + [1] * 4
    y_pred = [0, 0, 0, 1, 1, 0, 1, 1, 1]

    rows = rows_by_class(
        ClassifyTask(),
        classify_trainer(validator_dir, y_true, y_pred, ["mouse", "rat"]),
    )

    assert rows.loc["mouse", "precision"] == pytest.approx(0.75)
    assert rows.loc["mouse", "recall"] == pytest.approx(0.6)


@st.composite
def labelled_predictions(draw):
    """Random class count with matching true/predicted label lists."""
    nc = draw(st.integers(2, 5))
    n = draw(st.integers(1, 40))
    label = st.integers(0, nc - 1)
    return (
        nc,
        draw(st.lists(label, min_size=n, max_size=n)),
        draw(st.lists(label, min_size=n, max_size=n)),
    )


def confusion_matrix_oracle(nc, y_true, y_pred):
    """Per-class counts from an explicit confusion matrix, NaN where undefined."""
    matrix = np.zeros((nc, nc))
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    tp = np.diag(matrix)
    support, predictions = matrix.sum(1), matrix.sum(0)
    with np.errstate(all="ignore"):
        precision = np.where(predictions > 0, tp / predictions, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(
            support + predictions > 0, 2 * tp / (support + predictions), np.nan
        )
    return precision, recall, f1, support, predictions


@given(labelled_predictions())
@settings(max_examples=30, deadline=None)
def test_classify_metrics_match_a_confusion_matrix_oracle(validator_dir, case):
    """Test 13: precision/recall/F1/support/predictions agree with a hand-built matrix."""
    nc, y_true, y_pred = case
    names = [f"c{i}" for i in range(nc)]

    records = (
        ClassifyTask()
        .epoch_metrics(classify_trainer(validator_dir, y_true, y_pred, names))
        .rows(1)
    )

    frame = pd.DataFrame(records)
    precision, recall, f1, support, predictions = confusion_matrix_oracle(
        nc, y_true, y_pred
    )
    assert frame["class_name"].tolist() == names
    np.testing.assert_allclose(frame["precision"], precision, equal_nan=True)
    np.testing.assert_allclose(frame["recall"], recall, equal_nan=True)
    np.testing.assert_allclose(frame["f1"], f1, equal_nan=True)
    np.testing.assert_array_equal(frame["support"], support)
    np.testing.assert_array_equal(frame["predictions"], predictions)
    assert frame["support"].sum() == len(y_true)
    np.testing.assert_allclose(frame["macro_f1"], np.nanmean(f1))


def test_classify_recorder_writes_each_class_once_per_epoch(
    tmp_path, fake_yolo, classify_dataset, validator_dir
):
    """Test 13: through train(), every class appears exactly once per recorded epoch."""
    trainer = make_trainer("classify", classify_dataset, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake_yolo[0].scripted_trainers = [
        classify_trainer(
            validator_dir, [0, 1, 1], [0, 1, 0], ["a", "b"], run_dir, epoch=e, epochs=2
        )
        for e in (0, 1)
    ]

    frame = trainer.train()

    assert frame.groupby(["epoch", "class_name"]).size().eq(1).all()
    assert len(frame) == 4


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_tune_passes_merged_arguments_without_recorder(
    tmp_path, fake_yolo, request, task
):
    """Test 14: tune gets data + merged args with tuple bounds and returns the tune dir."""
    dataset = request.getfixturevalue(f"{task}_dataset")
    trainer = make_trainer(task, dataset, tmp_path)

    result = trainer.tune()

    assert result == tmp_path / "train"
    assert fake_yolo[0].tune_kwargs == {
        "data": str(dataset / "data.yaml") if task == "detect" else str(dataset),
        "project": str(tmp_path / "train"),
        "optimizer": "AdamW",
        "epochs": 3,
        "iterations": 2,
        "space": {"lr0": (1e-5, 1e-2)},
    }
    assert fake_yolo[0].callbacks == []


def test_tune_without_tune_kwargs_is_an_error(tmp_path, fake_yolo, detect_dataset):
    """Test 14: tuning an unconfigured run raises ValueError."""
    trainer = make_trainer("detect", detect_dataset, tmp_path, tune_kwargs=None)

    with pytest.raises(ValueError):
        trainer.tune()


@pytest.mark.parametrize("first", ["train", "tune"])
@pytest.mark.parametrize("second", ["train", "tune"])
def test_a_trainer_fits_only_once(tmp_path, fake_yolo, detect_dataset, first, second):
    """Test 15: any second train()/tune() on one instance raises RuntimeError."""
    trainer = make_trainer("detect", detect_dataset, tmp_path)
    getattr(trainer, first)()

    with pytest.raises(RuntimeError):
        getattr(trainer, second)()


def test_mlflow_train_and_tune_select_their_own_experiments(
    tmp_path, fake_yolo, detect_dataset, mlflow_doubles
):
    """Test 16: training and tuning (separate instances) use their own experiments."""
    client, ultralytics_settings = mlflow_doubles

    make_trainer("detect", detect_dataset, tmp_path, mlflow=MLFLOW).train()
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "yolo-training"
    make_trainer("detect", detect_dataset, tmp_path, mlflow=MLFLOW).tune()

    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "yolo-tuning"
    assert os.environ["MLFLOW_TRACKING_URI"] == "sqlite:///mlflow.db"
    assert client.set_experiment.call_args_list == [
        call("yolo-training"),
        call("yolo-tuning"),
    ]
    assert ultralytics_settings.update.call_args_list == [call({"mlflow": True})] * 2


@pytest.mark.parametrize("operation", ["train", "tune"])
def test_missing_mlflow_section_disables_ultralytics_mlflow(
    tmp_path, fake_yolo, detect_dataset, mlflow_doubles, operation
):
    """Test 16: without an mlflow section Ultralytics' mlflow setting is turned off."""
    client, ultralytics_settings = mlflow_doubles

    getattr(make_trainer("detect", detect_dataset, tmp_path), operation)()

    ultralytics_settings.update.assert_called_once_with({"mlflow": False})
    client.set_experiment.assert_not_called()


def test_export_forwards_export_kwargs_and_returns_the_path(
    tmp_path, fake_yolo, detect_dataset
):
    """Test 17: export passes export_kwargs verbatim and returns Ultralytics' result."""
    trainer = make_trainer("detect", detect_dataset, tmp_path)

    assert trainer.export() == "model.onnx"
    assert fake_yolo[0].export_kwargs == {"format": "onnx"}


# ------------------------------------------------------------- relative project paths


def test_relative_projects_reach_ultralytics_as_absolute_cwd_paths(
    tmp_path, monkeypatch, fake_yolo, detect_dataset
):
    """A relative project is anchored at the cwd, not at Ultralytics' runs_dir."""
    monkeypatch.chdir(tmp_path)
    train_kwargs = {"project": "runs/train", "optimizer": "AdamW", "epochs": 3}
    tune_kwargs = {
        "iterations": 1,
        "project": "runs/tune",
        "space": {"lr0": [1e-5, 1e-2]},
    }
    expected_train = str(Path.cwd() / "runs" / "train")
    expected_tune = str(Path.cwd() / "runs" / "tune")
    kwargs = {"train_kwargs": train_kwargs, "tune_kwargs": tune_kwargs}

    make_trainer("detect", detect_dataset, tmp_path, **kwargs).train()
    make_trainer("detect", detect_dataset, tmp_path, **kwargs).tune()

    assert fake_yolo[0].train_kwargs["project"] == expected_train
    assert fake_yolo[1].tune_kwargs["project"] == expected_tune


def test_trainer_projects_are_the_absolute_train_project_with_tune_fallback(
    tmp_path, monkeypatch, fake_yolo, detect_dataset
):
    """train_project / tune_project: cwd-anchored, tune falls back to train."""
    monkeypatch.chdir(tmp_path)
    train_kwargs = {"project": "runs/train", "optimizer": "AdamW"}
    fallback = make_trainer(
        "detect", detect_dataset, tmp_path, train_kwargs=train_kwargs
    )
    overridden = make_trainer(
        "detect",
        detect_dataset,
        tmp_path,
        train_kwargs=train_kwargs,
        tune_kwargs={"iterations": 1, "project": "runs/tune", "space": {"lr0": [0, 1]}},
    )

    assert fallback.train_project == Path.cwd() / "runs" / "train"
    assert fallback.tune_project == Path.cwd() / "runs" / "train"
    assert overridden.tune_project == Path.cwd() / "runs" / "tune"
