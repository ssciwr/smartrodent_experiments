"""Native Ultralytics head-training contracts on temporary YOLO models and data."""

import hashlib
import os
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.pytorch
import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from training_test_support import (
    assert_registered_classifier,
    download_artifact,
    native_checkpoint,
    wide_native_encoder,
)
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.nn.tasks import ClassificationModel

from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
)
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration


@pytest.fixture
def native_runs(monkeypatch):
    """Observe actual native optimization, not a substitute trainer or epoch loop."""
    observed = []
    original = ClassificationTrainer.train

    def train(self, *args, **kwargs):
        """Capture live state before/after native epochs without touching model behavior."""
        record = SimpleNamespace(trainer=self, before=None, after=None, modes=[])
        observed.append(record)

        def start(trainer):
            record.before = deepcopy(trainer.model.state_dict())

        def batch(trainer):
            record.modes.append(
                {
                    name: module.training
                    for name, module in trainer.model.named_modules()
                }
            )

        def epoch(trainer):
            record.after = deepcopy(trainer.model.state_dict())

        self.add_callback("on_train_start", start)
        self.add_callback("on_train_batch_start", batch)
        self.add_callback("on_train_epoch_end", epoch)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationTrainer, "train", train)
    return observed


def _head_prefix(model, boundary):
    """Identify the selected head in the native model's state namespace."""
    prefix = f"model.{len(model.model) - 1}."
    return prefix + "linear." if boundary == "linear" else prefix


def _signature(loader):
    """Describe image identities from the native classification batch contract."""
    result = []
    for batch in loader:
        images, labels = batch["img"], batch["cls"]
        assert images.ndim == 4 and len(images) == len(labels)
        result.extend(
            (int(label), round(float(image[0, 0, 0]) * 255))
            for label, image in zip(labels, images, strict=True)
        )
    return result


def test_training_loader_is_single_view_shuffled_and_covers_source_once(
    classifier_configuration, native_runs
):
    """Native training consumes configured grayscale/resize; validation stays unaugmented."""
    for path in classifier_configuration.train_dataset.rglob("*.png"):
        with Image.open(path) as image:
            color = image.getpixel((0, 0))
        Image.new("RGB", (51, 19), color).save(path)
    config = replace(
        classifier_configuration,
        epochs=1,
        transforms={
            **classifier_configuration.transforms,
            "image_size": 64,
            "random_grayscale": {"p": 1.0},
        },
    )
    ClassifierExperiment(config).run()
    assert len(native_runs) == 1
    trainer = native_runs[0].trainer
    loader = trainer.train_loader
    first, second = _signature(loader), _signature(loader)
    assert len(first) == len(second) == len(set(first)) == 10
    assert Counter(first) == Counter(second)
    assert Counter(label for label, _ in first) == {0: 5, 1: 5}
    assert first != second
    images = next(iter(loader))["img"]
    assert images.shape[1:] == (3, 64, 64)
    torch.testing.assert_close(images[:, 0], images[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(images[:, 0], images[:, 2], rtol=0, atol=0)
    validation = next(iter(trainer.test_loader))["img"]
    assert validation.shape[1:] == (3, 64, 64)
    assert not torch.equal(validation[:, 0], validation[:, 1])


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_validation_loader_is_complete_and_repeatable(
    classifier_configuration, native_runs, batch_size
):
    """Native validation sees every requested image with the existing preprocessing."""
    config = replace(classifier_configuration, epochs=1, batch_size=batch_size)
    ClassifierExperiment(config).run()
    first = _signature(native_runs[0].trainer.test_loader)
    assert first == _signature(native_runs[0].trainer.test_loader)
    assert len(first) == len(set(first)) == 5
    assert Counter(label for label, _ in first) == {0: 3, 1: 2}
    assert {red for _, red in first} == {30, 50, 70}


def test_native_validation_computes_real_scores_and_unequal_tail_loss(
    classifier_configuration, classifier_assets, tmp_path, monkeypatch
):
    """Preserve the configured full-split CE/scikit metrics inside native validation."""
    from sklearn.metrics import balanced_accuracy_score

    yolo = YOLO(str(classifier_assets.ultralytics), task="classify")
    with torch.no_grad():
        yolo.model.model[-1].linear.weight.zero_()
        yolo.model.model[-1].linear.bias.copy_(torch.tensor([-2.0, 3.0]))
    donor = tmp_path / "known-scores.pt"
    yolo.save(donor)
    config = replace(
        classifier_configuration,
        epochs=1,
        batch_size=2,
        train_kwargs={**classifier_configuration.train_kwargs, "lr0": 1e-6},
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        initialization={
            **classifier_configuration.initialization,
            "head": {
                "replace": "linear",
                "source": "yolo_checkpoint",
                "checkpoint": str(donor),
            },
        },
    )
    expected = []
    original = ClassificationValidator.__call__

    def validate(self, trainer=None, model=None):
        if trainer is not None:
            network = trainer.ema.ema if trainer.ema else trainer.model
            network.eval()
            losses, targets, predictions = [], [], []
            with torch.inference_mode():
                for batch in trainer.test_loader:
                    labels = batch["cls"].long()
                    logits = network(batch["img"])[1]
                    losses.append(torch.nn.functional.cross_entropy(logits, labels))
                    targets.append(labels)
                    predictions.append(logits.argmax(1))
                labels = torch.cat(targets)
                guessed = torch.cat(predictions)
                # Native val/loss averages batch means; ce_loss_mean remains full-split CE.
                native_mean = torch.stack(losses).mean().item()
                weighted_mean = sum(
                    loss.item() * len(target)
                    for loss, target in zip(losses, targets, strict=True)
                ) / len(labels)
            assert native_mean != pytest.approx(weighted_mean)
            score = balanced_accuracy_score(labels.numpy(), guessed.numpy())
            assert score == 0.5 and (
                labels == guessed
            ).float().mean().item() == pytest.approx(0.4)
            expected.append((weighted_mean, score))
        return original(self, trainer=trainer, model=model)

    monkeypatch.setattr(ClassificationValidator, "__call__", validate)
    result = ClassifierExperiment(config).run_with_result()
    assert len(expected) == 1
    loss, score = expected[0]
    assert result.best_value == pytest.approx(loss, abs=0.00005)
    card = yaml.safe_load(
        result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
    )
    assert card["validation"]["metrics"]["ce_loss_mean"] == pytest.approx(
        loss, abs=0.00005
    )
    assert card["validation"]["metrics"]["balanced_accuracy"] == score
    assert config.tracked_metric_definition == {
        "function": "torch.nn.functional.cross_entropy",
        "reduction": "mean",
    }


# Expected frozen name prefixes for the fixture network (0=Conv, 1=Dropout, 2=Classify).
# Index 2 is the head: freezing it is only valid for a linear probe and freezes the
# head's convolution, so Linear stays trainable in every configuration.
FREEZE_CASES = [
    ("linear", 3, ("model.0.", "model.1.", "model.2.conv.")),
    ("linear", [2], ("model.2.conv.",)),
    ("linear", 0, ()),
    ("linear", [], ()),
    ("full", 2, ("model.0.", "model.1.")),
    ("full", [1], ("model.1.",)),
    ("full", 0, ()),
]


# encoder_batchnorm only has an effect on frozen layers, so it is required when
# something is frozen and must be omitted (None) when nothing is.
FREEZE_RUNS = [
    (*case, batchnorm)
    for case in FREEZE_CASES
    for batchnorm in (("eval", "adapt") if case[2] else (None,))
]


def _mode_dependent_modules(model):
    """Yield modules whose behavior depends on train/eval mode.

    Stateless modules such as Ultralytics' shared default SiLU are excluded: one
    instance may belong to frozen and trainable layers but is listed only once.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            yield name, module


def _is_frozen(name, frozen_prefixes):
    """Match a module/parameter name to layer prefixes without confusing layer 1 with 10."""
    return any(f"{name}.".startswith(prefix) for prefix in frozen_prefixes)


@pytest.mark.parametrize("head_replace,freeze,frozen,batchnorm", FREEZE_RUNS)
def test_configured_freeze_controls_native_gradients_modes_and_updates(
    classifier_configuration,
    native_runs,
    head_replace,
    freeze,
    frozen,
    batchnorm,
):
    """A freeze count/index list fixes exactly those layers; everything else trains."""
    config = replace(
        classifier_configuration,
        epochs=2,
        freeze=freeze,
        encoder_batchnorm=batchnorm,
        initialization={
            **classifier_configuration.initialization,
            "head": {"replace": head_replace, "source": "random"},
        },
    )
    result = ClassifierExperiment(config).run_with_result()
    assert isinstance(result, ExperimentResult) and len(native_runs) == 1
    record = native_runs[0]
    model = record.trainer.model
    assert isinstance(model, ClassificationModel)
    # Native setup receives the SupCon encoder as-is: no reload, reset, or resume.
    boundary = _head_prefix(model, head_replace)
    source = YoloSupConModel.from_checkpoint(
        config.initialization["encoder"]["checkpoint"]
    )
    try:
        for name, tensor in source.full_model.model.state_dict().items():
            if not name.startswith(boundary):
                assert torch.equal(record.before[name], tensor), name
    finally:
        source.finalize()
    parameters = dict(model.named_parameters())
    assert all(
        parameter.requires_grad == (not _is_frozen(name, frozen))
        for name, parameter in parameters.items()
    )
    for name in parameters:
        if _is_frozen(name, frozen):
            assert torch.equal(record.before[name], record.after[name]), name
        elif name.endswith(".weight"):
            assert not torch.equal(record.before[name], record.after[name]), name
    # Frozen layers are fully in eval mode (including Dropout); only their
    # BatchNorm statistics may adapt. Every other Dropout/BatchNorm trains natively.
    assert record.modes
    for modes in record.modes:
        for name, module in _mode_dependent_modules(model):
            if _is_frozen(name, frozen):
                expected = (
                    isinstance(module, torch.nn.BatchNorm2d) and batchnorm == "adapt"
                )
            else:
                expected = True
            assert modes[name] == expected, name
    for name in record.before:
        if _is_frozen(name, frozen) and name.endswith(("running_mean", "running_var")):
            assert torch.equal(record.before[name], record.after[name]) == (
                batchnorm == "eval"
            ), name
        elif _is_frozen(name, frozen) and batchnorm == "eval":
            assert torch.equal(record.before[name], record.after[name]), name
    optimizer = record.trainer.optimizer
    assert isinstance(optimizer, torch.optim.AdamW) and optimizer.state
    assert all(parameter.requires_grad for parameter in optimizer.state)
    assert all(int(state["step"]) < 99 for state in optimizer.state.values())
    assert result.last_checkpoint == config.output_directory / "weights" / "last.pt"
    assert result.best_checkpoint == config.output_directory / "weights" / "best.pt"
    assert {
        path.name for path in (config.output_directory / "weights").glob("*.pt")
    } == {"best.pt", "last.pt"}
    assert (config.output_directory / "results.csv").is_file()
    restored = YOLO(str(result.best_checkpoint), task="classify").model
    assert isinstance(restored, ClassificationModel) and restored.names == {
        0: "mouse",
        1: "rat",
    }


@pytest.mark.parametrize(
    "head_replace,freeze",
    [
        ("full", 3),
        ("full", [2]),
        ("full", [0, 5]),
        ("linear", 4),
        ("linear", [3]),
    ],
)
def test_freeze_beyond_the_trainable_boundary_fails_before_native_training(
    classifier_configuration, native_runs, head_replace, freeze
):
    """Full replacement trains the entire head and Linear is never frozen."""
    config = replace(
        classifier_configuration,
        freeze=freeze,
        initialization={
            **classifier_configuration.initialization,
            "head": {"replace": head_replace, "source": "random"},
        },
    )
    with pytest.raises(ValueError, match="freeze"):
        ClassifierExperiment(config).run()
    assert not native_runs
    assert not config.output_directory.exists()


# Duplicate layer indices are our rule; value types and ranges are pydantic's.
@pytest.mark.parametrize("freeze", [[0, 0]])
def test_invalid_freeze_values_fail_at_configuration(classifier_configuration, freeze):
    """Only a nonnegative layer count or a list of unique layer indices is accepted."""
    with pytest.raises((ValueError, TypeError), match="freeze"):
        replace(classifier_configuration, freeze=freeze)
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("freeze", [0, []])
@pytest.mark.parametrize("batchnorm", ["eval", "adapt"])
def test_encoder_batchnorm_without_frozen_layers_is_rejected(
    classifier_configuration, freeze, batchnorm
):
    """With nothing frozen, every BatchNorm trains, so the setting would be silently ignored."""
    with pytest.raises(ValueError, match="encoder_batchnorm"):
        replace(classifier_configuration, freeze=freeze, encoder_batchnorm=batchnorm)
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("freeze", [1, [0], [2]])
def test_frozen_layers_require_an_explicit_encoder_batchnorm(
    classifier_configuration, freeze
):
    """Frozen BatchNorm statistics are either fixed or adapted, never a silent default."""
    with pytest.raises(ValueError, match="encoder_batchnorm"):
        replace(classifier_configuration, freeze=freeze, encoder_batchnorm=None)
    values = {**classifier_configuration.to_dict(), "freeze": freeze}
    del values["encoder_batchnorm"]
    with pytest.raises(ValueError, match="encoder_batchnorm"):
        ClassifierConfiguration.from_mapping(values)


@pytest.mark.parametrize("source_format", ["supcon", "ultralytics"])
def test_trainer_accepts_existing_encoder_formats_and_writes_native_checkpoints(
    classifier_configuration,
    classifier_assets,
    source_format,
    native_runs,
):
    """Serialization format selects an input reader; the training backend remains YOLO."""
    config = replace(
        classifier_configuration,
        epochs=1,
        initialization={
            "encoder": {
                "source": source_format,
                "checkpoint": str(getattr(classifier_assets, source_format)),
            },
            "head": {"replace": "linear", "source": "random"},
        },
    )
    path = ClassifierExperiment(config).run()
    assert (
        len(native_runs) == 1
        and path == config.output_directory / "weights" / "last.pt"
    )
    assert YOLO(str(path), task="classify").model.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_trainer_honors_checkpoint_initialized_head(
    classifier_configuration, classifier_assets, native_runs, head_replace
):
    """The native trainer sees the prepared donor component, not a second initialization."""
    config = replace(
        classifier_configuration,
        epochs=1,
        initialization={
            **classifier_configuration.initialization,
            "head": {
                "replace": head_replace,
                "source": "yolo_checkpoint",
                "checkpoint": str(classifier_assets.ultralytics),
            },
        },
    )
    ClassifierExperiment(config).run()
    donor = YOLO(str(classifier_assets.ultralytics), task="classify").model
    prefix = _head_prefix(donor, head_replace)
    for name, tensor in donor.state_dict().items():
        if name.startswith(prefix):
            assert torch.equal(native_runs[0].before[name], tensor), name


@pytest.mark.parametrize("boundary", ["linear", "full"])
def test_native_finetuning_adapts_a_thousand_class_head_to_twenty_six_species(
    classifier_configuration,
    classifier_assets,
    native_runs,
    tmp_path,
    boundary,
):
    """Exercise the failed production shape transition through real native CPU training."""
    donor = native_checkpoint(
        classifier_assets.architecture,
        tmp_path / "imagenet-like.pt",
        names={index: f"imagenet_{index}" for index in range(1000)},
    )
    labels = [f"species_{index:02}" for index in range(26)]
    for split in ("train", "val"):
        for index, name in enumerate(labels):
            directory = tmp_path / "species-data" / split / name
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 32), (index * 8, 30, 90)).save(
                directory / "image.png"
            )
    config = replace(
        classifier_configuration,
        epochs=1,
        batch_size=26,
        class_names=labels,
        train_dataset=tmp_path / "species-data/train",
        validation_dataset=tmp_path / "species-data/val",
        initialization={
            **classifier_configuration.initialization,
            "head": {
                "replace": boundary,
                "source": "yolo_checkpoint",
                "checkpoint": str(donor.path),
            },
        },
    )
    checksum = hashlib.sha256(donor.path.read_bytes()).hexdigest()
    original_train = ClassificationTrainer.train
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def train(self, *args, **kwargs):
            if boundary == "linear":
                assert any(
                    "random" in str(w.message).lower()
                    and "1000" in str(w.message)
                    and "26" in str(w.message)
                    for w in caught
                )
            return original_train(self, *args, **kwargs)

        # Check warning timing at entry to the real native training method.
        from unittest.mock import patch

        with patch.object(ClassificationTrainer, "train", train):
            result = ClassifierExperiment(config).run_with_result()
    assert len(native_runs) == 1
    restored = YOLO(str(result.best_checkpoint), task="classify").model
    assert restored.model[-1].linear.out_features == 26
    assert restored.names == dict(enumerate(labels))
    assert hashlib.sha256(donor.path.read_bytes()).hexdigest() == checksum
    if boundary == "linear":
        assert any(
            "random" in str(w.message).lower()
            and "1000" in str(w.message)
            and "26" in str(w.message)
            for w in caught
        )


# Each direction and each tracking setting once.
@pytest.mark.parametrize("mode,tracking", [("min", False), ("max", True)])
def test_configured_metric_drives_native_best_stopping_and_tracking(
    classifier_configuration,
    tmp_path,
    monkeypatch,
    controlled_scores,
    native_runs,
    mode,
    tracking,
):
    """Configured fitness must beat native top-1 defaults and survive final evaluation."""
    tracking_config = None
    if tracking:
        from mlflow import MlflowClient

        tracking_config = MlflowConfiguration(
            tracking_uri=f"sqlite:///{tmp_path / 'ce.db'}", experiment_name="ce-tests"
        )
        client = MlflowClient(tracking_uri=tracking_config.tracking_uri)
        client.create_experiment(
            "ce-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
    # Force native autologging on in memory, without changing persisted Ultralytics settings.
    import importlib

    import mlflow

    utils = importlib.import_module("ultralytics.utils")
    callbacks = importlib.import_module("ultralytics.utils.callbacks.mlflow")
    monkeypatch.setattr(utils, "SETTINGS", {**utils.SETTINGS, "mlflow": True})
    monkeypatch.setattr(callbacks, "TESTS_RUNNING", False)
    autolog_calls = []
    monkeypatch.setattr(
        mlflow, "autolog", lambda *args, **kwargs: autolog_calls.append((args, kwargs))
    )
    original_train = ClassificationTrainer.train

    def train(self, *args, **kwargs):
        # Fail before any unsafe callback can touch a caller's tracking backend.
        assert not any(
            callback.__module__ == callbacks.__name__
            for group in self.callbacks.values()
            for callback in group
        )
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationTrainer, "train", train)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'unwanted.db'}")
    config = replace(
        classifier_configuration,
        epochs=8,
        mlflow=tracking_config,
        tracked_metric="balanced_accuracy",
        tracked_mode=mode,
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        train_kwargs={**classifier_configuration.train_kwargs, "patience": 2},
    )
    values = [0.8, 0.2, 0.3, 0.4] if mode == "min" else [0.2, 0.8, 0.7, 0.6]
    controlled_scores.score_sets = {0.002: values}
    result = ClassifierExperiment(config).run_with_result()
    # The configured score picks epoch 2, and patience 2 stops after epoch 4.
    assert result.best_epoch == 2 and result.best_value == values[1]
    assert len(controlled_scores.validations) == 4
    # Native MLflow autologging never runs, and the environment URI is not used.
    assert len(native_runs) == 1 and autolog_calls == []
    assert not (tmp_path / "unwanted.db").exists()
    card = yaml.safe_load(
        result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
    )
    assert card["validation"]["epoch"] == 2
    # best.pt keeps the complete native run history, not only its own epoch.
    best = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert best["train_results"]["epoch"][-1] == 4
    if tracking:
        run = client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED" and run.data.metrics["best_epoch"] == 2
        assert run.data.metrics["best_validation_balanced_accuracy"] == values[1]
        assert [
            metric.value
            for metric in client.get_metric_history(
                result.mlflow_run_id, "validation/balanced_accuracy"
            )
        ] == values
        assert len(client.search_runs([run.info.experiment_id])) == 1
    else:
        assert result.mlflow_run_id is None
    assert mlflow.active_run() is None


def test_empty_training_class_is_not_silently_ignored(classifier_configuration):
    """Requested species cannot silently disappear from native training."""
    for path in (classifier_configuration.train_dataset / "rat").glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="empty|nonempty"):
        ClassifierExperiment(classifier_configuration)
    assert not classifier_configuration.output_directory.exists()


def test_configuration_round_trip_uses_own_initialization_nodes(
    classifier_configuration, tmp_path
):
    """Native CE options and custom input/metric settings remain independent owned values."""
    document = classifier_configuration.to_dict()
    assert document["initialization"] == classifier_configuration.initialization
    assert document["train_kwargs"] == classifier_configuration.train_kwargs
    assert (
        not {"optimizer", "scheduler", "early_stopping", "freeze_encoder"}
        & document.keys()
    )
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(document))
    assert ClassifierConfiguration.from_yaml(path) == classifier_configuration
    assert ClassifierConfiguration.from_mapping(document) == classifier_configuration


def test_training_without_validation_is_explicit_and_still_reports_a_result(
    classifier_configuration, native_runs
):
    """Native training-only execution selects training loss without inventing validation."""
    import csv

    config = replace(classifier_configuration, validation_dataset=None)
    result = ClassifierExperiment(config).run_with_result()
    assert len(native_runs) == 1 and result.validation_dataset is None
    with (config.output_directory / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    losses = [
        float(next(value for key, value in row.items() if key.strip() == "train/loss"))
        for row in rows
    ]
    # Native logging rounds train/loss to 5 decimals before writing results.csv.
    assert result.best_value == pytest.approx(min(losses), abs=0.000005)
    card = yaml.safe_load(
        result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
    )
    assert card["validation"]["status"] == "not_validated"
    restored = YOLO(str(result.best_checkpoint), task="classify").model
    assert restored.transforms(Image.new("RGB", (51, 19))).shape == (3, 32, 32)


def test_training_only_native_selection_and_stopping_compare_every_epoch(
    classifier_configuration, native_runs, monkeypatch
):
    """A worsening native training loss must retain the first best epoch and stop early."""
    original = ClassificationTrainer.train
    losses = [0.2, 0.4, 0.5, 0.6, 0.7]
    completed = []

    def train(self, *args, **kwargs):
        def set_epoch_loss(trainer):
            completed.append(trainer.epoch + 1)
            trainer.tloss["loss"] = torch.tensor(
                losses[trainer.epoch], device=trainer.device
            )

        self.add_callback("on_train_epoch_end", set_epoch_loss)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationTrainer, "train", train)
    config = replace(
        classifier_configuration,
        validation_dataset=None,
        epochs=5,
        train_kwargs={**classifier_configuration.train_kwargs, "patience": 1},
    )
    result = ClassifierExperiment(config).run_with_result()
    assert result.best_epoch == 1 and result.best_value == pytest.approx(0.2)
    assert completed == [1, 2]
    assert result.validation_dataset is None
    snapshot = torch.load(
        result.best_checkpoint, map_location="cpu", weights_only=False
    )
    assert snapshot["train_metrics"]["ce_loss_mean"] == pytest.approx(0.2)
    card = yaml.safe_load(
        result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
    )
    assert card["validation"]["status"] == "not_validated"


@pytest.mark.parametrize("value", [float("nan"), None])  # non-finite, missing
def test_nonfinite_or_missing_tracked_metric_does_not_fall_back_to_native_fitness(
    classifier_configuration, monkeypatch, value
):
    """A valid native top-1 fitness cannot hide a missing/invalid requested objective."""
    original = ClassificationValidator.__call__

    def validate(self, trainer=None, model=None):
        metrics = original(self, trainer=trainer, model=model)
        if trainer is not None:
            metrics.pop("balanced_accuracy", None)
            if value is not None:
                metrics["balanced_accuracy"] = value
        return metrics

    monkeypatch.setattr(ClassificationValidator, "__call__", validate)
    config = replace(
        classifier_configuration,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    with pytest.raises(
        (ValueError, FloatingPointError), match="balanced_accuracy|tracked|validation"
    ):
        ClassifierExperiment(config).run()
    assert not list(config.output_directory.rglob("*.pt"))


def test_native_checkpoint_failure_propagates_and_releases_training_mode(
    classifier_configuration, native_runs, monkeypatch
):
    """A failed native save must not be reported as a completed run or keep training mode."""
    original = ClassificationTrainer.save_model
    writes = 0

    def save_model(self):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated native checkpoint failure")
        return original(self)

    monkeypatch.setattr(ClassificationTrainer, "save_model", save_model)
    with pytest.raises(OSError, match="checkpoint failure"):
        ClassifierExperiment(classifier_configuration).run()
    assert writes == 2
    model = native_runs[0].trainer.model
    assert all(not module.training for module in model.modules())
    last = classifier_configuration.output_directory / "weights" / "last.pt"
    assert torch.load(last, map_location="cpu", weights_only=False)["epoch"] == 0
    assert YOLO(str(last), task="classify").model.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"tracked_metric": "unknown"}, "tracked_metric"),
        ({"class_names": ("mouse", "mouse")}, "class_names"),
    ],
)
def test_invalid_classifier_configuration_fails_early(
    classifier_configuration, updates, message
):
    """Malformed or removed options are errors, not ignored compatibility aliases."""
    with pytest.raises((ValueError, TypeError), match=message):
        replace(classifier_configuration, **updates)
    assert not classifier_configuration.output_directory.exists()


# One key per ownership category (model, data, outputs, frozen boundary, run state,
# shared batching, evaluation policy, augmentation); the full set is defined once in
# classifier_training and every key takes the same rejection path.
@pytest.mark.parametrize(
    "key,value",
    [
        ("model", "other.pt"),
        ("data", "other-data"),
        ("project", "other-output"),
        ("freeze", 0),
        ("resume", True),
        ("batch", 999),
        ("val", False),
        ("auto_augment", "randaugment"),
    ],
)
def test_native_kwargs_cannot_override_owned_model_data_outputs_or_policy(
    classifier_configuration, key, value
):
    """One owner controls shared inputs, output layout, frozen boundary, and run state."""
    with pytest.raises(ValueError, match=key):
        replace(
            classifier_configuration,
            train_kwargs={**classifier_configuration.train_kwargs, key: value},
        )
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("replacement,freeze", [("linear", 3), ("full", 2)])
@pytest.mark.parametrize("dropout", [0.25, None])
def test_native_dropout_configures_only_the_head_dropout_for_every_boundary(
    classifier_configuration,
    classifier_assets,
    native_runs,
    tmp_path,
    replacement,
    freeze,
    dropout,
):
    """train_kwargs.dropout sets the Classify dropout before Linear; omitted means native 0.0.

    The encoder's own Dropout (layer 1, p=0.5) is never rewritten, unlike native
    get_model(), which changes every Dropout module in the network.
    """
    # The source head has drop.p=0.35, so the omitted case proves an active reset.
    source = native_checkpoint(
        classifier_assets.architecture, tmp_path / "dropout-source.pt"
    )
    initialization = {
        "encoder": {"source": "ultralytics", "checkpoint": str(source.path)},
        "head": {"replace": replacement, "source": "random"},
    }
    options = dict(classifier_configuration.train_kwargs)
    if dropout is not None:
        options["dropout"] = dropout
    config = replace(
        classifier_configuration,
        epochs=1,
        freeze=freeze,
        initialization=initialization,
        train_kwargs=options,
    )
    result = ClassifierExperiment(config).run_with_result()
    expected = 0.0 if dropout is None else dropout
    record = native_runs[0]
    network = record.trainer.model
    assert network.model[-1].drop.p == expected
    assert network.model[1].p == 0.5
    name = f"model.{len(network.model) - 1}.drop"
    assert record.modes and all(modes[name] for modes in record.modes)
    assert (
        YOLO(str(result.best_checkpoint), task="classify").model.model[-1].drop.p
        == expected
    )


def test_freeze_index_prefixes_do_not_capture_later_layers(
    classifier_configuration, native_runs, tmp_path
):
    """Freezing layer 1 must leave layer 10 trainable and in training mode."""
    architecture = tmp_path / "deep-cls.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [
                    [-1, 1, "Conv", [8, 3, 2]],
                    *([-1, 1, "nn.Dropout", [0.1]] for _ in range(9)),
                    [-1, 1, "Conv", [8, 3, 1]],
                ],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    source = native_checkpoint(architecture, tmp_path / "deep-cls.pt")
    config = replace(
        classifier_configuration,
        epochs=1,
        freeze=[1],
        initialization={
            "encoder": {"source": "ultralytics", "checkpoint": str(source.path)},
            "head": {"replace": "linear", "source": "random"},
        },
    )
    ClassifierExperiment(config).run()
    record = native_runs[0]
    network = record.trainer.model
    assert len(network.model) == 12
    assert all(parameter.requires_grad for parameter in network.parameters())
    assert record.modes
    for modes in record.modes:
        assert not modes["model.1"]
        assert all(
            modes[name]
            for name, _ in _mode_dependent_modules(network)
            if not _is_frozen(name, ("model.1.",))
        )


def test_unknown_native_training_options_fail_before_execution(
    classifier_configuration,
):
    """Native argument validation must reject misspelled options without starting training."""
    with pytest.raises((ValueError, SyntaxError), match="not_a_native_option"):
        replace(classifier_configuration, train_kwargs={"not_a_native_option": True})
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
def test_output_cannot_be_inside_either_input_split(classifier_configuration, split):
    with pytest.raises(ValueError, match="outside|output_directory"):
        replace(
            classifier_configuration,
            output_directory=getattr(classifier_configuration, split) / "output",
        )


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
@pytest.mark.parametrize("artifact", ["weights", "args.yaml"])
def test_native_output_aliases_cannot_redirect_writes_into_input_splits(
    classifier_configuration, split, artifact
):
    """Protect split contents even when the run root itself is outside both datasets."""
    config = classifier_configuration
    dataset = getattr(config, split)
    protected = dataset / "protected.yaml"
    protected.write_bytes(b"read-only dataset metadata")
    config.output_directory.mkdir()
    target = dataset if artifact == "weights" else protected
    (config.output_directory / artifact).symlink_to(
        target, target_is_directory=artifact == "weights"
    )
    with pytest.raises(ValueError, match="outside|dataset"):
        replace(config)
    assert protected.read_bytes() == b"read-only dataset metadata"
    assert not (dataset / "best.pt").exists()
    assert not (dataset / "last.pt").exists()


# Fixed native names and the epoch-dependent names that are only caught by the plot
# and weights-directory globs; alternating which split is hit.
@pytest.mark.parametrize(
    "split,artifact",
    [
        ("train_dataset", "args.yaml"),
        ("train_dataset", "weights/last.pt"),
        ("validation_dataset", "weights/epoch0.pt"),
        ("train_dataset", "train_batch0.jpg"),
    ],
)
def test_existing_output_hardlinks_cannot_truncate_dataset_images(
    classifier_configuration, split, artifact
):
    """Check output inode ownership without scanning every image in large input datasets."""
    config = classifier_configuration
    protected = next(getattr(config, split).rglob("*.png"))
    before = protected.read_bytes()
    destination = config.output_directory / artifact
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(protected, destination)
    with pytest.raises(ValueError, match="hardlink|source|dataset"):
        replace(config)
    assert protected.read_bytes() == before
    assert destination.samefile(protected)


def test_native_output_directory_symlink_keeps_the_validated_destination(
    classifier_configuration, native_runs, tmp_path
):
    """Native args, weights and our config must share the same resolved run directory."""
    actual = tmp_path / "native-parent" / "actual-run"
    actual.mkdir(parents=True)
    link = tmp_path / "requested-link"
    link.symlink_to(actual, target_is_directory=True)
    unrelated = actual.parent / link.name
    unrelated.mkdir()
    marker = unrelated / "args.yaml"
    marker.write_text("protected: unrelated run\n")
    config = replace(classifier_configuration, epochs=1, output_directory=link)
    result = ClassifierExperiment(config).run_with_result()
    assert native_runs[0].trainer.save_dir.resolve() == actual
    assert result.best_checkpoint.resolve() == actual / "weights/best.pt"
    assert (actual / "config.yaml").is_file() and (actual / "results.csv").is_file()
    assert marker.read_text() == "protected: unrelated run\n"
    assert not (unrelated / "weights").exists()


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize(
    "alias",
    [
        "val/loss",  # a fixed native column name
        "lr/pg99",  # any learning-rate parameter-group column
    ],
)
def test_native_reserved_metric_aliases_are_rejected_before_outputs(
    classifier_configuration, alias, tracked
):
    """Configured sklearn aliases must never be overwritten by native diagnostics or fitness."""
    with pytest.raises(ValueError, match="reserved|override"):
        replace(
            classifier_configuration,
            validation_metrics={alias: {"function": "balanced_accuracy_score"}},
            tracked_metric=alias if tracked else "ce_loss_mean",
        )
    assert not classifier_configuration.output_directory.exists()


def test_native_csv_preserves_bookkeeping_training_loss_and_configured_scores(
    classifier_configuration, native_runs
):
    """Native history must keep unique native fields alongside independent sklearn scores."""
    import csv

    config = replace(
        classifier_configuration,
        epochs=2,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    result = ClassifierExperiment(config).run_with_result()
    with (config.output_directory / "results.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert len(fields) == len(set(fields))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert all(float(row["time"]) >= 0 for row in rows)
    trainer = native_runs[0].trainer
    assert float(rows[-1]["train/loss"]) == pytest.approx(
        float(trainer.tloss["loss"]), abs=0.00005
    )
    assert float(rows[-1]["lr/pg0"]) == pytest.approx(trainer.lr["lr/pg0"])
    assert float(rows[result.best_epoch - 1]["balanced_accuracy"]) == pytest.approx(
        result.best_value
    )


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    batch_size=st.integers(1, 16),
    batchnorm=st.sampled_from(["eval", "adapt"]),
    lr=st.floats(
        min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False
    ),
    freeze=st.one_of(
        st.integers(0, 64), st.lists(st.integers(0, 64), unique=True, max_size=6)
    ),
)
def test_configuration_preserves_batch_native_options_bn_and_freeze_policy(
    classifier_configuration, batch_size, batchnorm, lr, freeze
):
    config = replace(
        classifier_configuration,
        batch_size=batch_size,
        # The BatchNorm policy only exists when some layer is frozen.
        encoder_batchnorm=batchnorm if freeze not in (0, []) else None,
        freeze=freeze,
        train_kwargs={**classifier_configuration.train_kwargs, "lr0": lr},
    )
    assert ClassifierConfiguration.from_mapping(config.to_dict()) == config


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(indices=st.lists(st.integers(0, 64), min_size=1, max_size=6), data=st.data())
def test_duplicate_freeze_indices_are_rejected(classifier_configuration, indices, data):
    """A repeated layer index is an ambiguous configuration, not a silently deduplicated one."""
    duplicated = [*indices, data.draw(st.sampled_from(indices))]
    with pytest.raises(ValueError, match="freeze"):
        replace(classifier_configuration, freeze=duplicated)


@pytest.mark.parametrize(
    "metric,mode", [("ce_loss_mean", "min"), ("balanced_accuracy", "max")]
)
def test_omitted_direction_uses_the_configured_metric_default(
    classifier_configuration, metric, mode
):
    config = replace(
        classifier_configuration,
        tracked_metric=metric,
        tracked_mode=None,
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    assert config.effective_tracked_mode == mode


@pytest.mark.parametrize(
    "node,artifact,alias",
    [
        ("encoder", "weights/last.pt", "direct"),
        ("head", "weights/best.pt", "direct"),
        ("encoder", "args.yaml", "artifact_symlink"),
        ("encoder", "weights/best.model_card.yaml", "artifact_symlink"),
        ("encoder", "weights/best.pt", "source_symlink"),
        ("encoder", "args.yaml", "hardlink"),
    ],
)
def test_output_artifacts_cannot_overwrite_initialization_sources(
    classifier_configuration, classifier_assets, node, artifact, alias
):
    """Native output names must retain direct/symlink/hardlink input protection."""
    config = classifier_configuration
    output = config.output_directory
    destination = output / artifact
    destination.parent.mkdir(parents=True, exist_ok=True)
    initialization = deepcopy(config.initialization)
    original = (
        classifier_assets.supcon if node == "encoder" else classifier_assets.ultralytics
    )
    if alias == "direct":
        destination.write_bytes(original.read_bytes())
        source = destination
    elif alias == "artifact_symlink":
        destination.symlink_to(original)
        source = original
    elif alias == "source_symlink":
        destination.write_bytes(original.read_bytes())
        source = output.parent / "source-link.pt"
        source.symlink_to(destination)
    else:
        os.link(original, destination)
        source = original
    initialization[node].update(
        source="yolo_checkpoint" if node == "head" else "supcon", checkpoint=str(source)
    )
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    with pytest.raises(ValueError, match="checkpoint|source|collision|collid"):
        replace(config, initialization=initialization)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    assert sorted(str(path.relative_to(output)) for path in output.rglob("*")) == before


# --- Per-epoch observer hook (used by Optuna pruning; the trainer stays Optuna-free) ---


def _balanced_accuracy_configuration(configuration, **changes):
    """Track a controllable validation score instead of the fixture's CE loss."""
    return replace(
        configuration,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        **changes,
    )


def test_epoch_callback_observes_each_native_epoch_after_its_checkpoint(
    classifier_configuration, controlled_scores
):
    """The hook sees each epoch's configured score, never native fitness, once last.pt exists."""
    config = _balanced_accuracy_configuration(classifier_configuration, epochs=3)
    controlled_scores.score_sets = {0.002: [0.2, 0.8, 0.6]}
    observed = []

    def callback(epoch, value):
        """Record the call and whether the epoch's native checkpoint is already written."""
        observed.append(
            (epoch, value, (config.output_directory / "weights/last.pt").is_file())
        )

    result = ClassifierExperiment(config).run_with_result(epoch_callback=callback)
    assert observed == [(1, 0.2, True), (2, 0.8, True), (3, 0.6, True)]
    assert result.best_epoch == 2 and result.best_value == 0.8


def test_epoch_callback_ends_with_native_early_stopping(
    classifier_configuration, controlled_scores
):
    """Native patience ends the hook sequence; the final best-model evaluation is not an epoch."""
    config = _balanced_accuracy_configuration(
        classifier_configuration,
        epochs=6,
        train_kwargs={**classifier_configuration.train_kwargs, "patience": 1},
    )
    controlled_scores.score_sets = {0.002: [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]}
    observed = []
    ClassifierExperiment(config).run_with_result(
        epoch_callback=lambda epoch, value: observed.append((epoch, value))
    )
    assert observed == [(1, 0.5), (2, 0.4)]


def test_epoch_callback_exception_stops_native_training_and_propagates_unchanged(
    classifier_configuration, controlled_scores
):
    """A hook may stop a run (e.g. pruning) before any model card or publication is written."""

    class StopRequested(Exception):
        """A caller-owned exception type that must not be wrapped or swallowed."""

    def callback(epoch, value):
        raise StopRequested(f"stop after epoch {epoch}")

    config = _balanced_accuracy_configuration(classifier_configuration, epochs=3)
    controlled_scores.score_sets = {0.002: [0.2, 0.8, 0.6]}
    with pytest.raises(StopRequested, match="stop after epoch 1"):
        ClassifierExperiment(config).run_with_result(epoch_callback=callback)
    assert len(controlled_scores.validations) == 1
    assert not (config.output_directory / "weights/best.model_card.yaml").exists()


@pytest.fixture
def standalone_configuration(pipeline_document, tmp_path):
    """Train a real local SupCon source, then configure standalone CE without pipeline context."""
    source_values = deepcopy(pipeline_document["supcon"])
    source_values.update(pipeline_document["data"])
    source_values.update(
        epochs=1,
        optimizer={"lr": 0.001},
        output_directory=str(tmp_path / "standalone-source"),
    )
    source = SupConExperiment(
        TrainingConfiguration.from_mapping(source_values)
    ).run_with_result()
    values = deepcopy(pipeline_document["classifier"])
    values.update(pipeline_document["data"])
    values.update(
        epochs=3,
        output_directory=str(tmp_path / "standalone-ce"),
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    values["initialization"]["encoder"] = {
        "source": "supcon",
        "checkpoint": str(source.best_checkpoint),
    }
    return ClassifierConfiguration.from_mapping(values)


@pytest.mark.parametrize("tracking", [False, True])
def test_artifact_only_runs_retain_selected_metadata_without_extra_model_restore(
    standalone_configuration, tracking_store, scripted_validation, monkeypatch, tracking
):
    """Cards reuse native-finalized metadata without another model reconstruction."""
    import smartrodent.classifier_model as classifier_models

    restores = []
    original = classifier_models.load_checkpoint

    def restore(weight, *args, **kwargs):
        restores.append(Path(weight).resolve())
        return original(weight, *args, **kwargs)

    monkeypatch.setattr(classifier_models, "load_checkpoint", restore)
    config = replace(
        standalone_configuration,
        mlflow=(
            MlflowConfiguration(**tracking_store.configuration) if tracking else None
        ),
    )
    result = ClassifierExperiment(
        config, encoder_provenance={"encoder_run_id": "selected-encoder-run"}
    ).run_with_result()
    assert restores == []
    assert result.best_epoch == 2 and result.best_value == 0.9
    card = yaml.safe_load(
        result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
    )
    assert card["classes"] == {0: "mouse", 1: "rat"}
    assert card["validation"]["epoch"] == 2
    assert card["validation"]["metrics"]["balanced_accuracy"] == 0.9
    assert card["provenance"]["encoder_run_id"] == "selected-encoder-run"
    assert (
        card["provenance"]["checkpoint_sha256"]
        == hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
    )
    checkpoint = YOLO(str(result.best_checkpoint), task="classify").model
    assert checkpoint.names == card["classes"]
    assert checkpoint.provenance["encoder_run_id"] == "selected-encoder-run"
    if tracking:
        run = tracking_store.client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED" and run.data.metrics["best_epoch"] == 2
        downloaded = download_artifact(
            tracking_store, result.mlflow_run_id, "best_model/best.model_card.yaml"
        )
        assert (
            downloaded.read_bytes()
            == result.best_checkpoint.with_suffix(".model_card.yaml").read_bytes()
        )
    else:
        assert result.mlflow_run_id is None


def test_corrupt_native_best_checkpoint_cannot_be_reported_or_published(
    standalone_configuration, monkeypatch
):
    """Skipping unnecessary model reconstruction must not hide a corrupt selected file."""
    from pickle import UnpicklingError

    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    original = ClassificationTrainer.save_model
    writes = []

    def save(self):
        original(self)
        writes.append(self.best)
        self.best.write_bytes(b"corrupt native best checkpoint")

    monkeypatch.setattr(ClassificationTrainer, "save_model", save)
    config = replace(standalone_configuration, epochs=1)
    with pytest.raises(
        (ValueError, UnpicklingError), match="checkpoint|finaliz|pickle"
    ):
        ClassifierExperiment(config).run_with_result()
    assert writes == [config.output_directory / "weights/best.pt"]
    assert not (config.output_directory / "weights/best.model_card.yaml").exists()
    assert not (config.output_directory / "weights/best.model_card.md").exists()


@pytest.mark.parametrize("validation", [False, True])
def test_standalone_ce_registers_its_best_classifier_and_only_promotes_validated_models(
    standalone_configuration, tracking_store, scripted_validation, tmp_path, validation
):
    """Standalone publication remains root-level; no-val runs cannot replace a validated champion."""
    configuration = standalone_configuration
    previous_version = None
    if not validation:
        configuration = replace(
            configuration,
            validation_dataset=None,
            tracked_metric="ce_loss_mean",
            tracked_mode="min",
            validation_metrics=None,
        )
        tracking_store.client.create_registered_model(tracking_store.registry_name)
        prior = tmp_path / "prior-validated-model"
        prior.mkdir()
        previous_version = tracking_store.client.create_model_version(
            tracking_store.registry_name, prior.as_uri()
        ).version
        tracking_store.client.set_registered_model_alias(
            tracking_store.registry_name,
            tracking_store.configuration["champion_alias"],
            previous_version,
        )
    configuration = replace(
        configuration,
        mlflow=MlflowConfiguration(
            **tracking_store.configuration,
            registered_model_name=tracking_store.registry_name,
        ),
    )
    before = deepcopy(configuration.to_dict())
    result = ClassifierExperiment(configuration).run_with_result()
    assert configuration.to_dict() == before
    assert result.mlflow_run_id
    run = tracking_store.client.get_run(result.mlflow_run_id)
    assert run.info.status == "FINISHED" and "mlflow.parentRunId" not in run.data.tags
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    versions = tracking_store.client.search_model_versions(
        f"name='{tracking_store.registry_name}'"
    )
    published = [
        version for version in versions if version.run_id == result.mlflow_run_id
    ]
    assert len(published) == 1
    version = published[0]
    assert version.tags["validation_status"] == (
        "passed" if validation else "not_validated"
    )
    if validation:
        assert result.best_epoch == 2
        assert_registered_classifier(tracking_store, result)
        expected_alias = version.version
    else:
        card = yaml.safe_load(
            result.best_checkpoint.with_suffix(".model_card.yaml").read_text()
        )
        assert card["validation"]["status"] == "not_validated"
        assert str(card["model"]["registry_version"]) == str(version.version)
        expected_alias = previous_version
    assert (
        tracking_store.client.get_model_version_by_alias(
            tracking_store.registry_name, tracking_store.configuration["champion_alias"]
        ).version
        == expected_alias
    )
    assert mlflow.active_run() is None


def test_out_of_range_freeze_fails_before_any_tracking_run_or_output(
    standalone_configuration, tracking_store
):
    """Preparation errors surface before tracking starts, so no failed run is left behind."""
    # Pipeline network: 0=Conv encoder, 1=Classify head; a linear probe can freeze 0..1.
    config = replace(
        standalone_configuration,
        freeze=3,
        mlflow=MlflowConfiguration(**tracking_store.configuration),
    )
    with pytest.raises(ValueError, match="freeze"):
        ClassifierExperiment(config).run_with_result()
    assert tracking_store.client.search_runs([tracking_store.experiment_id]) == []
    assert not config.output_directory.exists()
    assert mlflow.active_run() is None


def _exporting_classifier(
    standalone_configuration, pipeline_document, tracking_store, tmp_path, *, register
):
    """Configure a tracked CE run on an encoder wide enough for native ONNX export."""
    encoder = wide_native_encoder(
        Path(pipeline_document["supcon"]["model_name"]), tmp_path / "wide-cls.pt"
    )
    settings = {**tracking_store.configuration}
    if register:
        settings["registered_model_name"] = tracking_store.registry_name
    return replace(
        standalone_configuration,
        onnx_export={},
        mlflow=MlflowConfiguration(**settings),
        initialization={
            "encoder": {"source": "ultralytics", "checkpoint": str(encoder)},
            "head": {"replace": "linear", "source": "random"},
        },
    )


@pytest.mark.parametrize("register", [False, True])
def test_tracked_classifier_logs_its_onnx_export_with_the_best_model(
    standalone_configuration,
    pipeline_document,
    tracking_store,
    scripted_validation,
    tmp_path,
    register,
):
    """The exported file is published next to best.pt, with or without registration."""
    config = _exporting_classifier(
        standalone_configuration,
        pipeline_document,
        tracking_store,
        tmp_path,
        register=register,
    )
    result = ClassifierExperiment(config).run_with_result()
    downloaded = download_artifact(
        tracking_store, result.mlflow_run_id, "best_model/best.onnx"
    )
    assert downloaded.read_bytes() == result.onnx_model.read_bytes()


def test_failed_classifier_export_fails_the_run_before_publication(
    standalone_configuration,
    pipeline_document,
    tracking_store,
    scripted_validation,
    tmp_path,
    monkeypatch,
):
    """A failed export leaves a failed run: no ONNX artifact and no registered version."""

    def fail(*args, **kwargs):
        raise RuntimeError("simulated ONNX export failure")

    config = _exporting_classifier(
        standalone_configuration,
        pipeline_document,
        tracking_store,
        tmp_path,
        register=True,
    )
    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="export failure"):
        ClassifierExperiment(config).run_with_result()
    (run,) = tracking_store.client.search_runs([tracking_store.experiment_id])
    assert run.info.status == "FAILED"
    assert "best_model" not in {
        item.path for item in tracking_store.client.list_artifacts(run.info.run_id)
    }
    assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None


def test_standalone_ce_tracking_without_registration_remains_a_single_completed_run(
    standalone_configuration, tracking_store, scripted_validation
):
    """Adding pipeline contexts must not require one or silently register standalone tracking runs."""
    config = replace(
        standalone_configuration,
        mlflow=MlflowConfiguration(**tracking_store.configuration),
    )
    result = ClassifierExperiment(config).run_with_result()
    run = tracking_store.client.get_run(result.mlflow_run_id)
    assert run.info.status == "FINISHED" and "mlflow.parentRunId" not in run.data.tags
    assert (
        run.data.metrics["best_epoch"] == 2
        and run.data.metrics["best_validation_balanced_accuracy"] == 0.9
    )
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None
