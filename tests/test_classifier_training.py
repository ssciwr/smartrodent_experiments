"""Contracts for the standalone CE trainer, reusing the approved classifier model."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from ultralytics import YOLO

from smartrodent.classifier_model import YoloClassifierModel
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration, SchedulerConfiguration
from smartrodent.supcon_model import YoloSupConModel


@pytest.fixture
def training_api():
    """Require the new trainer without preventing existing test collection."""
    from smartrodent import classifier_training

    return classifier_training


@pytest.fixture
def classifier_assets(tmp_path):
    """Create tiny temporary data and all three supported encoder checkpoint formats."""
    torch.set_num_threads(2)
    for split, counts in [("train", (5, 5)), ("val", (3, 2))]:
        for label, (name, count) in enumerate(zip(("mouse", "rat"), counts, strict=True)):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (16, 16), (30 + 20 * index, 255 * label, 80)).save(
                    directory / f"{index}.png"
                )
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(yaml.safe_dump({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "nn.Dropout", [0.5]]],
        "head": [[-1, 1, "Classify", [2]]],
    }))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        source = YoloSupConModel(str(architecture), num_classes=2, output_features=4)
        source.requires_grad_(True)
        source.train()
        source.label_names = {"mouse": 0, "rat": 1}
        source.full_model.model.names = {0: "mouse", 1: "rat"}
        optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        _, embeddings = source(torch.rand(4, 3, 16, 16))
        SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
        optimizer.step()
        for state in optimizer.state.values():
            state["step"].fill_(99)
        supcon = tmp_path / "supcon-best.pt"
        torch.save({
            "epoch": 99,
            "architecture": deepcopy(source.full_model.model.yaml),
            "model_configuration": {"dropout": 0.0, "output_features": 4, "num_classes": 2, "freeze": []},
            "model_state_dict": source.state_dict(),
            "label_names": dict(source.label_names),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {"best": -123.0, "num_bad_epochs": 50},
        }, supcon)
        source.finalize()
        native = tmp_path / "native-cls.pt"
        yolo = YOLO(str(architecture), task="classify")
        yolo.model.names = {0: "mouse", 1: "rat"}
        yolo.save(native)
        classifier = tmp_path / "classifier-source.pt"
        converted = YoloClassifierModel.initialize(
            supcon, source_format="supcon", class_names=("mouse", "rat")
        )
        torch.save(converted.checkpoint_state(), classifier)
    return SimpleNamespace(
        train=tmp_path / "train", val=tmp_path / "val",
        supcon=supcon, ultralytics=native, classifier=classifier,
    )


@pytest.fixture
def classifier_configuration(training_api, classifier_assets, tmp_path):
    """Use plain CE batch sizes, with no SupCon sampler or projection settings."""
    return training_api.ClassifierConfiguration(
        train_dataset=classifier_assets.train,
        validation_dataset=classifier_assets.val,
        output_directory=tmp_path / "ce-output",
        class_names=("rat", "mouse"),
        initialization={
            "encoder": {"source": "supcon", "checkpoint": str(classifier_assets.supcon)},
            "head": {"replace": "linear", "source": "random"},
        },
        freeze_encoder=True,
        encoder_batchnorm="eval",
        transforms={
            "image_size": 16,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
            "random_grayscale": {"p": 0.0},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
            "random_horizontal_flip": {"p": 0.0},
            "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
        },
        optimizer={"lr": 0.002},
        epochs=3, batch_size=4, workers=0, device="cpu", seed=42,
        tracked_metric="ce_loss_mean", tracked_mode="min",
    )


@pytest.fixture
def initialized_models(monkeypatch):
    """Observe real initialized state without replacing the model or training behavior."""
    observed = []
    original = YoloClassifierModel.initialize

    def initialize(cls, *args, **kwargs):
        """Retain the live model and independent pre-training tensors for behavioral checks."""
        model = original(*args, **kwargs)
        observed.append((model, deepcopy(model.state_dict())))
        return model

    monkeypatch.setattr(YoloClassifierModel, "initialize", classmethod(initialize))
    return observed


def _encoder_prefix(model):
    """Identify the explicitly selected replacement boundary in native YOLO state names."""
    name = list(model.network.model.named_children())[-1][0]
    return f"network.model.{name}." + ("linear." if model.head_replace == "linear" else "")


def _signature(loader):
    """Describe single-view image identities and labels independent of sampler classes."""
    batches = list(loader)
    assert all(images.ndim == 4 and len(labels) == len(images) for labels, images in batches)
    return [(int(label), round(float(image[0, 0, 0]) * 255)) for labels, images in batches for label, image in zip(labels, images, strict=True)]


def test_training_loader_is_single_view_shuffled_and_covers_source_once(training_api, classifier_configuration):
    """CE uses ordinary epochs, not P–K resampling, two views, or a dropped tail."""
    trainer = training_api.ClassifierExperiment(classifier_configuration)
    dataset, loader = trainer._build_loader()
    first = _signature(loader)
    second = _signature(loader)
    assert len(dataset) == len(first) == len(second) == 10
    assert Counter(first) == Counter(second)
    assert len(set(first)) == 10
    assert Counter(label for label, _ in first) == {0: 5, 1: 5}
    assert first != second
    _, replay = training_api.ClassifierExperiment(classifier_configuration)._build_loader()
    assert _signature(replay) == first


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_validation_loader_is_complete_and_repeatable(training_api, classifier_configuration, batch_size):
    """The existing val split is traversed exactly once, including incomplete/small batches."""
    config = replace(classifier_configuration, batch_size=batch_size)
    loader = training_api.ClassifierExperiment(config)._build_validation_loader()
    first = _signature(loader)
    expected = [
        (int(label), round(float(image[0, 0, 0]) * 255))
        for label, image in (loader.dataset[index] for index in range(len(loader.dataset)))
    ]
    assert first == expected
    assert len(first) == len(set(first)) == 5
    assert Counter(label for label, _ in first) == {0: 3, 1: 2}
    assert _signature(loader) == first
    assert len(list(loader)[-1][0]) == (5 % batch_size or min(5, batch_size))


class _LogitProbe(torch.nn.Module):
    """Give deliberately unequal batch losses while recording evaluation behavior."""

    def __init__(self):
        """Record mode and gradient state for every evaluated image."""
        super().__init__()
        self.observed = []

    @staticmethod
    def logits(images):
        """Return raw scores biased toward class zero, varying by image identity."""
        red = images[:, 0, 0, 0]
        return torch.stack([3 * red, -red], dim=1)

    def forward(self, images):
        """Observe evaluation context without changing the score calculation."""
        self.observed.append((len(images), self.training, torch.is_grad_enabled()))
        return self.logits(images)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("initial_training", [False, True])
def test_validation_uses_raw_logits_sample_weighted_ce_and_all_labels(
    training_api, classifier_configuration, batch_size, initial_training
):
    """An unweighted average of batch losses or softmax-before-CE must fail this oracle."""
    config = replace(
        classifier_configuration, batch_size=batch_size,
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
    )
    trainer = training_api.ClassifierExperiment(config)
    loader = trainer._build_validation_loader()
    batches = list(loader)
    labels = torch.cat([batch[0] for batch in batches])
    images = torch.cat([batch[1] for batch in batches])
    model = _LogitProbe().train(initial_training)
    expected = torch.nn.functional.cross_entropy(model.logits(images), labels).item()
    if len(batches) > 1:
        naive = sum(torch.nn.functional.cross_entropy(model.logits(x), y).item() for y, x in batches) / len(batches)
        assert not math.isclose(expected, naive, rel_tol=1e-5)
    metrics = trainer._validate(model, loader)
    assert metrics["ce_loss_mean"] == pytest.approx(expected)
    assert metrics["balanced_accuracy"] == 0.5
    assert sum(count for count, _, _ in model.observed) == 5
    assert all(not training and not gradients for _, training, gradients in model.observed)
    assert model.training == initial_training


def test_training_ce_uses_raw_logits_not_softmax_scores(
    training_api, classifier_configuration, initialized_models
):
    """Compare a real one-batch update to CE on logits; parameter changes alone are insufficient."""
    config = replace(classifier_configuration, epochs=1, batch_size=10)
    result = training_api.ClassifierExperiment(config).run_with_result()
    model, before = initialized_models[0]
    expected_model = deepcopy(model)
    expected_model.load_state_dict(before)
    expected_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in expected_model.parameters() if parameter.requires_grad), **config.optimizer
    )
    _, loader = training_api.ClassifierExperiment(config)._build_loader()
    labels, images = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(expected_model(images), labels).backward()
    optimizer.step()
    actual = torch.load(result.last_checkpoint, weights_only=True)["optimizer_state_dict"]["state"]
    # AdamW can hide a uniform gradient scaling in its first weight update. Its
    # saved moments expose softmax-before-CE even if those weights look similar.
    # The default linear head has distinct weight/bias shapes, so IDs/order need
    # not be assumed when matching their observable optimizer state.
    actual_moments = {tuple(state["exp_avg"].shape): state["exp_avg"] for state in actual.values()}
    expected_moments = {tuple(state["exp_avg"].shape): state["exp_avg"] for state in optimizer.state.values()}
    assert actual_moments.keys() == expected_moments.keys()
    for shape in expected_moments:
        torch.testing.assert_close(actual_moments[shape], expected_moments[shape], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("head_replace", ["linear", "full"])
@pytest.mark.parametrize("freeze,batchnorm", [(True, "eval"), (True, "adapt"), (False, "eval"), (False, "adapt")])
def test_real_ce_training_honors_freezing_and_starts_a_fresh_filtered_optimizer(
    training_api, classifier_configuration, initialized_models, head_replace, freeze, batchnorm
):
    """CE updates its head without resuming source optimizer state or undoing the freeze policy."""
    config = replace(
        classifier_configuration, freeze_encoder=freeze, encoder_batchnorm=batchnorm,
        initialization={**classifier_configuration.initialization, "head": {"replace": head_replace, "source": "random"}},
    )
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert isinstance(result, ExperimentResult)
    model, before = initialized_models[0]
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert not model.training and all(not module.training for module in model.modules())
    boundary = _encoder_prefix(model)
    parameters = dict(model.named_parameters())
    encoder = {name: value for name, value in parameters.items() if not name.startswith(boundary)}
    head = {name: value for name, value in parameters.items() if name.startswith(boundary)}
    assert head and encoder
    assert any(not torch.equal(before[name], value) for name, value in head.items())
    assert any(not torch.equal(before[name], value) for name, value in encoder.items()) == (not freeze)
    assert all(value.requires_grad == (not freeze) for value in encoder.values())
    assert all(value.requires_grad for value in head.values())
    bn_buffers = {name: value for name, value in model.named_buffers() if not name.startswith(boundary) and name.endswith(("running_mean", "running_var"))}
    assert bn_buffers
    assert any(not torch.equal(before[name], value) for name, value in bn_buffers.items()) == (batchnorm == "adapt")
    if freeze and batchnorm == "eval":
        for name, value in model.state_dict().items():
            if not name.startswith(boundary):
                assert torch.equal(before[name], value), name
    optimizer = last["optimizer_state_dict"]
    assert len(optimizer["param_groups"][0]["params"]) == sum(parameter.requires_grad for parameter in model.parameters())
    assert optimizer["param_groups"][0]["lr"] == config.optimizer["lr"]
    steps = math.ceil(10 / config.batch_size) * config.epochs
    assert optimizer["state"] and all(int(state["step"]) == steps for state in optimizer["state"].values())
    assert last["scheduler_state_dict"] is None
    assert last["epoch"] == config.epochs and len(last["history"]) == config.epochs
    assert last["checkpoint_type"] == "classifier"
    assert result.tracked_metric == "ce_loss_mean" and result.tracked_mode == "min"
    assert result.validation_dataset == config.validation_dataset


@pytest.mark.parametrize("source_format", ["supcon", "ultralytics", "classifier"])
def test_trainer_wires_explicit_encoder_formats_to_the_existing_initializer(
    training_api, classifier_configuration, classifier_assets, source_format
):
    """Standalone CE supports each approved source format without using the old joint trainer."""
    config = replace(
        classifier_configuration, epochs=1,
        initialization={
            "encoder": {"source": source_format, "checkpoint": str(getattr(classifier_assets, source_format))},
            "head": {"replace": "linear", "source": "random"},
        },
    )
    path = training_api.ClassifierExperiment(config).run()
    assert path == config.output_directory / "last.pt"
    restored = YoloClassifierModel.from_checkpoint(path)
    assert restored.label_names == {"mouse": 0, "rat": 1}


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_trainer_honors_checkpoint_initialized_head(
    training_api, classifier_configuration, classifier_assets, initialized_models, head_replace
):
    """The classifier initialization node must reach the existing native-head loader."""
    config = replace(
        classifier_configuration, epochs=1,
        initialization={
            **classifier_configuration.initialization,
            "head": {"replace": head_replace, "source": "yolo_checkpoint", "checkpoint": str(classifier_assets.ultralytics)},
        },
    )
    training_api.ClassifierExperiment(config).run()
    model, before = initialized_models[0]
    donor = YOLO(str(classifier_assets.ultralytics), task="classify").model.model[-1]
    component = donor.linear if head_replace == "linear" else donor
    prefix = _encoder_prefix(model)
    for name, tensor in component.state_dict().items():
        assert torch.equal(before[prefix + name], tensor), name


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("tracking", [False, True])
def test_configured_metric_drives_best_stopping_scheduler_and_tracking(
    training_api, classifier_configuration, tmp_path, monkeypatch, mode, tracking
):
    """The user's metric, not CE loss, chooses best.pt and the reported best value."""
    tracking_config = None
    if tracking:
        from mlflow import MlflowClient

        tracking_config = MlflowConfiguration(
            tracking_uri=f"sqlite:///{tmp_path / 'ce.db'}", experiment_name="ce-tests"
        )
        client = MlflowClient(tracking_uri=tracking_config.tracking_uri)
        client.create_experiment("ce-tests", artifact_location=(tmp_path / "artifacts").as_uri())
    config = replace(
        classifier_configuration, epochs=8, mlflow=tracking_config,
        tracked_metric="balanced_accuracy", tracked_mode=mode,
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
        scheduler=SchedulerConfiguration("ReduceLROnPlateau", {"mode": mode, "patience": 0, "factor": 0.5}),
        early_stopping={"patience": 2, "tolerance": 0.0, "comparison": "less" if mode == "min" else "greater"},
    )
    values = [0.8, 0.2, 0.2, 0.5] if mode == "min" else [0.2, 0.8, 0.8, 0.5]
    scores = iter(zip(values, [0.1, 0.2, 0.3, 0.4], strict=True))

    def validate(self, model, loader):
        """Supply conflicting validation metrics to expose hardcoded-loss selection."""
        metric, ce = next(scores)
        return {"balanced_accuracy": metric, "ce_loss_mean": ce}

    monkeypatch.setattr(training_api.ClassifierExperiment, "_validate", validate)
    source = Path(config.initialization["encoder"]["checkpoint"])
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_configuration = deepcopy(config.to_dict())
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    best = torch.load(result.best_checkpoint, weights_only=True)
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert best["epoch"] == result.best_epoch == 2 and last["epoch"] == 4
    assert result.best_value == values[1]
    assert result.tracked_metric == config.tracked_metric and result.tracked_mode == mode
    assert [row["validation"]["balanced_accuracy"] for row in last["history"]] == values
    assert [row["next_learning_rate"] for row in last["history"]] == [0.002, 0.002, 0.001, 0.0005]
    assert best["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.002
    assert best["scheduler_state_dict"]["best"] == values[1]
    assert last["configuration"]["tracked_metric"] == config.tracked_metric
    assert last["configuration"]["tracked_mode"] == mode
    assert {path.name for path in config.output_directory.glob("*.pt")} == {"best.pt", "last.pt"}
    assert yaml.safe_load((config.output_directory / "history.yaml").read_text()) == last["history"]
    assert yaml.safe_load((config.output_directory / "config.yaml").read_text()) == expected_configuration
    assert config.to_dict() == expected_configuration
    assert best["provenance"]["encoder_checkpoint"] == str(source.resolve())
    assert best["provenance"]["encoder_checkpoint_sha256"] == checksum
    card = yaml.safe_load((config.output_directory / "best.model_card.yaml").read_text())
    assert card["validation"]["epoch"] == card["provenance"]["epoch"] == 2
    assert card["validation"]["metrics"]["balanced_accuracy"] == values[1]
    assert card["classes"] == {0: "mouse", 1: "rat"}
    assert card["provenance"]["checkpoint_sha256"] == hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
    assert card["provenance"]["encoder_checkpoint_sha256"] == checksum
    assert (config.output_directory / "best.model_card.md").is_file()
    restored = YoloClassifierModel.from_checkpoint(result.best_checkpoint)
    assert restored.label_names == {"mouse": 0, "rat": 1}
    if tracking:
        run = client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED"
        assert run.data.metrics["best_epoch"] == 2
        assert run.data.metrics["best_validation_balanced_accuracy"] == values[1]
        metrics = client.get_metric_history(result.mlflow_run_id, "validation/balanced_accuracy")
        assert [metric.value for metric in metrics] == values
        assert [metric.step for metric in metrics] == [1, 2, 3, 4]
        import mlflow

        assert mlflow.active_run() is None
    else:
        assert result.mlflow_run_id is None


def test_training_without_validation_is_explicit_and_still_reports_a_result(training_api, classifier_configuration):
    """Standalone no-val training selects by training CE and does not pretend it was validated."""
    config = replace(classifier_configuration, validation_dataset=None)
    result = training_api.ClassifierExperiment(config).run_with_result()
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert result.validation_dataset is None
    assert result.best_value == min(row["mean"] for row in last["history"])
    card = yaml.safe_load((config.output_directory / "best.model_card.yaml").read_text())
    assert card["validation"]["status"] == "not_validated"


def test_checkpoint_write_failure_preserves_last_complete_epoch(
    training_api, classifier_configuration, initialized_models, monkeypatch
):
    """The CE trainer must reuse atomic retention rather than overwrite valid weights in place."""
    original_save = torch.save
    writes = 0

    def fail_second_epoch(snapshot, destination, *args, **kwargs):
        """Simulate quota exhaustion after writing some bytes of epoch two."""
        nonlocal writes
        if Path(destination).name == ".last.pt.tmp":
            writes += 1
            if writes == 2:
                Path(destination).write_bytes(b"partial checkpoint")
                raise OSError("simulated quota failure")
        return original_save(snapshot, destination, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_second_epoch)
    with pytest.raises(OSError, match="quota failure"):
        training_api.ClassifierExperiment(classifier_configuration).run()
    output = classifier_configuration.output_directory
    assert torch.load(output / "last.pt", weights_only=True)["epoch"] == 1
    assert torch.load(output / "best.pt", weights_only=True)["epoch"] == 1
    assert not list(output.glob(".*.tmp"))
    assert not initialized_models[0][0].training


def test_nonfinite_training_loss_is_rejected_before_optimizer_step(
    training_api, classifier_configuration, initialized_models, monkeypatch
):
    """A graph-connected NaN loss must not poison the initialized model or create a checkpoint."""
    monkeypatch.setattr(
        torch.nn.functional, "cross_entropy", lambda input, target, *args, **kwargs: input.sum() * float("nan")
    )
    with pytest.raises(FloatingPointError, match="loss|CE|ce"):
        training_api.ClassifierExperiment(classifier_configuration).run()
    model, before = initialized_models[0]
    assert not model.training
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
    assert not list(classifier_configuration.output_directory.glob("*.pt"))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_tracked_validation_metric_cannot_be_saved(
    training_api, classifier_configuration, monkeypatch, value
):
    """Valid training CE is not a substitute for an invalid configured validation objective."""
    monkeypatch.setattr(
        training_api.ClassifierExperiment, "_validate", lambda self, model, loader: {"ce_loss_mean": value}
    )
    with pytest.raises(FloatingPointError, match="tracked|validation"):
        training_api.ClassifierExperiment(classifier_configuration).run()
    assert not list(classifier_configuration.output_directory.glob("*.pt"))


def test_missing_tracked_metric_does_not_fall_back_to_ce(training_api, classifier_configuration, monkeypatch):
    """Missing configured classification metrics fail explicitly before checkpointing."""
    config = replace(
        classifier_configuration, tracked_metric="balanced_accuracy", tracked_mode="max",
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
    )
    monkeypatch.setattr(training_api.ClassifierExperiment, "_validate", lambda self, model, loader: {"ce_loss_mean": 0.1})
    with pytest.raises(ValueError, match="balanced_accuracy"):
        training_api.ClassifierExperiment(config).run()
    assert not list(config.output_directory.glob("*.pt"))


def test_empty_training_class_is_not_silently_ignored(training_api, classifier_configuration):
    """Requested species with no training images cannot silently disappear from CE training."""
    for path in (classifier_configuration.train_dataset / "rat").glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="empty|nonempty"):
        training_api.ClassifierExperiment(classifier_configuration).run()
    assert not list(classifier_configuration.output_directory.glob("*.pt"))


def test_configuration_round_trip_uses_own_initialization_nodes(training_api, classifier_configuration, tmp_path):
    """Classifier configuration stays independent of SupCon-only sampling/projection parameters."""
    document = classifier_configuration.to_dict()
    assert document["initialization"] == classifier_configuration.initialization
    assert "classes_per_batch" not in document and "temperature" not in document
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(document))
    assert training_api.ClassifierConfiguration.from_yaml(path) == classifier_configuration
    assert training_api.ClassifierConfiguration.from_mapping(document) == classifier_configuration


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"epochs": 0}, "epochs"), ({"epochs": True}, "epochs"),
        ({"batch_size": 0}, "batch_size"), ({"batch_size": True}, "batch_size"),
        ({"workers": -1}, "workers"), ({"workers": True}, "workers"),
        ({"optimizer": {"lr": float("nan")}}, "optimizer"),
        ({"optimizer": {"lr": 0}}, "optimizer"),
        ({"tracked_metric": "unknown"}, "tracked_metric"),
        ({"tracked_mode": "lowest"}, "tracked_mode"),
        ({"freeze_encoder": 1}, "freeze_encoder"),
        ({"encoder_batchnorm": "train"}, "encoder_batchnorm"),
        ({"class_names": ("mouse", "mouse")}, "class_names"),
    ],
)
def test_invalid_classifier_configuration_fails_early(classifier_configuration, updates, message):
    """Invalid settings fail before training or output creation."""
    with pytest.raises(ValueError, match=message):
        replace(classifier_configuration, **updates)
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
def test_output_cannot_be_inside_either_input_split(classifier_configuration, split):
    """Both training and validation directories remain read-only inputs."""
    with pytest.raises(ValueError, match="outside|output_directory"):
        replace(classifier_configuration, output_directory=getattr(classifier_configuration, split) / "output")


@pytest.mark.parametrize(
    "updates",
    [
        {"scheduler": SchedulerConfiguration("ReduceLROnPlateau", {"mode": "max"})},
        {"early_stopping": {"patience": 2, "tolerance": 0.0, "comparison": "greater"}},
    ],
)
def test_tracking_direction_must_agree_with_scheduler_and_stopping(classifier_configuration, updates):
    """One direction applies to the same configured metric throughout the stage."""
    with pytest.raises(ValueError, match="direction|tracked_mode|agree"):
        replace(classifier_configuration, **updates)


@pytest.mark.parametrize(
    "initialization,message",
    [
        ({"encoder": {"source": "auto", "checkpoint": "x.pt"}, "head": {"replace": "linear", "source": "random"}}, "source"),
        ({"encoder": {"source": "supcon"}, "head": {"replace": "linear", "source": "random"}}, "checkpoint"),
        ({"encoder": {"source": "supcon", "checkpoint": "x.pt"}, "head": {"replace": "linear", "source": "yolo_checkpoint"}}, "checkpoint"),
        ({"encoder": {"source": "supcon", "checkpoint": "x.pt"}, "head": {"replace": "other", "source": "random"}}, "replace"),
        ({"encoder": {"source": "supcon", "checkpoint": "x.pt"}, "head": {"replace": "linear", "source": "random", "typo": True}}, "head|unexpected|typo"),
    ],
)
def test_invalid_initialization_nodes_are_not_silently_defaulted(classifier_configuration, initialization, message):
    """Bad source/head choices fail before any model I/O or training."""
    with pytest.raises(ValueError, match=message):
        replace(classifier_configuration, initialization=initialization)


# The source fixture is read-only across examples; every example builds a fresh
# configuration mapping, never an experiment or dataset mutation.
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch_size=st.integers(1, 16), freeze=st.booleans(), batchnorm=st.sampled_from(["eval", "adapt"]))
def test_configuration_preserves_batch_and_freeze_options(
    training_api, classifier_configuration, batch_size, freeze, batchnorm
):
    """Round-trip parsing cannot silently reset the model or batching policy."""
    config = replace(classifier_configuration, batch_size=batch_size, freeze_encoder=freeze, encoder_batchnorm=batchnorm)
    assert training_api.ClassifierConfiguration.from_mapping(config.to_dict()) == config


@pytest.mark.parametrize(
    "updates",
    [
        {"scheduler": SchedulerConfiguration("ReduceLROnPlateau", {"mode": "min"})},
        {"early_stopping": {"patience": 2, "tolerance": 0.0, "comparison": "less"}},
        {"tracked_metric": "balanced_accuracy", "tracked_mode": "max", "validation_metrics": {"balanced_accuracy": {"function": "balanced_accuracy_score"}}},
    ],
)
def test_validation_dependent_options_cannot_fall_back_to_training_loss(classifier_configuration, updates):
    """No-val training is explicit, not a substitute for a requested validation objective."""
    with pytest.raises(ValueError, match="validation_dataset"):
        replace(classifier_configuration, validation_dataset=None, **updates)


@pytest.mark.parametrize("metric,mode", [("ce_loss_mean", "min"), ("balanced_accuracy", "max")])
def test_omitted_direction_uses_the_configured_metric_default(classifier_configuration, metric, mode):
    """Losses default to minimization and classification scores to maximization."""
    config = replace(
        classifier_configuration, tracked_metric=metric, tracked_mode=None,
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
    )
    assert config.effective_tracked_mode == mode


def test_noncolliding_source_can_share_the_output_directory(
    training_api, classifier_configuration, classifier_assets
):
    """Source safety does not forbid harmless source filenames in the run directory."""
    output = classifier_configuration.output_directory
    output.mkdir()
    source = output / "encoder-source.pt"
    source.write_bytes(classifier_assets.supcon.read_bytes())
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    initialization = deepcopy(classifier_configuration.initialization)
    initialization["encoder"]["checkpoint"] = str(source)
    config = replace(classifier_configuration, initialization=initialization, epochs=1)
    training_api.ClassifierExperiment(config).run()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize(
    "node,artifact,alias",
    [
        ("encoder", "last.pt", "direct"),
        ("encoder", "best.pt", "direct"),
        ("encoder", ".last.pt.tmp", "direct"),
        ("encoder", ".best.pt.tmp", "direct"),
        ("head", "best.pt", "direct"),
        ("encoder", "config.yaml", "artifact_symlink"),
        ("encoder", "history.yaml", "artifact_symlink"),
        ("encoder", "best.model_card.yaml", "artifact_symlink"),
        ("encoder", "best.model_card.md", "artifact_symlink"),
        ("encoder", "best.pt", "source_symlink"),
        ("encoder", "config.yaml", "hardlink"),
    ],
)
def test_output_artifacts_cannot_overwrite_initialization_sources(
    classifier_configuration, classifier_assets, node, artifact, alias
):
    """Reject direct and aliased source collisions before any output is written."""
    config = classifier_configuration
    output = config.output_directory
    output.mkdir()
    initialization = deepcopy(config.initialization)
    original = classifier_assets.supcon if node == "encoder" else classifier_assets.ultralytics
    destination = output / artifact
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
    if node == "head":
        initialization["head"] = {"replace": "linear", "source": "yolo_checkpoint", "checkpoint": str(source)}
    else:
        initialization["encoder"]["checkpoint"] = str(source)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    before = {path.name for path in output.iterdir()}
    with pytest.raises(ValueError, match="checkpoint|source|collision"):
        replace(config, initialization=initialization)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    assert {path.name for path in output.iterdir()} == before


@pytest.mark.parametrize("workers", [0, 1])
def test_classifier_cli_trains_a_portable_cpu_checkpoint(training_api, classifier_configuration, tmp_path, workers):
    """The actual thin script works with ordinary and multiprocessing DataLoaders."""
    config = replace(classifier_configuration, epochs=1, workers=workers)
    path = tmp_path / "cli.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_classifier.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True, timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Final checkpoint:" in completed.stdout
    assert torch.load(config.output_directory / "last.pt", weights_only=True)["epoch"] == 1
    restored = YoloClassifierModel.from_checkpoint(config.output_directory / "best.pt")
    assert restored.label_names == {"mouse": 0, "rat": 1}
