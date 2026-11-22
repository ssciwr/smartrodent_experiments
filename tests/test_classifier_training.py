"""Native Ultralytics head-training contracts on temporary YOLO models and data."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.nn.tasks import ClassificationModel

from smartrodent.classifier_model import YoloClassifierModel
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.supcon_model import YoloSupConModel
from test_classifier_model import _native_checkpoint


@pytest.fixture
def training_api():
    """Import the public experiment/configuration API without requiring future classes."""
    from smartrodent import classifier_training

    return classifier_training


@pytest.fixture
def classifier_assets(tmp_path):
    """Create tiny images and native, SupCon, and existing portable CE inputs."""
    torch.set_num_threads(2)
    for split, counts in [("train", (5, 5)), ("val", (3, 2))]:
        for label, (name, count) in enumerate(zip(("mouse", "rat"), counts, strict=True)):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (32, 32), (30 + 20 * index, 255 * label, 80)).save(
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
        _, embeddings = source(torch.rand(4, 3, 32, 32))
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
        # Existing portable CE files remain readable inputs, not the new output format.
        classifier = tmp_path / "classifier-source.pt"
        converted = YoloClassifierModel.initialize(
            {"encoder": {"source": "supcon", "checkpoint": supcon},
             "head": {"replace": "linear", "source": "random"}},
            label_names={"mouse": 0, "rat": 1},
        )
        torch.save(converted.checkpoint_state(), classifier)
    return SimpleNamespace(
        train=tmp_path / "train", val=tmp_path / "val",
        supcon=supcon, ultralytics=native, classifier=classifier, architecture=architecture,
    )


@pytest.fixture
def classifier_configuration(training_api, classifier_assets, tmp_path):
    """Keep shared data/batching settings while passing CE-specific options natively."""
    return training_api.ClassifierConfiguration(
        train_dataset=classifier_assets.train,
        validation_dataset=classifier_assets.val,
        output_directory=tmp_path / "ce-output",
        class_names=("rat", "mouse"),
        initialization={
            "encoder": {"source": "supcon", "checkpoint": str(classifier_assets.supcon)},
            "head": {"replace": "linear", "source": "random"},
        },
        encoder_batchnorm="eval",
        transforms={
            "image_size": 32,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
            "random_grayscale": {"p": 0.0},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
            "random_horizontal_flip": {"p": 0.0},
            "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
        },
        train_kwargs={
            "optimizer": "AdamW", "lr0": 0.002, "lrf": 1.0,
            "weight_decay": 0.0, "warmup_epochs": 0.0, "nbs": 1,
            "patience": 0, "amp": False, "plots": False,
        },
        epochs=3, batch_size=4, workers=0, device="cpu", seed=42,
        tracked_metric="ce_loss_mean", tracked_mode="min",
    )


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
            record.modes.append({name: module.training for name, module in trainer.model.named_modules()})

        def epoch(trainer):
            record.after = deepcopy(trainer.model.state_dict())

        self.add_callback("on_train_start", start)
        self.add_callback("on_train_batch_start", batch)
        self.add_callback("on_train_epoch_end", epoch)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationTrainer, "train", train)
    return observed


@pytest.fixture
def controlled_scores(monkeypatch):
    """Control only returned epoch scores after real native validation has completed."""
    observed = SimpleNamespace(score_sets={}, epochs={}, validations=[], fail_at=None)
    original = ClassificationValidator.__call__

    def validate(self, trainer=None, model=None):
        metrics = original(self, trainer=trainer, model=model)
        # Native final best-model evaluation is not another training epoch.
        if trainer is None:
            return metrics
        directory = str(trainer.save_dir)
        epoch = trainer.epoch + 1
        observed.epochs[directory] = epoch
        observed.validations.append((directory, epoch))
        rate = trainer.args.lr0
        if observed.fail_at == (rate, epoch):
            raise RuntimeError("deliberate later classifier failure")
        if rate in observed.score_sets:
            metrics["balanced_accuracy"] = observed.score_sets[rate][epoch - 1]
            # Deliberately disagree with the requested score and direction.
            metrics["ce_loss_mean"] = 0.1 * epoch
        return metrics

    monkeypatch.setattr(ClassificationValidator, "__call__", validate)
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
        result.extend((int(label), round(float(image[0, 0, 0]) * 255)) for label, image in zip(labels, images, strict=True))
    return result


def test_training_loader_is_single_view_shuffled_and_covers_source_once(
    training_api, classifier_configuration, native_runs
):
    """Native training consumes configured grayscale/resize; validation stays unaugmented."""
    for path in classifier_configuration.train_dataset.rglob("*.png"):
        with Image.open(path) as image:
            color = image.getpixel((0, 0))
        Image.new("RGB", (51, 19), color).save(path)
    config = replace(
        classifier_configuration, epochs=1,
        transforms={**classifier_configuration.transforms, "image_size": 64, "random_grayscale": {"p": 1.0}},
    )
    training_api.ClassifierExperiment(config).run()
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
    training_api, classifier_configuration, native_runs, batch_size
):
    """Native validation sees every requested image with the existing preprocessing."""
    config = replace(classifier_configuration, epochs=1, batch_size=batch_size)
    training_api.ClassifierExperiment(config).run()
    first = _signature(native_runs[0].trainer.test_loader)
    assert first == _signature(native_runs[0].trainer.test_loader)
    assert len(first) == len(set(first)) == 5
    assert Counter(label for label, _ in first) == {0: 3, 1: 2}
    assert {red for _, red in first} == {30, 50, 70}


def test_native_validation_computes_real_scores_and_unequal_tail_loss(
    training_api, classifier_configuration, classifier_assets, tmp_path, monkeypatch
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
        classifier_configuration, epochs=1, batch_size=2,
        train_kwargs={**classifier_configuration.train_kwargs, "lr0": 1e-6},
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
        initialization={**classifier_configuration.initialization,
                        "head": {"replace": "linear", "source": "yolo_checkpoint", "checkpoint": str(donor)}},
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
                weighted_mean = sum(loss.item() * len(target) for loss, target in zip(losses, targets, strict=True)) / len(labels)
            assert native_mean != pytest.approx(weighted_mean)
            score = balanced_accuracy_score(labels.numpy(), guessed.numpy())
            assert score == 0.5 and (labels == guessed).float().mean().item() == pytest.approx(0.4)
            expected.append((weighted_mean, score))
        return original(self, trainer=trainer, model=model)

    monkeypatch.setattr(ClassificationValidator, "__call__", validate)
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert len(expected) == 1
    loss, score = expected[0]
    assert result.best_value == pytest.approx(loss, abs=0.00005)
    card = yaml.safe_load(result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
    assert card["validation"]["metrics"]["ce_loss_mean"] == pytest.approx(loss, abs=0.00005)
    assert card["validation"]["metrics"]["balanced_accuracy"] == score
    assert config.tracked_metric_definition == {"function": "torch.nn.functional.cross_entropy", "reduction": "mean"}


@pytest.mark.parametrize("head_replace", ["linear", "full"])
@pytest.mark.parametrize("batchnorm", ["eval", "adapt"])
def test_real_ce_training_honors_freezing_and_starts_a_fresh_optimizer(
    training_api, classifier_configuration, native_runs, head_replace, batchnorm
):
    """Native setup cannot replace the SupCon state, unfreeze it, or resume its optimizer."""
    config = replace(
        classifier_configuration, epochs=2, encoder_batchnorm=batchnorm,
        initialization={**classifier_configuration.initialization, "head": {"replace": head_replace, "source": "random"}},
    )
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert isinstance(result, ExperimentResult) and len(native_runs) == 1
    record = native_runs[0]
    model = record.trainer.model
    assert isinstance(model, ClassificationModel)
    boundary = _head_prefix(model, head_replace)
    source = YoloSupConModel.from_checkpoint(config.initialization["encoder"]["checkpoint"])
    try:
        for name, tensor in source.full_model.model.state_dict().items():
            if not name.startswith(boundary):
                assert torch.equal(record.before[name], tensor), name
    finally:
        source.finalize()
    parameters = dict(model.named_parameters())
    assert all(parameter.requires_grad == name.startswith(boundary) for name, parameter in parameters.items())
    assert any(not torch.equal(record.before[name], record.after[name]) for name in parameters if name.startswith(boundary))
    for name in parameters:
        if not name.startswith(boundary):
            assert torch.equal(record.before[name], record.after[name]), name
    encoder_bn = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d) and not name.startswith(boundary)}
    assert encoder_bn and record.modes
    assert all(modes[name] == (batchnorm == "adapt") for modes in record.modes for name in encoder_bn)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and not name.startswith(boundary):
            assert all(not modes[name] for modes in record.modes)
    bn_changed = any(
        not torch.equal(record.before[name], record.after[name])
        for name in record.before
        if not name.startswith(boundary) and name.endswith(("running_mean", "running_var"))
    )
    assert bn_changed == (batchnorm == "adapt")
    if batchnorm == "eval":
        for name, tensor in record.before.items():
            if not name.startswith(boundary):
                assert torch.equal(tensor, record.after[name]), name
    assert isinstance(record.trainer.optimizer, torch.optim.AdamW)
    assert record.trainer.optimizer.state
    assert all(parameter.requires_grad for parameter in record.trainer.optimizer.state)
    assert all(int(state["step"]) < 99 for state in record.trainer.optimizer.state.values())
    assert result.last_checkpoint == config.output_directory / "weights" / "last.pt"
    assert result.best_checkpoint == config.output_directory / "weights" / "best.pt"
    assert (config.output_directory / "results.csv").is_file()
    restored = YOLO(str(result.best_checkpoint), task="classify").model
    assert isinstance(restored, ClassificationModel) and restored.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize("source_format", ["supcon", "ultralytics", "classifier"])
def test_trainer_accepts_existing_encoder_formats_and_writes_native_checkpoints(
    training_api, classifier_configuration, classifier_assets, source_format, native_runs
):
    """Serialization format selects an input reader; the training backend remains YOLO."""
    config = replace(
        classifier_configuration, epochs=1,
        initialization={"encoder": {"source": source_format, "checkpoint": str(getattr(classifier_assets, source_format))},
                        "head": {"replace": "linear", "source": "random"}},
    )
    path = training_api.ClassifierExperiment(config).run()
    assert len(native_runs) == 1 and path == config.output_directory / "weights" / "last.pt"
    assert YOLO(str(path), task="classify").model.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize("head_replace", ["linear", "full"])
def test_trainer_honors_checkpoint_initialized_head(
    training_api, classifier_configuration, classifier_assets, native_runs, head_replace
):
    """The native trainer sees the prepared donor component, not a second initialization."""
    config = replace(
        classifier_configuration, epochs=1,
        initialization={**classifier_configuration.initialization,
                        "head": {"replace": head_replace, "source": "yolo_checkpoint", "checkpoint": str(classifier_assets.ultralytics)}},
    )
    training_api.ClassifierExperiment(config).run()
    donor = YOLO(str(classifier_assets.ultralytics), task="classify").model
    prefix = _head_prefix(donor, head_replace)
    for name, tensor in donor.state_dict().items():
        if name.startswith(prefix):
            assert torch.equal(native_runs[0].before[name], tensor), name


@pytest.mark.parametrize("boundary", ["linear", "full"])
def test_native_finetuning_adapts_a_thousand_class_head_to_twenty_six_species(
    training_api, classifier_configuration, classifier_assets, native_runs, tmp_path, boundary
):
    """Exercise the failed production shape transition through real native CPU training."""
    donor = _native_checkpoint(
        classifier_assets.architecture, tmp_path / "imagenet-like.pt",
        names={index: f"imagenet_{index}" for index in range(1000)},
    )
    labels = [f"species_{index:02}" for index in range(26)]
    for split in ("train", "val"):
        for index, name in enumerate(labels):
            directory = tmp_path / "species-data" / split / name
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 32), (index * 8, 30, 90)).save(directory / "image.png")
    config = replace(
        classifier_configuration, epochs=1, batch_size=26, class_names=labels,
        train_dataset=tmp_path / "species-data/train", validation_dataset=tmp_path / "species-data/val",
        initialization={**classifier_configuration.initialization,
                        "head": {"replace": boundary, "source": "yolo_checkpoint", "checkpoint": str(donor.path)}},
    )
    checksum = hashlib.sha256(donor.path.read_bytes()).hexdigest()
    original_train = ClassificationTrainer.train
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def train(self, *args, **kwargs):
            if boundary == "linear":
                assert any("random" in str(w.message).lower() and "1000" in str(w.message) and "26" in str(w.message) for w in caught)
            return original_train(self, *args, **kwargs)

        # Check warning timing at entry to the real native training method.
        from unittest.mock import patch

        with patch.object(ClassificationTrainer, "train", train):
            result = training_api.ClassifierExperiment(config).run_with_result()
    assert len(native_runs) == 1
    restored = YOLO(str(result.best_checkpoint), task="classify").model
    assert restored.model[-1].linear.out_features == 26
    assert restored.names == dict(enumerate(labels))
    assert hashlib.sha256(donor.path.read_bytes()).hexdigest() == checksum
    if boundary == "linear":
        assert any("random" in str(w.message).lower() and "1000" in str(w.message) and "26" in str(w.message) for w in caught)


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("tracking", [False, True])
def test_configured_metric_drives_native_best_stopping_and_tracking(
    training_api, classifier_configuration, tmp_path, monkeypatch, controlled_scores, native_runs, mode, tracking
):
    """Configured fitness must beat native top-1 defaults and survive final evaluation."""
    tracking_config = None
    if tracking:
        from mlflow import MlflowClient

        tracking_config = MlflowConfiguration(tracking_uri=f"sqlite:///{tmp_path / 'ce.db'}", experiment_name="ce-tests")
        client = MlflowClient(tracking_uri=tracking_config.tracking_uri)
        client.create_experiment("ce-tests", artifact_location=(tmp_path / "artifacts").as_uri())
    # Force native autologging on in memory, without changing persisted Ultralytics settings.
    import importlib
    import mlflow

    utils = importlib.import_module("ultralytics.utils")
    callbacks = importlib.import_module("ultralytics.utils.callbacks.mlflow")
    monkeypatch.setattr(utils, "SETTINGS", {**utils.SETTINGS, "mlflow": True})
    monkeypatch.setattr(callbacks, "TESTS_RUNNING", False)
    autolog_calls = []
    monkeypatch.setattr(mlflow, "autolog", lambda *args, **kwargs: autolog_calls.append((args, kwargs)))
    original_train = ClassificationTrainer.train

    def train(self, *args, **kwargs):
        # Fail before any unsafe callback can touch a caller's tracking backend.
        assert not any(callback.__module__ == callbacks.__name__
                       for group in self.callbacks.values() for callback in group)
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(ClassificationTrainer, "train", train)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'unwanted.db'}")
    config = replace(
        classifier_configuration, epochs=8, mlflow=tracking_config,
        tracked_metric="balanced_accuracy", tracked_mode=mode,
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
        train_kwargs={**classifier_configuration.train_kwargs, "patience": 2},
    )
    values = [0.8, 0.2, 0.3, 0.4] if mode == "min" else [0.2, 0.8, 0.7, 0.6]
    controlled_scores.score_sets = {0.002: values}
    source = Path(config.initialization["encoder"]["checkpoint"])
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert result.best_epoch == 2 and result.best_value == values[1]
    assert len(controlled_scores.validations) == 4
    assert len(native_runs) == 1 and autolog_calls == []
    assert utils.SETTINGS["mlflow"] is True
    assert result.tracked_mode == mode and result.tracked_metric == "balanced_accuracy"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    assert not (tmp_path / "unwanted.db").exists()
    assert yaml.safe_load((config.output_directory / "config.yaml").read_text()) == config.to_dict()
    card = yaml.safe_load(result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
    assert card["validation"]["epoch"] == card["provenance"]["epoch"] == 2
    assert card["validation"]["metrics"]["balanced_accuracy"] == values[1]
    assert card["classes"] == {0: "mouse", 1: "rat"}
    assert card["provenance"]["checkpoint_sha256"] == hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
    assert card["provenance"]["encoder_checkpoint_sha256"] == checksum
    assert result.best_checkpoint.with_suffix(".model_card.md").is_file()
    assert YOLO(str(result.best_checkpoint), task="classify").model.names == {0: "mouse", 1: "rat"}
    if tracking:
        run = client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED" and run.data.metrics["best_epoch"] == 2
        assert run.data.metrics["best_validation_balanced_accuracy"] == values[1]
        assert [metric.value for metric in client.get_metric_history(result.mlflow_run_id, "validation/balanced_accuracy")] == values
        assert len(client.search_runs([run.info.experiment_id])) == 1
    else:
        assert result.mlflow_run_id is None
    import mlflow

    assert mlflow.active_run() is None


def test_empty_training_class_is_not_silently_ignored(training_api, classifier_configuration):
    """Requested species cannot silently disappear from native training."""
    for path in (classifier_configuration.train_dataset / "rat").glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="empty|nonempty"):
        training_api.ClassifierExperiment(classifier_configuration)
    assert not classifier_configuration.output_directory.exists()


def test_configuration_round_trip_uses_own_initialization_nodes(training_api, classifier_configuration, tmp_path):
    """Native CE options and custom input/metric settings remain independent owned values."""
    document = classifier_configuration.to_dict()
    assert document["initialization"] == classifier_configuration.initialization
    assert document["train_kwargs"] == classifier_configuration.train_kwargs
    assert not {"optimizer", "scheduler", "early_stopping", "freeze_encoder"} & document.keys()
    path = tmp_path / "classifier.yaml"
    path.write_text(yaml.safe_dump(document))
    assert training_api.ClassifierConfiguration.from_yaml(path) == classifier_configuration
    assert training_api.ClassifierConfiguration.from_mapping(document) == classifier_configuration


def test_training_without_validation_is_explicit_and_still_reports_a_result(
    training_api, classifier_configuration, native_runs
):
    """Native training-only execution selects training loss without inventing validation."""
    import csv

    config = replace(classifier_configuration, validation_dataset=None)
    result = training_api.ClassifierExperiment(config).run_with_result()
    assert len(native_runs) == 1 and result.validation_dataset is None
    with (config.output_directory / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    losses = [float(next(value for key, value in row.items() if key.strip() == "train/loss")) for row in rows]
    assert result.best_value == pytest.approx(min(losses), rel=0.0001)
    card = yaml.safe_load(result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
    assert card["validation"]["status"] == "not_validated"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_nonfinite_or_missing_tracked_metric_does_not_fall_back_to_native_fitness(
    training_api, classifier_configuration, monkeypatch, value
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
    config = replace(classifier_configuration, tracked_metric="balanced_accuracy", tracked_mode="max",
                     validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}})
    with pytest.raises((ValueError, FloatingPointError), match="balanced_accuracy|tracked|validation"):
        training_api.ClassifierExperiment(config).run()
    assert not list(config.output_directory.rglob("*.pt"))


def test_native_checkpoint_failure_propagates_and_releases_training_mode(
    training_api, classifier_configuration, native_runs, monkeypatch
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
        training_api.ClassifierExperiment(classifier_configuration).run()
    assert writes == 2
    model = native_runs[0].trainer.model
    assert all(not module.training for module in model.modules())
    last = classifier_configuration.output_directory / "weights" / "last.pt"
    assert torch.load(last, map_location="cpu", weights_only=False)["epoch"] == 0
    assert YOLO(str(last), task="classify").model.names == {0: "mouse", 1: "rat"}


@pytest.mark.parametrize("updates,message", [
    ({"epochs": 0}, "epochs"), ({"epochs": True}, "epochs"),
    ({"batch_size": 0}, "batch_size"), ({"batch_size": True}, "batch_size"),
    ({"workers": -1}, "workers"), ({"workers": True}, "workers"),
    ({"tracked_metric": "unknown"}, "tracked_metric"), ({"tracked_mode": "lowest"}, "tracked_mode"),
    ({"encoder_batchnorm": "train"}, "encoder_batchnorm"), ({"class_names": ("mouse", "mouse")}, "class_names"),
    ({"optimizer": {"lr": 0.1}}, "optimizer"), ({"freeze_encoder": False}, "freeze_encoder"),
    ({"scheduler": {"name": "ReduceLROnPlateau"}}, "scheduler"),
    ({"early_stopping": {"patience": 2}}, "early_stopping"),
])
def test_invalid_classifier_configuration_fails_early(classifier_configuration, updates, message):
    """Malformed or removed options are errors, not ignored compatibility aliases."""
    with pytest.raises((ValueError, TypeError), match=message):
        replace(classifier_configuration, **updates)
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("key,value", [
    ("model", "other.pt"), ("data", "other-data"), ("project", "other-output"),
    ("name", "other-run"), ("save_dir", "other-output"), ("exist_ok", False),
    ("freeze", 0), ("resume", True), ("pretrained", False), ("save", False),
    ("epochs", 999), ("batch", 999), ("imgsz", 999), ("workers", 10),
    ("device", "cuda:1"), ("seed", 99), ("val", False), ("classes", [0]), ("single_cls", True),
])
def test_native_kwargs_cannot_override_owned_model_data_outputs_or_policy(classifier_configuration, key, value):
    """One owner controls shared inputs, output layout, frozen boundary, and run state."""
    with pytest.raises(ValueError, match=key):
        replace(classifier_configuration, train_kwargs={**classifier_configuration.train_kwargs, key: value})
    assert not classifier_configuration.output_directory.exists()


def test_unknown_native_training_options_fail_before_execution(classifier_configuration):
    """Native argument validation must reject misspelled options without starting training."""
    with pytest.raises((ValueError, SyntaxError), match="not_a_native_option"):
        replace(classifier_configuration, train_kwargs={"not_a_native_option": True})
    assert not classifier_configuration.output_directory.exists()


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
def test_output_cannot_be_inside_either_input_split(classifier_configuration, split):
    with pytest.raises(ValueError, match="outside|output_directory"):
        replace(classifier_configuration, output_directory=getattr(classifier_configuration, split) / "output")


@pytest.mark.parametrize("initialization,message", [
    ({"encoder": {"source": "auto", "checkpoint": "x.pt"}, "head": {"replace": "linear", "source": "random"}}, "source"),
    ({"encoder": {"source": "supcon"}, "head": {"replace": "linear", "source": "random"}}, "checkpoint"),
    ({"encoder": {"source": "supcon", "checkpoint": "x.pt"}, "head": {"replace": "linear", "source": "yolo_checkpoint"}}, "checkpoint"),
    ({"encoder": {"source": "supcon", "checkpoint": "x.pt"}, "head": {"replace": "other", "source": "random"}}, "replace"),
])
def test_invalid_initialization_nodes_are_not_silently_defaulted(classifier_configuration, initialization, message):
    with pytest.raises(ValueError, match=message):
        replace(classifier_configuration, initialization=initialization)


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(batch_size=st.integers(1, 16), batchnorm=st.sampled_from(["eval", "adapt"]), lr=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False))
def test_configuration_preserves_batch_native_options_and_bn_policy(training_api, classifier_configuration, batch_size, batchnorm, lr):
    config = replace(classifier_configuration, batch_size=batch_size, encoder_batchnorm=batchnorm,
                     train_kwargs={**classifier_configuration.train_kwargs, "lr0": lr})
    assert training_api.ClassifierConfiguration.from_mapping(config.to_dict()) == config


@pytest.mark.parametrize("metric,mode", [("ce_loss_mean", "min"), ("balanced_accuracy", "max")])
def test_omitted_direction_uses_the_configured_metric_default(classifier_configuration, metric, mode):
    config = replace(classifier_configuration, tracked_metric=metric, tracked_mode=None,
                     validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}})
    assert config.effective_tracked_mode == mode


@pytest.mark.parametrize("node,artifact,alias", [
    ("encoder", "weights/last.pt", "direct"), ("encoder", "weights/best.pt", "direct"),
    ("head", "weights/best.pt", "direct"), ("encoder", "args.yaml", "artifact_symlink"),
    ("encoder", "config.yaml", "artifact_symlink"), ("encoder", "results.csv", "artifact_symlink"),
    ("encoder", "weights/best.model_card.yaml", "artifact_symlink"),
    ("encoder", "weights/best.model_card.md", "artifact_symlink"),
    ("encoder", "weights/best.pt", "source_symlink"), ("encoder", "args.yaml", "hardlink"),
])
def test_output_artifacts_cannot_overwrite_initialization_sources(classifier_configuration, classifier_assets, node, artifact, alias):
    """Native output names must retain direct/symlink/hardlink input protection."""
    config = classifier_configuration
    output = config.output_directory
    destination = output / artifact
    destination.parent.mkdir(parents=True, exist_ok=True)
    initialization = deepcopy(config.initialization)
    original = classifier_assets.supcon if node == "encoder" else classifier_assets.ultralytics
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
    initialization[node].update(source="yolo_checkpoint" if node == "head" else "supcon", checkpoint=str(source))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    with pytest.raises(ValueError, match="checkpoint|source|collision|collid"):
        replace(config, initialization=initialization)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    assert sorted(str(path.relative_to(output)) for path in output.rglob("*")) == before


@pytest.mark.parametrize("workers", [0, 1])
def test_classifier_cli_trains_a_native_cpu_checkpoint_without_touching_inputs(training_api, classifier_configuration, tmp_path, workers):
    """The actual CLI uses native training and cannot create caches inside input splits."""
    config = replace(classifier_configuration, epochs=1, workers=workers)
    path = tmp_path / "cli.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.rglob("*") if path.is_file() and not path.is_relative_to(config.output_directory)}
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_classifier.py"
    completed = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Final checkpoint:" in completed.stdout
    assert YOLO(str(config.output_directory / "weights" / "best.pt"), task="classify").model.names == {0: "mouse", 1: "rat"}
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in tmp_path.rglob("*") if path.is_file() and not path.is_relative_to(config.output_directory)}
    assert after == inputs
