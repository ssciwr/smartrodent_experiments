"""Exercise the standalone experiment, library modules, and portable checkpoints."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

import numpy as np
from pydantic import ValidationError
import pytest
from hypothesis import given
from hypothesis import strategies as st
import torch
import yaml
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from smartrodent import supcon_training as train_supcon
from smartrodent.supcon_training import (
    SchedulerConfiguration,
    SupConExperiment,
    TrainingConfiguration,
)
from smartrodent.datasets import (
    ImageDataset,
    MultiViewImageDataset,
    collate_views,
)
from smartrodent.supcon_model import YoloSupConModel


@pytest.fixture
def experiment_configuration(tmp_path):
    """Create temporary images and a tiny Ultralytics architecture without downloads."""
    dataset = tmp_path / "images"
    for name, color in (("mouse", "red"), ("rat", "blue")):
        directory = dataset / name
        directory.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (16, 16), color).save(directory / f"{index}.png")
        (directory / "notes.txt").write_text("Ignore non-image files")
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    return TrainingConfiguration(
        train_dataset=dataset,
        output_directory=tmp_path / "outputs",
        class_names=("rat", "mouse"),
        model_name=str(architecture),
        epochs=2,
        classes_per_batch=2,
        samples_per_class=2,
        samples_per_epoch=4,
        workers=0,
        device="cpu",
        output_features=4,
        transforms={
            "image_size": 16,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.2},
            "random_grayscale": {"p": 0.2},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.2},
            "random_horizontal_flip": {"p": 0.2},
            "random_resized_crop": {"scale": [0.75, 1.0], "ratio": [0.75, 1.33]},
        },
        optimizer={"lr": 0.001},
    )


def test_library_datasets_pickle_and_multiview_batching(experiment_configuration):
    """Imported datasets preserve image labels through pickling and collation."""
    config = experiment_configuration
    transform = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])
    single = ImageDataset(config.train_dataset, config.class_names, transform)
    assert len(single) == 6 and single[0][0] == 0
    dataset = MultiViewImageDataset(config.train_dataset, config.class_names, transform)
    restored = pickle.loads(pickle.dumps(dataset))
    labels, images = collate_views([restored[0], restored[3]])
    assert labels.tolist() == [0, 1, 0, 1]
    assert images.shape == (4, 3, 16, 16)
    assert 0 <= images.min() <= images.max() <= 1


def test_wrapper_propagates_gradients_and_respects_modes(experiment_configuration):
    """SupCon updates the encoder and projector, leaving the classifier outside the loss."""
    torch.set_num_threads(2)
    model = YoloSupConModel(
        experiment_configuration.model_name, num_classes=2, output_features=4
    )
    model.requires_grad_(True)
    model.train()
    assert (
        model.training and model.full_model.model.training and model.projector.training
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    encoder = next(model.full_model.model.model[0].parameters())
    before = encoder.detach().clone()
    logits, embeddings = model(torch.rand(4, 3, 16, 16))
    assert logits.shape == (4, 2) and embeddings.shape == (4, 4)
    np.testing.assert_allclose(embeddings.detach().norm(dim=1).numpy(), 1, atol=1e-5)
    SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
    assert encoder.grad is not None and model.projector[-1].weight.grad is not None
    assert model.head.linear.weight.grad is None
    optimizer.step()
    assert not torch.equal(before, encoder)
    model.eval()
    assert (
        not model.training
        and not model.full_model.model.training
        and not model.projector.training
    )
    model.finalize()
    assert model.embeddings is None


def test_experiment_saves_reusable_checkpoints(experiment_configuration):
    """A fresh session can restore architecture, class mapping, and exact outputs."""
    torch.set_num_threads(2)
    config = experiment_configuration
    result = SupConExperiment(config).run()
    assert result.name == "last.pt"
    assert {path.name for path in config.output_directory.glob("*.pt")} == {
        "best.pt",
        "last.pt",
    }
    checkpoint = torch.load(result, weights_only=True)
    assert checkpoint["epoch"] == 2 and len(checkpoint["history"]) == 2
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["scheduler_state_dict"] is None
    assert all(
        row["learning_rate"] == config.optimizer["lr"] for row in checkpoint["history"]
    )
    assert all(
        row["next_learning_rate"] == config.optimizer["lr"]
        for row in checkpoint["history"]
    )
    assert checkpoint["label_names"] == {"mouse": 0, "rat": 1}
    assert all(np.isfinite(row["std"]) for row in checkpoint["history"])
    history = yaml.safe_load((config.output_directory / "history.yaml").read_text())
    assert len(history) == 2 and history[-1]["epoch"] == 2
    # Rebuilding must use the embedded architecture, not reopen original weights/YAML.
    Path(config.model_name).unlink()
    first = YoloSupConModel.from_checkpoint(result)
    legacy = deepcopy(checkpoint)
    del legacy["scheduler_state_dict"]
    del legacy["configuration"]["scheduler"]
    legacy_path = config.output_directory / "legacy.pt"
    torch.save(legacy, legacy_path)
    second = YoloSupConModel.from_checkpoint(legacy_path)
    assert first.label_names == checkpoint["label_names"]
    assert first.training_configuration["transforms"]["image_size"] == 16
    cards = list(config.output_directory.glob("*.model_card.yaml"))
    assert len(cards) == 1
    card = yaml.safe_load(cards[0].read_text())
    assert (
        card["validation"]["epoch"]
        == min(history, key=lambda row: row["mean"])["epoch"]
    )
    assert card["validation"]["status"] == "not_validated"
    assert card["provenance"]["epoch"] == card["validation"]["epoch"]
    assert card["provenance"]["checkpoint"] == "best.pt"
    assert len(card["provenance"]["checkpoint_sha256"]) == 64
    assert card["provenance"]["git_commit"]
    assert card["provenance"]["run_id"] is None
    assert card["provenance"]["model_id"] is None
    assert not first.training and not first.projector.training
    for name, parameter in first.state_dict().items():
        assert torch.equal(parameter, checkpoint["model_state_dict"][name])
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        first_logits, first_features = first(images)
        second_logits, second_features = second(images)
    torch.testing.assert_close(first_logits[0], second_logits[0])
    torch.testing.assert_close(first_features, second_features)
    first.finalize()
    second.finalize()


def test_freezing_is_applied_after_gradient_enable(
    experiment_configuration, monkeypatch
):
    """Requested frozen layers stay fixed while the projection branch learns."""
    torch.set_num_threads(2)
    created = []
    original = train_supcon.YoloSupConModel

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(
            (model, next(model.full_model.model.model[0].parameters()).detach().clone())
        )
        return model

    monkeypatch.setattr(train_supcon, "YoloSupConModel", create_model)
    config = replace(experiment_configuration, epochs=1, freeze=(0,))
    checkpoint = SupConExperiment(config).run()
    model, before = created[0]
    parameter = next(model.full_model.model.model[0].parameters())
    assert not parameter.requires_grad and torch.equal(parameter, before)
    assert torch.load(checkpoint, weights_only=True)["model_configuration"][
        "freeze"
    ] == [0]
    assert model.embeddings is None and not model.training


def test_experiment_finalizes_after_loss_failure(experiment_configuration, monkeypatch):
    """Exceptions propagate while the training feature hook and modes are finalized."""
    torch.set_num_threads(2)
    created = []
    original = train_supcon.YoloSupConModel

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(model)
        return model

    class FailingLoss(torch.nn.Module):
        """Simulate a failure after encoder and projector forward execution."""

        def forward(self, embeddings, labels):
            """Raise an explicit training error."""
            raise RuntimeError("deliberate loss failure")

    monkeypatch.setattr(train_supcon, "YoloSupConModel", create_model)
    monkeypatch.setattr(train_supcon, "SupConLoss", lambda **kwargs: FailingLoss())
    with pytest.raises(RuntimeError, match="deliberate"):
        SupConExperiment(experiment_configuration).run()
    assert created[0].embeddings is None and not created[0].training


def test_experiment_rejects_nonfinite_epoch_loss(experiment_configuration, monkeypatch):
    """A nonfinite objective cannot be recorded as a successful checkpoint."""
    torch.set_num_threads(2)

    class NonfiniteLoss(torch.nn.Module):
        """Return a graph-connected NaN loss."""

        def forward(self, embeddings, labels):
            """Keep a gradient path while deliberately making the value invalid."""
            return embeddings.sum() * float("nan")

    monkeypatch.setattr(train_supcon, "SupConLoss", lambda **kwargs: NonfiniteLoss())
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        SupConExperiment(experiment_configuration).run()
    assert not list(experiment_configuration.output_directory.glob("*.pt"))


def test_interrupted_checkpoint_preserves_last_complete_state(
    experiment_configuration, monkeypatch
):
    """A failed replacement leaves valid last and best checkpoints from before it."""
    torch.set_num_threads(2)
    original_save = torch.save
    latest_save_count = 0

    def interrupt_second_latest_save(value, path, *args, **kwargs):
        """Leave a partial temporary file and simulate an interrupted filesystem write."""
        nonlocal latest_save_count
        if Path(path).name == ".last.pt.tmp":
            latest_save_count += 1
            if latest_save_count == 2:
                Path(path).write_bytes(b"partial checkpoint")
                raise OSError("simulated checkpoint write failure")
        return original_save(value, path, *args, **kwargs)

    monkeypatch.setattr(train_supcon.torch, "save", interrupt_second_latest_save)
    with pytest.raises(OSError, match="simulated checkpoint"):
        SupConExperiment(experiment_configuration).run()

    output = experiment_configuration.output_directory
    assert torch.load(output / "last.pt", weights_only=True)["epoch"] == 1
    assert torch.load(output / "best.pt", weights_only=True)["epoch"] == 1
    assert not list(output.glob(".*.tmp"))


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"epochs": 0}, "epochs"),
        ({"epochs": True}, "epochs"),
        ({"workers": -1}, "workers"),
        ({"class_names": "mouse"}, "class_names"),
        ({"class_names": ("", "rat")}, "class_names"),
        ({"class_names": ("rat", "rat")}, "unique"),
        ({"classes_per_batch": 1}, "classes_per_batch"),
        ({"classes_per_batch": 3}, "between 2"),
        ({"samples_per_epoch": 1}, "full batch"),
        ({"optimizer": {"lr": float("nan")}}, "optimizer.lr"),
        ({"temperature": 0}, "temperature"),
        ({"dropout": 2}, "dropout"),
        ({"freeze": (-1,)}, "freeze"),
    ],
)
def test_invalid_training_configuration_fails(
    experiment_configuration, updates, message
):
    """Invalid experiment settings fail before image loading and training."""
    with pytest.raises(ValueError, match=message):
        replace(experiment_configuration, **updates)


def test_output_directory_cannot_be_inside_input(experiment_configuration):
    """The experiment never writes checkpoints into its input images."""
    with pytest.raises(ValueError, match="outside"):
        replace(
            experiment_configuration,
            output_directory=experiment_configuration.train_dataset / "outputs",
        )


def test_missing_classes_fail_before_training(experiment_configuration):
    """P-K batches cannot silently omit an empty represented class."""
    empty = experiment_configuration.train_dataset / "rat"
    for path in empty.glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="nonempty classes"):
        SupConExperiment(experiment_configuration).run()


@pytest.mark.parametrize(
    "document",
    ["[]", "class_names: mouse\ntrain_dataset: images\noutput_directory: outputs\n"],
)
def test_yaml_requires_mapping_and_class_list(tmp_path, document):
    """Malformed YAML structure fails without model construction."""
    path = tmp_path / "config.yaml"
    path.write_text(document)
    with pytest.raises(ValueError):
        TrainingConfiguration.from_yaml(path)


def test_invalid_checkpoint_metadata_fails(tmp_path):
    """Old or unrelated tensor files are rejected with a format explanation."""
    path = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="train_supcon.py"):
        YoloSupConModel.from_checkpoint(path)


def test_script_cli_trains_without_notebook_imports(experiment_configuration, tmp_path):
    """The actual script can run a CPU epoch and save a loadable checkpoint."""
    configuration = replace(experiment_configuration, epochs=1)
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Final checkpoint:" in result.stdout
    assert (configuration.output_directory / "last.pt").is_file()
    assert (configuration.output_directory / "best.pt").is_file()


def test_script_cli_supports_multiprocessing_workers(
    experiment_configuration, tmp_path
):
    """Imported datasets and collation work in fresh worker processes."""
    configuration = replace(experiment_configuration, epochs=1, workers=1)
    config_path = tmp_path / "workers.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (configuration.output_directory / "last.pt").is_file()
    assert (configuration.output_directory / "best.pt").is_file()


def test_early_stopping_ignores_dips_that_never_beat_the_best():
    """Noisy values that only beat the previous value do not reset patience."""
    stopper = train_supcon.EarlyStopping(patience=3, tolerance=0.0, comparison="less")
    decisions = [stopper(value) for value in (1.0, 1.2, 1.1, 1.05)]
    assert decisions == [False, False, False, True]


def test_early_stopping_greater_requires_beating_best_by_tolerance():
    """In greater mode, values within the tolerance above the best do not improve."""
    stopper = train_supcon.EarlyStopping(
        patience=2, tolerance=0.1, comparison="greater"
    )
    assert not stopper(0.5)
    assert not stopper(0.65)  # improves: 0.65 > 0.5 + 0.1
    assert not stopper(0.7)  # within tolerance of the best
    assert stopper(0.6)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_scheduler_uses_absolute_improvement_threshold(mode):
    """Small improvements accumulate against the best accepted metric, in either mode."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = SchedulerConfiguration(
        "ReduceLROnPlateau",
        {
            "mode": mode,
            "patience": 1,
            "factor": 0.5,
            "threshold": 0.1,
            "threshold_mode": "abs",
        },
    ).build(optimizer)
    for value in (1.0, 0.95, 0.94):
        scheduler.step(value if mode == "min" else 2 - value)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    # This beats the accepted baseline, but is within tolerance of the last value.
    for value in (0.85, 0.84):
        scheduler.step(value if mode == "min" else 2 - value)
    assert optimizer.param_groups[0]["lr"] == 0.0005


@given(
    values=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=1, max_size=40
    ),
    patience=st.integers(min_value=1, max_value=5),
    tolerance=st.floats(min_value=0, max_value=1, allow_nan=False),
    comparison=st.sampled_from(["less", "greater"]),
)
def test_early_stopping_stops_after_patience_values_without_new_best(
    values, patience, tolerance, comparison
):
    """Stopping happens exactly when `patience` values in a row miss the best."""
    stopper = train_supcon.EarlyStopping(patience, tolerance, comparison)
    sign = 1 if comparison == "less" else -1
    best = values[0]
    misses = 0
    assert not stopper(values[0])
    for value in values[1:]:
        if sign * value < sign * best - tolerance:
            best, misses = value, 0
        else:
            misses += 1
        assert stopper(value) == (misses == patience)
        if misses == patience:
            break


@pytest.mark.parametrize(
    "document,location",
    [
        ([], ()),
        ({"name": "ReduceLROnPlateau"}, ("parameters",)),
        ({"name": "StepLR", "parameters": {}}, ("name",)),
        ({"name": "ReduceLROnPlateau", "parameters": []}, ("parameters",)),
        (
            {"name": "ReduceLROnPlateau", "parameters": {"unknown": 1}},
            ("parameters", "unknown"),
        ),
    ],
)
def test_invalid_scheduler_mapping_fails(document, location):
    """Malformed scheduler settings identify the rejected section before training."""
    with pytest.raises(ValidationError) as error:
        SchedulerConfiguration.from_mapping(document)
    assert location in [item["loc"] for item in error.value.errors()]


@pytest.mark.parametrize(
    "parameters",
    [
        {"mode": "less"},
        {"threshold_mode": "other"},
        {"factor": 0},
        {"factor": 1},
        {"factor": float("nan")},
        {"factor": True},
        {"factor": "0.5"},
        {"patience": -1},
        {"patience": 1.5},
        {"patience": True},
        {"cooldown": -1},
        {"cooldown": False},
        {"threshold": -1},
        {"threshold": float("inf")},
        {"min_lr": -1},
        {"min_lr": [0.001]},
        {"eps": -1},
    ],
)
def test_invalid_scheduler_parameters_fail(parameters):
    """Native numeric parameters have explicit finite-value and range checks."""
    with pytest.raises(ValidationError) as error:
        SchedulerConfiguration("ReduceLROnPlateau", parameters)
    assert any(next(iter(parameters)) in item["loc"] for item in error.value.errors())


def test_scheduler_training_requirements(experiment_configuration):
    """Plateau scheduling requires validation, a sensible floor, and matching modes."""
    scheduler = SchedulerConfiguration("ReduceLROnPlateau", {"mode": "min"})
    with pytest.raises(ValueError, match="validation_dataset"):
        replace(experiment_configuration, scheduler=scheduler)
    with pytest.raises(ValidationError, match="scheduler"):
        replace(
            experiment_configuration,
            validation_dataset=experiment_configuration.train_dataset,
            scheduler={},
        )
    with pytest.raises(ValueError, match="min_lr"):
        replace(
            experiment_configuration,
            validation_dataset=experiment_configuration.train_dataset,
            scheduler=SchedulerConfiguration("ReduceLROnPlateau", {"min_lr": 1.0}),
        )
    with pytest.raises(ValueError, match="must agree"):
        replace(
            experiment_configuration,
            validation_dataset=experiment_configuration.train_dataset,
            scheduler=scheduler,
            early_stopping={"patience": 5, "tolerance": 0.001, "comparison": "greater"},
        )


def test_scheduler_configuration_round_trip_and_sweep(
    experiment_configuration, tmp_path
):
    """Expanded scheduler parameters survive YAML, checkpoint, and MLflow serialization."""
    config = replace(
        experiment_configuration,
        validation_dataset=experiment_configuration.train_dataset,
        scheduler=SchedulerConfiguration(
            "ReduceLROnPlateau", {"factor": 0.5, "patience": 2}
        ),
    )
    document = config.to_dict()
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(document))
    assert TrainingConfiguration.from_yaml(path) == config
    assert config.mlflow_parameters()["scheduler_name"] == "ReduceLROnPlateau"
    assert config.mlflow_parameters()["scheduler_factor"] == 0.5
    assert config.mlflow_parameters()["scheduler_patience"] == 2
    constant = experiment_configuration.to_dict()
    del constant["scheduler"]
    assert TrainingConfiguration.from_mapping(constant).scheduler is None
    assert SchedulerConfiguration.from_mapping(None) is None
    path.write_text(
        path.read_text().replace(
            "\n    factor: 0.5\n", "\n    factor: !sweep\n      values: [0.5, 0.1]\n"
        )
    )
    runs = train_supcon.load_training_configurations(path)
    assert [run.scheduler.parameters["factor"] for run in runs] == [0.5, 0.1]


@given(
    patience=st.integers(min_value=0, max_value=5),
    mode=st.sampled_from(["min", "max"]),
    factor=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
)
def test_scheduler_plateaus_improvements_and_floor(patience, mode, factor):
    """Both modes reduce after the allowed misses, reset on improvement, and honor the floor."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = SchedulerConfiguration(
        "ReduceLROnPlateau",
        {
            "mode": mode,
            "patience": patience,
            "factor": factor,
            "min_lr": 0.00001,
            "threshold": 0.0,
            "threshold_mode": "abs",
        },
    ).build(optimizer)
    scheduler.step(1.0)
    for _ in range(patience):
        scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] == 0.001
    scheduler.step(0.5 if mode == "min" else 1.5)
    for _ in range(patience):
        scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] == 0.001
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001 * factor)
    for _ in range(300):
        previous_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(1.0)
        assert 0.00001 <= optimizer.param_groups[0]["lr"] <= previous_rate
    # PyTorch ignores changes at or below eps, including a tiny final gap to min_lr.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00001, abs=1e-8)


def test_scheduler_honors_minimum_change_epsilon():
    """An explicitly configured eps can suppress a reduction that is too small."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = SchedulerConfiguration(
        "ReduceLROnPlateau", {"patience": 0, "factor": 0.5, "eps": 0.00075}
    ).build(optimizer)
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == 0.001


@pytest.mark.parametrize(
    "mode,metric,comparison",
    [("min", "supcon_loss_mean", "less"), ("max", "accuracy", "greater")],
)
@pytest.mark.parametrize("tracking", [False, True])
def test_scheduler_integrates_with_early_stopping_and_checkpoints(
    experiment_configuration,
    monkeypatch,
    capsys,
    mode,
    metric,
    comparison,
    tracking,
):
    """Real CPU epochs use the reduced LR before early stopping and save coherent state."""
    torch.set_num_threads(2)
    config = replace(
        experiment_configuration,
        epochs=10,
        validation_dataset=experiment_configuration.train_dataset,
        tracked_metric=metric,
        validation_metrics={"accuracy": {"function": "accuracy_score"}},
        scheduler=SchedulerConfiguration(
            "ReduceLROnPlateau",
            {
                "mode": mode,
                "patience": 2,
                "factor": 0.5,
                "threshold": 0.001,
                "threshold_mode": "abs",
                "min_lr": 0.000001,
            },
        ),
        early_stopping={"patience": 5, "tolerance": 0.001, "comparison": comparison},
        mlflow=(
            train_supcon.MlflowConfiguration(
                tracking_uri=f"sqlite:///{experiment_configuration.output_directory.parent / 'scheduler.db'}",
                experiment_name="scheduler-test",
            )
            if tracking
            else None
        ),
    )
    values = iter((1.0, 1.2, 1.1, 1.05, 1.1, 1.2))

    def validate(self, model, loader, criterion):
        """Supply noisy validation scores that never beat the first epoch."""
        value = next(values)
        return {metric: value if mode == "min" else 2 - value}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    result = SupConExperiment(config).run()
    assert result.name == "last.pt"
    assert {path.name for path in config.output_directory.glob("*.pt")} == {
        "best.pt",
        "last.pt",
    }
    history = yaml.safe_load((config.output_directory / "history.yaml").read_text())
    assert [row["learning_rate"] for row in history] == [0.001] * 4 + [0.0005] * 2
    assert [row["next_learning_rate"] for row in history] == [0.001] * 3 + [0.0005] * 3
    checkpoint = torch.load(result, weights_only=True)
    assert checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.0005
    restored = YoloSupConModel.from_checkpoint(result)
    restored.requires_grad_(True)
    optimizer = torch.optim.AdamW(restored.parameters(), **config.optimizer)
    scheduler = config.scheduler.build(optimizer)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scheduler.step(1.2 if mode == "min" else 0.8)
    assert optimizer.param_groups[0]["lr"] == 0.00025
    card = yaml.safe_load(
        (config.output_directory / "best.model_card.yaml").read_text()
    )
    assert card["validation"]["epoch"] == 1
    assert "Learning rate reduced" in capsys.readouterr().out
    restored.finalize()
    if tracking:
        client = train_supcon.MlflowClient(tracking_uri=config.mlflow.tracking_uri)
        experiment = client.get_experiment_by_name("scheduler-test")
        run = client.search_runs([experiment.experiment_id])[0]
        assert run.data.params["scheduler_name"] == "ReduceLROnPlateau"
        for metric_name in ("learning_rate", "next_learning_rate"):
            metrics = client.get_metric_history(run.info.run_id, metric_name)
            assert [item.step for item in metrics] == list(range(1, 7))
            assert [item.value for item in metrics] == [
                row[metric_name] for row in history
            ]


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=2, allow_nan=False), min_size=1, max_size=30
    )
)
def test_scheduler_state_restoration_preserves_future_rates(values):
    """Restoring scheduler and optimizer state preserves subsequent LR decisions."""
    config = SchedulerConfiguration(
        "ReduceLROnPlateau", {"patience": 1, "factor": 0.5, "min_lr": 0.000001}
    )
    original_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))], lr=0.001
    )
    original = config.build(original_optimizer)
    for value in values:
        original.step(value)
    restored_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))], lr=0.001
    )
    restored = config.build(restored_optimizer)
    restored_optimizer.load_state_dict(deepcopy(original_optimizer.state_dict()))
    restored.load_state_dict(deepcopy(original.state_dict()))
    for value in values:
        original.step(value)
        restored.step(value)
        assert (
            restored_optimizer.param_groups[0]["lr"]
            == original_optimizer.param_groups[0]["lr"]
        )


@pytest.fixture
def validation_configuration(experiment_configuration, tmp_path):
    """Create identifiable validation images only inside the temporary test fixture."""
    validation = tmp_path / "validation-images"
    for label, name in enumerate(sorted(experiment_configuration.class_names)):
        directory = validation / name
        directory.mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (16, 16), (20 + 30 * index, 255 * label, 80)).save(
                directory / f"{index}.png"
            )
    return replace(experiment_configuration, validation_dataset=validation)


@pytest.fixture
def isolated_validation_rng():
    """Restore global RNG state after simulating arbitrary training work."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def test_run_with_result_reports_training_only_best_without_changing_legacy_run(
    experiment_configuration,
):
    """The additive result API describes real snapshots; run() still returns last.pt."""
    from smartrodent import experiment_results

    torch.set_num_threads(2)
    config = experiment_configuration
    result = SupConExperiment(config).run_with_result()
    assert isinstance(result, experiment_results.ExperimentResult)
    assert result.output_directory == config.output_directory
    assert result.last_checkpoint == config.output_directory / "last.pt"
    assert result.best_checkpoint == config.output_directory / "best.pt"
    best = torch.load(result.best_checkpoint, weights_only=True)
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert last["epoch"] == config.epochs
    assert result.best_epoch == best["epoch"]
    assert result.tracked_metric == "supcon_loss_mean"
    assert result.tracked_mode == "min"
    assert result.best_value == min(row["mean"] for row in last["history"])
    assert result.validation_dataset is None
    assert result.mlflow_run_id is None

    legacy_config = replace(config, output_directory=config.output_directory / "legacy")
    legacy_path = SupConExperiment(legacy_config).run()
    assert legacy_path == legacy_config.output_directory / "last.pt"
    assert torch.load(legacy_path, weights_only=True)["epoch"] == config.epochs


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("tracking", [False, True])
def test_configured_metric_controls_checkpoint_result_scheduler_and_stopping(
    validation_configuration, monkeypatch, tmp_path, mode, tracking
):
    """Honor the configured metric even when validation SupCon loss prefers another epoch."""
    torch.set_num_threads(2)
    tracking_config = None
    if tracking:
        tracking_config = train_supcon.MlflowConfiguration(
            tracking_uri=f"sqlite:///{tmp_path / 'tracked.db'}",
            experiment_name="tracked-tests",
        )
        client = train_supcon.MlflowClient(tracking_uri=tracking_config.tracking_uri)
        client.create_experiment(
            tracking_config.experiment_name,
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
    config = replace(
        validation_configuration,
        epochs=8,
        mlflow=tracking_config,
        tracked_metric="balanced_accuracy",
        tracked_mode=mode,
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        scheduler=SchedulerConfiguration(
            "ReduceLROnPlateau", {"mode": mode, "patience": 0, "factor": 0.5}
        ),
        early_stopping={
            "patience": 2,
            "tolerance": 0.0,
            "comparison": "less" if mode == "min" else "greater",
        },
    )
    tracked_values = [0.8, 0.2, 0.2, 0.5] if mode == "min" else [0.2, 0.8, 0.8, 0.5]
    supcon_values = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]  # Loss prefers epoch 1, NOT the configured winner.
    scores = iter(zip(tracked_values, supcon_values, strict=True))

    def validate(self, model, loader, criterion):
        """Supply conflicting metrics so accidentally hardcoding loss is observable."""
        tracked, supcon = next(scores)
        return {"balanced_accuracy": tracked, "supcon_loss_mean": supcon}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    result = SupConExperiment(config).run_with_result()
    last = torch.load(result.last_checkpoint, weights_only=True)
    best = torch.load(result.best_checkpoint, weights_only=True)
    history = last["history"]
    assert last["epoch"] == 4  # Two misses after the configured metric's epoch-2 best.
    assert best["epoch"] == result.best_epoch == 2  # First exact tie wins.
    assert result.tracked_metric == config.tracked_metric
    assert result.tracked_mode == mode
    assert result.best_value == tracked_values[1]  # Original value, not a signed score.
    assert result.validation_dataset == config.validation_dataset
    assert [row["validation"]["balanced_accuracy"] for row in history] == tracked_values
    assert [row["next_learning_rate"] for row in history] == [
        0.001,
        0.001,
        0.0005,
        0.00025,
    ]
    assert best["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.001
    assert last["configuration"]["tracked_metric"] == "balanced_accuracy"
    assert last["configuration"]["tracked_mode"] == mode
    assert all("selection" not in row for row in history)  # No separate scoring pass.
    assert (
        yaml.safe_load((config.output_directory / "history.yaml").read_text())
        == history
    )
    card = yaml.safe_load(
        (config.output_directory / "best.model_card.yaml").read_text()
    )
    assert card["validation"]["epoch"] == card["provenance"]["epoch"] == 2
    assert card["validation"]["metrics"]["balanced_accuracy"] == tracked_values[1]
    assert {path.name for path in config.output_directory.glob("*.pt")} == {
        "best.pt",
        "last.pt",
    }
    if tracking:
        run = client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED"
        metrics = client.get_metric_history(
            result.mlflow_run_id, "validation/balanced_accuracy"
        )
        assert [item.step for item in metrics] == [1, 2, 3, 4]
        assert [item.value for item in metrics] == tracked_values
        assert (
            run.data.metrics["best_validation_balanced_accuracy"] == tracked_values[1]
        )
        assert run.data.metrics["best_epoch"] == 2
    else:
        assert result.mlflow_run_id is None


@pytest.mark.usefixtures("isolated_validation_rng")
def test_normal_validation_loader_repeats_batches_without_perturbing_training_rng(
    validation_configuration,
):
    """Reuse the existing validation data and loader, not a second evaluation pipeline."""
    trainer = SupConExperiment(validation_configuration)
    before_python = pickle.dumps(random.getstate())
    before_numpy = pickle.dumps(np.random.get_state())
    before_torch = torch.get_rng_state().clone()
    loader = trainer._build_validation_loader()
    reference = list(loader)
    assert reference
    assert pickle.dumps(random.getstate()) == before_python
    assert pickle.dumps(np.random.get_state()) == before_numpy
    assert torch.equal(torch.get_rng_state(), before_torch)
    for candidate_loader in (loader, trainer._build_validation_loader()):
        random.random()
        np.random.random(7)
        torch.rand(7)
        candidate = list(candidate_loader)
        assert len(candidate) == len(reference)
        for (labels, images), (expected_labels, expected_images) in zip(
            candidate, reference, strict=True
        ):
            torch.testing.assert_close(labels, expected_labels)
            torch.testing.assert_close(images, expected_images)
            _, counts = labels.unique(return_counts=True)
            assert len(counts) == validation_configuration.classes_per_batch
            assert torch.all(counts == validation_configuration.samples_per_class)


def test_tracking_configuration_round_trip_and_fixed_temperature_sweep(
    validation_configuration, tmp_path
):
    """Other hyperparameters may sweep; the configured metric and temperature remain fixed."""
    config = replace(
        validation_configuration,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    path = tmp_path / "tracked.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert TrainingConfiguration.from_yaml(path) == config
    assert config.mlflow_parameters()["tracked_mode"] == "max"
    path.write_text(
        path.read_text().replace(
            "\n  lr: 0.001\n", "\n  lr: !sweep\n    values: [0.001, 0.002]\n"
        )
    )
    runs = train_supcon.load_training_configurations(path)
    assert [run.optimizer["lr"] for run in runs] == [0.001, 0.002]
    assert all(run.temperature == config.temperature for run in runs)
    assert all(run.tracked_metric == "balanced_accuracy" for run in runs)
    assert all(run.effective_tracked_mode == "max" for run in runs)


def test_temperature_cannot_vary_within_a_supcon_sweep(
    experiment_configuration, tmp_path
):
    """Reject different temperatures before any run rather than adding another scoring pass."""
    path = tmp_path / "varying-temperature.yaml"
    path.write_text(
        yaml.safe_dump(experiment_configuration.to_dict()).replace(
            "\ntemperature: 0.1\n", "\ntemperature: !sweep\n  values: [0.05, 0.2]\n"
        )
    )
    with pytest.raises(ValueError, match="temperature.*sweep|sweep.*temperature"):
        train_supcon.load_training_configurations(path)
    assert not experiment_configuration.output_directory.exists()


@pytest.mark.parametrize(
    "metric,expected_mode",
    [(None, "min"), ("supcon_loss_mean", "min"), ("balanced_accuracy", "max")],
)
def test_omitted_tracking_direction_preserves_legacy_defaults(
    validation_configuration, metric, expected_mode
):
    """Existing configs keep their old direction; new configs can override it explicitly."""
    document = replace(
        validation_configuration,
        tracked_metric=metric,
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    ).to_dict()
    document.pop("tracked_mode", None)
    config = TrainingConfiguration.from_mapping(document)
    assert config.effective_tracked_mode == expected_mode


@pytest.mark.parametrize("mode", ["", "lowest", True])
def test_invalid_tracking_direction_fails_early(validation_configuration, mode):
    """Invalid directions cannot silently change the chosen checkpoint."""
    with pytest.raises(ValueError, match="tracked_mode"):
        replace(validation_configuration, tracked_mode=mode)


@pytest.mark.parametrize(
    "updates",
    [
        {"scheduler": SchedulerConfiguration("ReduceLROnPlateau", {"mode": "min"})},
        {"early_stopping": {"patience": 2, "tolerance": 0.0, "comparison": "less"}},
    ],
)
def test_explicit_tracking_direction_must_agree_with_scheduler_and_stopping(
    validation_configuration, updates
):
    """One configured direction applies consistently to the same tracked metric."""
    with pytest.raises(ValueError, match="tracked_mode|direction"):
        replace(
            validation_configuration,
            tracked_metric="balanced_accuracy",
            tracked_mode="max",
            validation_metrics={
                "balanced_accuracy": {"function": "balanced_accuracy_score"}
            },
            **updates,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_tracked_metric_aborts_before_checkpointing(
    validation_configuration, monkeypatch, value
):
    """A valid training loss cannot hide an invalid configured validation metric."""
    torch.set_num_threads(2)
    config = replace(
        validation_configuration,
        tracked_metric="balanced_accuracy",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    monkeypatch.setattr(
        SupConExperiment,
        "_validate",
        lambda self, model, loader, criterion: {
            "balanced_accuracy": value,
            "supcon_loss_mean": 0.1,
        },
    )
    with pytest.raises(FloatingPointError, match="tracked|validation"):
        SupConExperiment(config).run_with_result()
    assert not list(config.output_directory.glob("*.pt"))


def test_missing_tracked_metric_does_not_fall_back_to_loss(
    validation_configuration, monkeypatch
):
    """A missing configured metric must fail explicitly instead of switching objectives."""
    torch.set_num_threads(2)
    config = replace(
        validation_configuration,
        tracked_metric="balanced_accuracy",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    monkeypatch.setattr(
        SupConExperiment,
        "_validate",
        lambda self, model, loader, criterion: {"supcon_loss_mean": 0.1},
    )
    with pytest.raises(ValueError, match="balanced_accuracy"):
        SupConExperiment(config).run_with_result()
    assert not list(config.output_directory.glob("*.pt"))


# --- Per-epoch observer hook (used by Optuna pruning; the trainer stays Optuna-free) ---


def test_epoch_callback_observes_each_completed_epoch_after_its_checkpoint(
    validation_configuration, monkeypatch
):
    """The hook sees every epoch's configured tracked value once that epoch is saved."""
    torch.set_num_threads(2)
    config = replace(
        validation_configuration,
        epochs=3,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    tracked_values = iter([0.3, 0.9, 0.6])

    def validate(self, model, loader, criterion):
        """Disagree with SupCon loss so only the configured metric can match."""
        return {"balanced_accuracy": next(tracked_values), "supcon_loss_mean": 0.01}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    observed = []

    def callback(epoch, value):
        """Record the call and the checkpoint that already exists at that moment."""
        saved = torch.load(config.output_directory / "last.pt", weights_only=True)
        observed.append((epoch, value, saved["epoch"]))

    result = SupConExperiment(config).run_with_result(epoch_callback=callback)
    assert observed == [(1, 0.3, 1), (2, 0.9, 2), (3, 0.6, 3)]
    assert result.best_epoch == 2 and result.best_value == 0.9


def test_epoch_callback_reports_training_loss_when_no_validation_is_configured(
    experiment_configuration,
):
    """Training-only runs report the same fallback value used for best-epoch selection."""
    torch.set_num_threads(2)
    observed = []
    result = SupConExperiment(experiment_configuration).run_with_result(
        epoch_callback=lambda epoch, value: observed.append((epoch, value))
    )
    history = torch.load(result.last_checkpoint, weights_only=True)["history"]
    assert observed == [
        (index, row["mean"]) for index, row in enumerate(history, start=1)
    ]


def test_epoch_callback_stops_after_early_stopping_without_extra_calls(
    validation_configuration, monkeypatch
):
    """Early stopping ends the hook sequence at the last trained epoch."""
    torch.set_num_threads(2)
    config = replace(
        validation_configuration,
        epochs=8,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        early_stopping={"patience": 1, "tolerance": 0.0, "comparison": "greater"},
    )
    tracked_values = iter([0.5, 0.4, 0.3, 0.2])
    monkeypatch.setattr(
        SupConExperiment,
        "_validate",
        lambda self, model, loader, criterion: {
            "balanced_accuracy": next(tracked_values)
        },
    )
    observed = []
    SupConExperiment(config).run_with_result(
        epoch_callback=lambda epoch, value: observed.append(epoch)
    )
    assert observed == [1, 2]


def test_epoch_callback_exception_stops_training_and_propagates_unchanged(
    experiment_configuration, monkeypatch
):
    """A hook may stop a run (e.g. pruning); no further epoch runs and the model is released."""
    torch.set_num_threads(2)
    created = []
    original = train_supcon.YoloSupConModel

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(model)
        return model

    class StopRequested(Exception):
        """A caller-owned exception type that must not be wrapped or swallowed."""

    def callback(epoch, value):
        raise StopRequested(f"stop after epoch {epoch}")

    monkeypatch.setattr(train_supcon, "YoloSupConModel", create_model)
    config = replace(experiment_configuration, epochs=3)
    with pytest.raises(StopRequested, match="stop after epoch 1"):
        SupConExperiment(config).run_with_result(epoch_callback=callback)
    last = torch.load(config.output_directory / "last.pt", weights_only=True)
    assert last["epoch"] == 1 and len(last["history"]) == 1
    assert created[0].embeddings is None and not created[0].training
    assert not (config.output_directory / "best.model_card.yaml").exists()


def test_omitting_the_epoch_callback_keeps_existing_behavior(experiment_configuration):
    """The hook is optional; the existing result API is unchanged without it."""
    torch.set_num_threads(2)
    result = SupConExperiment(experiment_configuration).run_with_result(
        epoch_callback=None
    )
    assert torch.load(result.last_checkpoint, weights_only=True)["epoch"] == 2
