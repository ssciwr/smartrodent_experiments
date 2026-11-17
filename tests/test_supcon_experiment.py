"""Exercise the standalone experiment, library modules, and portable checkpoints."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
import torch
import yaml
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from scripts import train_supcon
from scripts.train_supcon import (
    SchedulerConfiguration,
    SupConExperiment,
    TrainingConfiguration,
)
from smartrodent.supcon_datasets import (
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
    assert result.name == "epoch_0002.pt"
    assert (config.output_directory / "epoch_0001.pt").is_file()
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
    assert card["provenance"]["checkpoint"].endswith(".pt")
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


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"epochs": 0}, "positive integers"),
        ({"epochs": True}, "positive integers"),
        ({"workers": -1}, "workers"),
        ({"class_names": "mouse"}, "sequence"),
        ({"class_names": ("", "rat")}, "nonempty strings"),
        ({"class_names": ("rat", "rat")}, "unique"),
        ({"classes_per_batch": 1}, "between 2"),
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
    assert (configuration.output_directory / "epoch_0001.pt").is_file()


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
    assert (configuration.output_directory / "epoch_0001.pt").is_file()


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
    "document,error,message",
    [
        ([], TypeError, "scheduler must be a mapping"),
        ({"name": "ReduceLROnPlateau"}, ValueError, "exactly name and parameters"),
        ({"name": "StepLR", "parameters": {}}, ValueError, "scheduler.name"),
        (
            {"name": "ReduceLROnPlateau", "parameters": []},
            TypeError,
            "parameters must be a mapping",
        ),
        (
            {"name": "ReduceLROnPlateau", "parameters": {"unknown": 1}},
            ValueError,
            "Unsupported",
        ),
    ],
)
def test_invalid_scheduler_mapping_fails(document, error, message):
    """Malformed scheduler settings fail before creating a training run."""
    with pytest.raises(error, match=message):
        SchedulerConfiguration.from_mapping(document)


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
    with pytest.raises(ValueError, match="scheduler.parameters"):
        SchedulerConfiguration("ReduceLROnPlateau", parameters)


def test_scheduler_training_requirements(experiment_configuration):
    """Plateau scheduling requires validation, a sensible floor, and matching modes."""
    scheduler = SchedulerConfiguration("ReduceLROnPlateau", {"mode": "min"})
    with pytest.raises(ValueError, match="validation_dataset"):
        replace(experiment_configuration, scheduler=scheduler)
    with pytest.raises(TypeError, match="SchedulerConfiguration"):
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
    assert result.name == "epoch_0006.pt"
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
        (config.output_directory / "epoch_0001.model_card.yaml").read_text()
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
