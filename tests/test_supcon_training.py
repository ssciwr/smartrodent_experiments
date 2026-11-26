"""Behavioral tests for SupCon training configuration, experiment runs, and tracking."""

import pickle
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from training_test_support import download_artifact

from smartrodent import supcon_training as train_supcon
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.supcon_training import (
    SchedulerConfiguration,
    SupConExperiment,
    TrainingConfiguration,
)


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
    first.finalize()


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
        ({"class_names": ("rat", "rat")}, "unique"),
        ({"classes_per_batch": 3}, "between 2"),
        ({"samples_per_epoch": 1}, "full batch"),
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


def test_yaml_configuration_must_be_a_mapping(tmp_path):
    """A non-mapping YAML document fails without model construction."""
    path = tmp_path / "config.yaml"
    path.write_text("[]")
    with pytest.raises(ValueError):
        TrainingConfiguration.from_yaml(path)


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


@pytest.mark.parametrize(
    "mode,metric,comparison",
    [("min", "supcon_loss_mean", "less"), ("max", "accuracy", "greater")],
)
def test_scheduler_integrates_with_early_stopping_and_checkpoints(
    experiment_configuration, monkeypatch, mode, metric, comparison
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
    )
    values = iter((1.0, 1.2, 1.1, 1.05, 1.1, 1.2))

    def validate(self, model, loader, criterion):
        """Supply noisy validation scores that never beat the first epoch."""
        value = next(values)
        return {metric: value if mode == "min" else 2 - value}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    result = SupConExperiment(config).run()
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
    # The saved scheduler state resumes its plateau count: one more miss reduces.
    assert optimizer.param_groups[0]["lr"] == 0.00025
    restored.finalize()


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


# Each direction and each tracking setting once.
@pytest.mark.parametrize("mode,tracking", [("min", False), ("max", True)])
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
    supcon_values = [0.1, 0.2, 0.3, 0.4]  # Loss prefers epoch 1, NOT the winner.
    scores = iter(zip(tracked_values, supcon_values, strict=True))

    def validate(self, model, loader, criterion):
        """Supply conflicting metrics so accidentally hardcoding loss is observable."""
        tracked, supcon = next(scores)
        return {"balanced_accuracy": tracked, "supcon_loss_mean": supcon}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    result = SupConExperiment(config).run_with_result()
    last = torch.load(result.last_checkpoint, weights_only=True)
    best = torch.load(result.best_checkpoint, weights_only=True)
    # Early stopping follows the configured metric: two misses after its epoch-2 best.
    assert last["epoch"] == 4
    assert best["epoch"] == result.best_epoch == 2  # first exact tie wins
    assert result.best_value == tracked_values[1]  # original value, not a signed score
    # The scheduler steps on the configured metric too: it reduces after each miss.
    assert [row["next_learning_rate"] for row in last["history"]] == [
        0.001,
        0.001,
        0.0005,
        0.00025,
    ]
    if tracking:
        run = client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED"
        assert [
            item.value
            for item in client.get_metric_history(
                result.mlflow_run_id, "validation/balanced_accuracy"
            )
        ] == tracked_values
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
def test_omitted_tracking_direction_follows_the_tracked_metric(
    validation_configuration, metric, expected_mode
):
    """Without tracked_mode, losses are minimized and scores maximized."""
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


@pytest.mark.parametrize("value", [float("nan")])  # inf takes the same finiteness check
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


def test_failed_supcon_export_fails_the_run_before_publication(
    pipeline_document, tracking_store, monkeypatch
):
    """Like the classifier, a failed encoder export publishes nothing to the failed run."""

    def fail(*args, **kwargs):
        raise RuntimeError("simulated ONNX export failure")

    values = deepcopy(pipeline_document["supcon"])
    values.update(pipeline_document["data"])
    values.update(
        epochs=1,
        optimizer={"lr": 0.001},
        onnx_export={},
        output_directory=str(
            Path(pipeline_document["output_directory"]).parent / "failed-export-supcon"
        ),
        mlflow={
            **tracking_store.configuration,
            "registered_model_name": "failed-supcon-encoder",
        },
    )
    config = TrainingConfiguration.from_mapping(values)
    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="export failure"):
        SupConExperiment(config).run_with_result()
    (run,) = tracking_store.client.search_runs([tracking_store.experiment_id])
    assert run.info.status == "FAILED"
    assert "best_model" not in {
        item.path for item in tracking_store.client.list_artifacts(run.info.run_id)
    }
    assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None


@pytest.mark.parametrize("register", [False, True])
def test_tracked_supcon_logs_its_onnx_export_with_the_best_model(
    pipeline_document, tracking_store, register
):
    """The encoder export is published next to the SupCon best.pt, with or without registration."""
    values = deepcopy(pipeline_document["supcon"])
    values.update(pipeline_document["data"])
    settings = {**tracking_store.configuration}
    if register:
        settings["registered_model_name"] = "exported-supcon-encoder"
    values.update(
        epochs=1,
        optimizer={"lr": 0.001},
        onnx_export={},
        output_directory=str(
            Path(pipeline_document["output_directory"]).parent / "exported-supcon"
        ),
        mlflow=settings,
    )
    result = SupConExperiment(
        TrainingConfiguration.from_mapping(values)
    ).run_with_result()
    downloaded = download_artifact(
        tracking_store, result.mlflow_run_id, "best_model/best.onnx"
    )
    assert downloaded.read_bytes() == result.onnx_model.read_bytes()


@pytest.mark.parametrize("upload_fails", [False, True])
def test_standalone_supcon_registry_publication_remains_an_explicit_encoder(
    pipeline_document, tracking_store, monkeypatch, upload_fails
):
    """A validated encoder is registered with its cards, then promoted; a failed card
    upload fails the run and promotes nothing."""
    values = deepcopy(pipeline_document["supcon"])
    values.update(pipeline_document["data"])
    values.update(
        epochs=1,
        optimizer={"lr": 0.001},
        output_directory=str(
            Path(pipeline_document["output_directory"]).parent / "standalone-supcon"
        ),
    )
    encoder_name = "m6-supcon-encoder"
    values["mlflow"] = {
        **tracking_store.configuration,
        "registered_model_name": encoder_name,
    }
    experiment = SupConExperiment(TrainingConfiguration.from_mapping(values))
    client = tracking_store.client
    if upload_fails:
        monkeypatch.setattr(
            train_supcon.MlflowClient,
            "log_model_artifact",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("upload failed")
            ),
        )
        with pytest.raises(RuntimeError, match="upload failed"):
            experiment.run_with_result()
        (run,) = client.search_runs([tracking_store.experiment_id])
        assert run.info.status == "FAILED"
        assert client.get_registered_model(encoder_name).aliases == {}
        assert mlflow.active_run() is None
        return
    result = experiment.run_with_result()
    versions = client.search_model_versions(f"name='{encoder_name}'")
    assert len(versions) == 1 and versions[0].run_id == result.mlflow_run_id
    card = yaml.safe_load(
        (result.output_directory / "best.model_card.yaml").read_text()
    )
    assert card["model"]["task"] == "contrastive-encoder"
    assert str(card["model"]["registry_version"]) == str(versions[0].version)
    assert card["provenance"]["run_id"] == result.mlflow_run_id
    assert card["provenance"]["model_id"]
    champion = client.get_model_version_by_alias(
        encoder_name, tracking_store.configuration["champion_alias"]
    )
    assert champion.version == versions[0].version
    assert client.get_run(result.mlflow_run_id).info.status == "FINISHED"
    assert mlflow.active_run() is None
