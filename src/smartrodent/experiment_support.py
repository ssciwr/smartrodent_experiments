"""Shared configuration and I/O for standalone training experiments."""

import functools
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, validate_call
from pydantic.dataclasses import dataclass

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
import numpy as np
import sklearn.metrics
import torch
from smartrodent.training_validation import (
    Count, Nonblank, NonnegativeNumber, PlateauSettings, ValidationMetrics, parse_settings,
)
from torchvision.transforms.v2 import (
    Compose,
    PILToTensor,
    RandomAdjustSharpness,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    Resize,
    ToDtype,
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MlflowConfiguration:
    """Configure MLflow tracking for a standalone experiment.

    Args:
        tracking_uri: MLflow tracking server or local database URI.
        experiment_name: Experiment receiving all expanded sweep runs.
        registered_model_name: Optional registry name for each run's best model.
        model_artifact_name: Name of the logged MLflow PyTorch model.
        champion_alias: Alias assigned to the best validated registered version.
        dataset_revision: Optional immutable dataset revision recorded as a tag.
    """

    tracking_uri: Nonblank
    experiment_name: Nonblank
    registered_model_name: Nonblank | None = None
    model_artifact_name: Nonblank = "classifier"
    champion_alias: Nonblank = "champion"
    dataset_revision: Nonblank | None = None

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "MlflowConfiguration | None":
        """Build optional MLflow settings from YAML-compatible values.

        Args:
            values: Optional nested ``mlflow`` configuration mapping.

        Returns:
            Validated MLflow settings, or ``None`` when tracking is disabled.

        Raises:
            ValueError: The schema rejects missing, unexpected, or invalid fields.
        """
        if values is None:
            return None
        return parse_settings(cls, values)

    def configure(self) -> None:
        """Select the configured tracking backend and experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str, *, parent_run_id: str | None = None):
        """Start an owned root run or an explicitly authorized pipeline child.

        Args:
            run_name: Display name for the new run.
            parent_run_id: Active pipeline parent, or None for standalone execution.

        Returns:
            MLflow's active-run context, which closes only this newly owned run.

        Raises:
            ValueError: Implicit resumption is requested, an unrelated run is active,
                or the requested parent is absent.
        """
        if os.environ.get("MLFLOW_RUN_ID"):
            raise ValueError("Implicit run resumption via MLFLOW_RUN_ID is not supported")
        active = mlflow.active_run()
        if parent_run_id is None and active is not None:
            raise ValueError("An unrelated MLflow run is already active")
        if parent_run_id is not None and (active is None or active.info.run_id != parent_run_id):
            raise ValueError("The requested MLflow parent run must be active")
        self.configure()
        return mlflow.start_run(run_name=run_name, nested=parent_run_id is not None)

    def log_best_artifacts(self, checkpoint: Path) -> None:
        """Publish a best checkpoint and its completed local cards to the owned run.

        Args:
            checkpoint: Selected best.pt whose YAML/Markdown cards already exist.
        """
        for path in (checkpoint, checkpoint.with_suffix(".model_card.yaml"), checkpoint.with_suffix(".model_card.md")):
            mlflow.log_artifact(str(path), artifact_path="best_model")

    def run_tags(self) -> dict[str, str]:
        """Return source revisions that make a tracked run reproducible.

        Returns:
            Git commit and, when configured, dataset revision tags.
        """
        tags = {"git_commit": _git_commit()}
        if self.dataset_revision is not None:
            tags["dataset_revision"] = self.dataset_revision
        return tags


class ModelPublication:
    """Publish a selected trained model with consistent cards and registry metadata.

    Args:
        configuration: Tracking/registry settings for this stage, not its pipeline.
        client: Optional existing client, preserving standalone publication callers.
    """

    def __init__(self, configuration: MlflowConfiguration, client: MlflowClient | None = None):
        """Keep publication separate from reconstruction, optionally using an existing client."""
        self.configuration = configuration
        self.client = client

    def publish(
        self,
        model: torch.nn.Module,
        *,
        best_epoch: int,
        validated: bool,
        tags: Mapping,
        write_card: Callable[[str, dict[str, str | None]], tuple[Path, Path]],
        promote: bool = True,
    ) -> str:
        """Register the supplied best model, attach cards, then optionally promote it.

        Args:
            model: CPU/eval model restored from the stage's selected checkpoint.
            best_epoch: Epoch represented by both model and checkpoint.
            validated: Whether validation was performed; only then promote the alias.
            tags: Source revisions and selected-encoder provenance.
            write_card: Callback receiving version and model identity metadata;
                returns the completed YAML and Markdown card paths.
            promote: Assign the alias immediately for validated models. Classifier
                sweeps disable this until every candidate has published successfully.

        Returns:
            The created registry version. Remote publication is not transactional;
            failures after version creation propagate without rollback guarantees.
        """
        config = self.configuration
        if config.registered_model_name is None:
            raise ValueError("Model publication requires registered_model_name")
        # MLflow 3.18 defaults to traced pt2 graphs. Preserve this project's
        # reconstructible wrappers and label metadata using trusted own-model pickle.
        info = mlflow.pytorch.log_model(
            pytorch_model=model, name=config.model_artifact_name,
            registered_model_name=config.registered_model_name,
            serialization_format="pickle",
        )
        if info.registered_model_version is None:
            raise RuntimeError("MLflow did not return a registered model version")
        version = str(info.registered_model_version)
        client = (
            self.client if self.client is not None
            else MlflowClient(tracking_uri=config.tracking_uri)
        )
        version_tags = {
            **tags,
            "validation_status": "passed" if validated else "not_validated",
            "best_epoch": best_epoch,
        }
        for key, value in version_tags.items():
            if value is not None:
                client.set_model_version_tag(config.registered_model_name, version, key, str(value))
        cards = write_card(version, {"run_id": info.run_id, "model_id": info.model_id})
        self._attach_model_cards(client, info.model_id, version, cards)
        if validated and promote:
            client.set_registered_model_alias(config.registered_model_name, config.champion_alias, version)
        return version

    def _attach_model_cards(
        self, client: MlflowClient, model_id: str, version: str, cards: tuple[Path, Path]
    ) -> None:
        """Attach completed cards and their description before alias promotion is allowed."""
        for path in cards:
            client.log_model_artifact(model_id, str(path))
        client.update_model_version(
            name=self.configuration.registered_model_name,
            version=version,
            description=cards[1].read_text(encoding="utf-8"),
        )


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SchedulerConfiguration:
    """Configure a native PyTorch validation-driven learning-rate scheduler.

    Args:
        name: Supported PyTorch scheduler class name, ``ReduceLROnPlateau``.
        parameters: Native scheduler arguments, excluding the optimizer.
    """

    name: Literal["ReduceLROnPlateau"]
    parameters: PlateauSettings

    @property
    def mode(self) -> str:
        """Return the native comparison mode, defaulting to PyTorch's minimum."""
        return self.parameters.get("mode", "min")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "SchedulerConfiguration | None":
        """Parse optional YAML scheduler settings.

        Args:
            values: Mapping with ``name`` and ``parameters``, or ``None``.

        Returns:
            Validated scheduler settings, or ``None`` for constant learning rate.
        """
        if values is None:
            return None
        return parse_settings(cls, values)

    def build(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Construct the scheduler bound to this run's optimizer.

        Args:
            optimizer: Optimizer whose learning rate will be adjusted.

        Returns:
            A native PyTorch plateau scheduler with the configured parameters.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.parameters)


class EarlyStopping:
    """Stop after repeated failures to improve on the best value seen so far.

    A value improves only when it beats the best value by more than the
    tolerance. Comparing against the best, not the previous value, keeps noisy
    metrics from resetting patience with dips that never beat the best.

    Args:
        patience: Consecutive non-improving values allowed before stopping.
        tolerance: Minimum improvement required to reset patience.
        comparison: ``less`` to minimize or ``greater`` to maximize the value.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, patience: Count, tolerance: NonnegativeNumber, comparison: Literal["less", "greater"]):
        """Validate settings and initialize independent stopping state."""
        self.patience = patience
        self.current_patience = patience
        self.tol = float(tolerance)
        self.comparison = comparison
        self.best: float | None = None
        self.history: list[float] = []

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EarlyStopping":
        """Build an early stopper from its YAML-compatible configuration.

        Args:
            config: Mapping containing patience, tolerance, and comparison.

        Returns:
            An early stopper with independent state.
        """
        return cls(**config)

    def _improves(self, value: float) -> bool:
        """Return whether a value beats the best value by more than the tolerance."""
        if self.comparison == "less":
            return value < self.best - self.tol
        elif self.comparison == "greater":
            return value > self.best + self.tol
        else:
            raise ValueError(f"Unsupported comparison '{self.comparison}'")

    def __call__(self, value: float) -> bool:
        """Record a value and report whether patience has been exhausted.

        Args:
            value: Finite metric value for the current epoch.

        Returns:
            Whether the configured patience has been exhausted.
        """
        if not np.isfinite(value):
            raise ValueError("Early-stopping values must be finite")
        self.history.append(value)
        if self.best is None:
            self.best = value
            return False

        if self._improves(value):
            self.best = value
            self.current_patience = self.patience
        else:
            self.current_patience -= 1
        return self.current_patience == 0


class ImageTransforms:
    """Build single-image preprocessing shared by training experiments.

    Args:
        settings: Validated image size and torchvision augmentation settings.
    """

    def __init__(self, settings: Mapping[str, Any]):
        """Prepare and verify both pipelines once for the owning experiment."""
        size = settings["image_size"]
        preprocessing = [
            PILToTensor(), ToDtype(torch.float32, scale=True),
            Resize((size, size), **settings["resize"]),
        ]
        self._training = Compose([
            *preprocessing,
            RandomAdjustSharpness(**settings["random_adjust_sharpness"]),
            RandomGrayscale(**settings["random_grayscale"]),
            RandomPerspective(**settings["random_perspective"]),
            RandomHorizontalFlip(**settings["random_horizontal_flip"]),
            RandomResizedCrop((size, size), **settings["random_resized_crop"]),
        ])
        self._validation = Compose(preprocessing)

    def training(self) -> Compose:
        """Return the prepared, independently sampled training pipeline.

        Returns:
            The existing SupCon augmentation pipeline, applicable to each view.
        """
        return self._training

    def validation(self) -> Compose:
        """Build deterministic preprocessing without training augmentations.

        Returns:
            RGB tensor conversion, zero-to-one scaling, and configured resize.
        """
        return self._validation


def resolve_validation_metrics(
    specs: ValidationMetrics | None,
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Prepare scorers from schema-validated specs and verify their native behavior.

    Args:
        specs: Validated named sklearn functions and keyword options, or ``None``.

    Returns:
        Scorers accepting target and predicted label arrays.

    Raises:
        ValueError: A scoring function or its options cannot produce a scalar score.
    """
    if specs is None:
        return {}
    scorers = {}
    for name, spec in specs.items():
        options = dict(spec)
        function_name = options.pop("function")
        function = getattr(sklearn.metrics, str(function_name), None)
        if not callable(function):
            raise ValueError(
                f"validation_metrics.{name}: unknown sklearn.metrics function "
                f"'{function_name}'"
            )
        scorer = functools.partial(function, **options)
        try:
            float(scorer(np.array([0, 1]), np.array([0, 1])))
        except Exception as error:
            raise ValueError(
                f"validation_metrics.{name}: '{function_name}' with options "
                f"{options} cannot score label arrays: {error}"
            ) from error
        scorers[name] = scorer
    return scorers


def validate_stopping(configuration, mode: str | None) -> None:
    """Check coupled learning-rate and comparison rules once for both trainers.

    Args:
        configuration: Validated training settings.
        mode: Required tracking direction; None preserves standalone SupCon defaults.
    """
    modes = {mode} if mode is not None else set()
    if configuration.early_stopping is not None:
        modes.add("min" if configuration.early_stopping["comparison"] == "less" else "max")
    if configuration.scheduler is not None:
        modes.add(configuration.scheduler.mode)
        if configuration.scheduler.parameters.get("min_lr", 0) > configuration.optimizer["lr"]:
            raise ValueError("scheduler min_lr must not exceed optimizer.lr")
    if len(modes) > 1:
        raise ValueError("scheduler and early_stopping directions must agree with tracked_mode")


class CheckpointStore:
    """Keep only the latest and best snapshots in one experiment directory.

    Publication is atomic for each file, not a transaction across both files.
    A directory must have a single writer and support hard links, as required by
    the existing SupCon checkpoint writer. This does not guarantee power-loss
    durability or delete artifacts left by older runs.

    Args:
        output_directory: Directory for ``last.pt`` and ``best.pt``.
    """

    def __init__(self, output_directory: str | Path):
        """Create the checkpoint directory without pruning existing artifacts."""
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: Mapping[str, Any], *, is_best: bool) -> Path:
        """Atomically publish the latest snapshot and optionally promote it to best.

        Args:
            snapshot: Checkpoint payload assembled by the stage-specific trainer.
            is_best: Whether this snapshot improves the trainer's selection metric.

        Returns:
            Path to the completed ``last.pt`` checkpoint.

        Raises:
            OSError: The filesystem cannot write, link, or publish the snapshot.
            RuntimeError: PyTorch cannot serialize the snapshot.
        """
        path = self.output_directory / "last.pt"
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            torch.save(dict(snapshot), temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        if is_best:
            self._replace_best(path)
        return path

    def _replace_best(self, checkpoint: Path) -> None:
        """Publish a hard link so promoting the latest state needs no extra data copy."""
        best_path = self.output_directory / "best.pt"
        temporary_path = best_path.with_name(f".{best_path.name}.tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            # Replacing last.pt next epoch leaves this inode, and therefore the
            # best state, unchanged. Never write in place to either checkpoint.
            os.link(checkpoint, temporary_path)
            temporary_path.replace(best_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _git_commit() -> str:
    """Return the exact Git commit associated with a tracked run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git returned an empty commit revision")
    return commit
