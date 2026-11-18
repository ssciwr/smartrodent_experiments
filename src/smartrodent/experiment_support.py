"""Shared configuration and I/O for standalone training experiments."""

import functools
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
import numpy as np
import sklearn.metrics
import torch
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


@dataclass(frozen=True)
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

    tracking_uri: str
    experiment_name: str
    registered_model_name: str | None = None
    model_artifact_name: str = "classifier"
    champion_alias: str = "champion"
    dataset_revision: str | None = None

    def __post_init__(self):
        """Reject missing or blank MLflow settings before training starts."""
        for field_name in (
            "tracking_uri",
            "experiment_name",
            "model_artifact_name",
            "champion_alias",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"mlflow.{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"mlflow.{field_name} must not be blank")
        for field_name in ("registered_model_name", "dataset_revision"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"mlflow.{field_name} must be a string or null")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"mlflow.{field_name} must not be blank")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "MlflowConfiguration | None":
        """Build optional MLflow settings from YAML-compatible values.

        Args:
            values: Optional nested ``mlflow`` configuration mapping.

        Returns:
            Validated MLflow settings, or ``None`` when tracking is disabled.

        Raises:
            TypeError: The supplied value is not a mapping.
            ValueError: Required keys are missing or unexpected keys are present.
        """
        if values is None:
            return None
        if not isinstance(values, Mapping):
            raise TypeError("mlflow must be a mapping")
        required = {"tracking_uri", "experiment_name"}
        optional = {
            "registered_model_name",
            "model_artifact_name",
            "champion_alias",
            "dataset_revision",
        }
        supplied = set(values)
        if not required.issubset(supplied) or not supplied.issubset(
            required | optional
        ):
            missing = sorted(required.difference(supplied))
            unexpected = sorted(supplied.difference(required | optional))
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected keys: {', '.join(unexpected)}")
            raise ValueError("Invalid mlflow configuration; " + "; ".join(details))
        return cls(**values)

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
    ) -> str:
        """Register the supplied best model, attach cards, then optionally promote it.

        Args:
            model: CPU/eval model restored from the stage's selected checkpoint.
            best_epoch: Epoch represented by both model and checkpoint.
            validated: Whether validation was performed; only then promote the alias.
            tags: Source revisions and selected-encoder provenance.
            write_card: Callback receiving version and model identity metadata;
                returns the completed YAML and Markdown card paths.

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
        if validated:
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


@dataclass(frozen=True)
class SchedulerConfiguration:
    """Configure a native PyTorch validation-driven learning-rate scheduler.

    Args:
        name: Supported PyTorch scheduler class name, ``ReduceLROnPlateau``.
        parameters: Native scheduler arguments, excluding the optimizer.
    """

    name: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        """Reject unsupported schedulers and invalid native parameters early."""
        if self.name != "ReduceLROnPlateau":
            raise ValueError("scheduler.name must be 'ReduceLROnPlateau'")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("scheduler.parameters must be a mapping")
        allowed = {
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
        }
        if set(self.parameters).difference(allowed):
            raise ValueError("Unsupported scheduler.parameters keys")
        if self.mode not in ("min", "max"):
            raise ValueError("scheduler.parameters.mode must be 'min' or 'max'")
        if self.parameters.get("threshold_mode", "rel") not in ("rel", "abs"):
            raise ValueError(
                "scheduler.parameters.threshold_mode must be 'rel' or 'abs'"
            )
        for name in ("patience", "cooldown"):
            value = self.parameters.get(name, 10 if name == "patience" else 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"scheduler.parameters.{name} must be a nonnegative integer"
                )
        defaults = {"factor": 0.1, "threshold": 0.0001, "min_lr": 0.0, "eps": 1e-8}
        for name, default in defaults.items():
            value = self.parameters.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"scheduler.parameters.{name} must be finite and nonnegative"
                )
        if not 0 < self.parameters.get("factor", 0.1) < 1:
            raise ValueError(
                "scheduler.parameters.factor must be between 0 and 1 exclusively"
            )

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
        if not isinstance(values, Mapping):
            raise TypeError("scheduler must be a mapping")
        if set(values) != {"name", "parameters"}:
            raise ValueError("scheduler requires exactly name and parameters")
        if not isinstance(values["parameters"], Mapping):
            raise TypeError("scheduler.parameters must be a mapping")
        return cls(values["name"], dict(values["parameters"]))

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

    def __init__(self, patience: int, tolerance: float, comparison: str):
        """Validate settings and initialize independent stopping state."""
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("early_stopping.patience must be a positive integer")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
        ):
            raise ValueError("early_stopping.tolerance must be finite")
        if tolerance < 0:
            raise ValueError("early_stopping.tolerance must be nonnegative")
        if comparison not in {"less", "greater"}:
            raise ValueError("early_stopping.comparison must be 'less' or 'greater'")
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
        if not isinstance(config, Mapping):
            raise TypeError("early_stopping must be a mapping")
        required = {"patience", "tolerance", "comparison"}
        supplied = set(config)
        if supplied != required:
            missing = sorted(required.difference(supplied))
            unexpected = sorted(supplied.difference(required))
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected keys: {', '.join(unexpected)}")
            raise ValueError(
                "Invalid early_stopping configuration; " + "; ".join(details)
            )

        return cls(config["patience"], config["tolerance"], config["comparison"])

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
        """Keep preprocessing settings separate from stage-specific data loading."""
        self.settings = settings

    def training(self) -> Compose:
        """Build independently sampled training augmentations.

        Returns:
            The existing SupCon augmentation pipeline, applicable to each view.
        """
        settings = self.settings
        size = settings["image_size"]
        return Compose(
            [
                *self._preprocessing_steps(),
                RandomAdjustSharpness(**settings["random_adjust_sharpness"]),
                RandomGrayscale(**settings["random_grayscale"]),
                RandomPerspective(**settings["random_perspective"]),
                RandomHorizontalFlip(**settings["random_horizontal_flip"]),
                RandomResizedCrop((size, size), **settings["random_resized_crop"]),
            ]
        )

    def validation(self) -> Compose:
        """Build deterministic preprocessing without training augmentations.

        Returns:
            RGB tensor conversion, zero-to-one scaling, and configured resize.
        """
        return Compose(self._preprocessing_steps())

    def _preprocessing_steps(self) -> list[torch.nn.Module]:
        """Keep conversion, scaling, and resizing identical across both pipelines."""
        size = self.settings["image_size"]
        return [
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Resize((size, size), **self.settings["resize"]),
        ]


def resolve_validation_metrics(
    specs: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Resolve configured sklearn metrics and reject invalid specs early.

    Args:
        specs: Named sklearn metric functions and keyword options, or ``None``.

    Returns:
        Scorers accepting target and predicted label arrays.

    Raises:
        TypeError: Metric specifications are not mappings.
        ValueError: A metric name, function, or option cannot produce a scalar score.
    """
    if specs is None:
        return {}
    if not isinstance(specs, Mapping):
        raise TypeError("validation_metrics must be a mapping")

    scorers = {}
    for name, spec in specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("validation metric names must be nonempty strings")
        if not isinstance(spec, Mapping):
            raise TypeError(f"validation_metrics.{name} must be a mapping")
        options = dict(spec)
        function_name = options.pop("function", None)
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
