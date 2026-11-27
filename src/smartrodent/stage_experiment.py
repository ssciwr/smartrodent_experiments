"""Run one configured stage: tracking, training, export, model card and publication."""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import lightning as L
import mlflow
import yaml
from torch import nn
from torch.utils.data import DataLoader

from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import ModelPublication, git_commit
from smartrodent.model_card import load_training_template, write_training_card
from smartrodent.onnx_export import OnnxExporter
from smartrodent.stage_configurations import StageConfiguration
from smartrodent.stage_runner import StageRunner


class StageExperiment(ABC):
    """Prepare a stage's task and data once, then train and publish it.

    Subclasses supply the task, loaders and what is exported or registered.

    Args:
        configuration: Validated stage settings.
    """

    # Text of the model card's task entry.
    card_task: ClassVar[str]
    # Whether registering a validated model moves the champion alias immediately.
    promotes_on_publication: ClassVar[bool] = False

    task: L.LightningModule
    train_loader: DataLoader
    validation_loader: DataLoader | None
    label_names: dict[str, int]

    def __init__(self, configuration: StageConfiguration):
        """Fail on malformed templates and export settings before any training."""
        self.configuration = configuration
        self.card_template = load_training_template(
            Path(configuration.model_card_template)
        )
        self.exporter = (
            OnnxExporter(configuration.onnx_export, configuration.transforms.image_size)
            if configuration.onnx_export is not None
            else None
        )

    @abstractmethod
    def _export(self, checkpoint: Path) -> Path:
        """Write the stage's ONNX file next to the checkpoint."""

    @abstractmethod
    def _published_model(self, checkpoint: Path) -> nn.Module:
        """Restore the best checkpoint's model as it should be registered."""

    @abstractmethod
    def _base_model(self) -> str:
        """Describe where the encoder came from, for the model card."""

    def _publication_tags(self, result: ExperimentResult) -> dict:
        """Return stage-specific registry tags; none by default."""
        return {}

    @classmethod
    def promote(
        cls, configuration: StageConfiguration, result: ExperimentResult
    ) -> str | None:
        """Compare a study's winner with the registered champion; none by default."""
        return None

    def run(self) -> Path:
        """Train once and return the final epoch's checkpoint path."""
        return self.run_with_result().last_checkpoint

    def run_with_result(
        self,
        *,
        parent_run_id: str | None = None,
        defer_promotion: bool = False,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> ExperimentResult:
        """Train once, export and publish the best checkpoint.

        Args:
            parent_run_id: Explicit active pipeline parent for a tracked child run.
            defer_promotion: Leave champion promotion to the owning study.
            epoch_callback: Optional observer called as ``(epoch, tracked_value)``
                after each scored epoch; its exceptions stop training unchanged.

        Returns:
            The completed run's checkpoints, score and optional registry version.
        """
        config = self.configuration
        context = (
            config.mlflow.start_run(
                config.output_directory.name, parent_run_id=parent_run_id
            )
            if config.mlflow
            else nullcontext()
        )
        with context as run:
            run_id = run.info.run_id if run is not None else None
            source_tags = (
                config.mlflow.run_tags()
                if config.mlflow
                else {"git_commit": git_commit()}
            )
            if config.mlflow:
                mlflow.set_tags(source_tags)
                mlflow.log_params(
                    {
                        "tracked_metric": self.task.objective.metric,
                        "tracked_mode": self.task.objective.mode,
                        "seed": config.seed,
                    }
                )
            artifacts = config.artifacts
            artifacts.directory.mkdir(parents=True, exist_ok=True)
            artifacts.config.write_text(
                yaml.safe_dump(config.to_dict()), encoding="utf-8"
            )
            result = StageRunner(
                artifacts, trainer=config.trainer, early_stopping=config.early_stopping
            ).fit(
                self.task,
                train_loader=self.train_loader,
                validation_loader=self.validation_loader,
                validation_dataset=config.validation_dataset,
                epoch_callback=epoch_callback,
                mlflow_run_id=run_id,
            )
            # Export before any publication: a failed export must leave nothing published.
            if self.exporter is not None:
                result = replace(
                    result, onnx_model=self._export(result.best_checkpoint)
                )
            checksum = hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
            snapshot = {
                "epoch": result.best_epoch,
                "label_names": self.label_names,
                "history": [
                    {
                        "validation": {result.tracked_metric: result.best_value}
                        if config.validation_dataset is not None
                        else {}
                    }
                ],
            }
            self._write_card(result, snapshot, checksum, source_tags)
            if config.mlflow:
                version = self._log_and_publish(result, snapshot, checksum, source_tags)
                if version is not None:
                    result = replace(result, registered_model_version=version)
                    if not defer_promotion:
                        self.promote(config, result)
            return result

    def _log_and_publish(self, result, snapshot, checksum, source_tags) -> str | None:
        """Log metrics and artifacts; register the model when a name is configured.

        Returns:
            The registry version, or None when nothing is registered.
        """
        config = self.configuration
        prefix = (
            "best_validation_"
            if config.validation_dataset is not None
            else "best_training_"
        )
        mlflow.log_metric(
            prefix + result.tracked_metric, result.best_value, step=result.best_epoch
        )
        mlflow.log_metric("best_epoch", result.best_epoch)
        native = config.artifacts.native_checkpoint
        if config.mlflow.registered_model_name is None:
            config.mlflow.log_best_artifacts(
                result.best_checkpoint, result.onnx_model, native
            )
            return None

        def write_card(version, identity):
            cards = self._write_card(
                result, snapshot, checksum, {**source_tags, **identity}, version
            )
            config.mlflow.log_best_artifacts(
                result.best_checkpoint, result.onnx_model, native
            )
            return cards

        return ModelPublication(config.mlflow).publish(
            self._published_model(result.best_checkpoint),
            best_epoch=result.best_epoch,
            validated=config.validation_dataset is not None,
            tags={**source_tags, **self._publication_tags(result)},
            write_card=write_card,
            promote=self.promotes_on_publication,
        )

    def _write_card(
        self, result, snapshot, checksum, provenance: Mapping, version=None
    ) -> tuple[Path, Path]:
        """Describe the exact selected snapshot using the shared template writer."""
        config = self.configuration
        return write_training_card(
            self.card_template,
            result.best_checkpoint,
            snapshot,
            model_details={
                "name": (config.mlflow.registered_model_name if config.mlflow else None)
                or config.output_directory.name,
                "registry_version": version,
                "base_model": self._base_model(),
                "task": self.card_task,
            },
            validation_dataset=config.validation_dataset,
            provenance={
                **provenance,
                "run_id": result.mlflow_run_id,
                "checkpoint_sha256": checksum,
                "dataset_revision": (
                    config.mlflow.dataset_revision if config.mlflow else None
                ),
            },
        )
