"""Run Lightning training stages with shared checkpointing and logging."""

from collections.abc import Callable, Mapping
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from mlflow import active_run, get_tracking_uri
from torch.utils.data import DataLoader

from smartrodent.experiment_results import ExperimentResult
from smartrodent.output_paths import RunArtifacts


class StageRunner:
    """Fit a task and report the checkpoints selected by its objective.

    Args:
        artifacts: Declared stage checkpoint and logger paths.
        trainer: Native Lightning Trainer keyword arguments.
        early_stopping: Optional native patience and min_delta settings.

    Raises:
        ValueError: Trainer settings set a runner-owned key or omit max_epochs.
    """

    # Trainer arguments that the runner fills in itself.
    reserved_trainer_keys = frozenset({"callbacks", "logger", "default_root_dir"})

    def __init__(
        self,
        artifacts: RunArtifacts,
        *,
        trainer: Mapping,
        early_stopping: Mapping | None = None,
    ):
        reserved = self.reserved_trainer_keys & trainer.keys()
        if reserved:
            raise ValueError(
                f"Trainer settings {sorted(reserved)} are owned by the stage runner"
            )
        if "max_epochs" not in trainer:
            raise ValueError("Trainer settings must state max_epochs explicitly")
        self.artifacts = artifacts
        self.trainer = dict(trainer)
        self.early_stopping = early_stopping

    def fit(
        self,
        task: L.LightningModule,
        *,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        validation_dataset: Path | None = None,
        epoch_callback: Callable[[int, float], None] | None = None,
        mlflow_run_id: str | None = None,
    ) -> ExperimentResult:
        """Train and select the best epoch with the task's metric and direction.

        Args:
            task: Lightning task exposing an objective with a metric and mode.
            train_loader: Training data batches.
            validation_loader: Optional evaluation data batches.
            validation_dataset: Path identifying the evaluation split, when present.
            epoch_callback: Optional notification after checkpointing a scored epoch.
            mlflow_run_id: Active MLflow run owned by the caller, when tracking.

        Returns:
            The completed run's selected and latest checkpoint information.
        """
        objective = task.objective
        validated = validation_loader is not None
        best = ModelCheckpoint(
            dirpath=self.artifacts.best_checkpoint.parent,
            filename=self.artifacts.best_checkpoint.stem,
            monitor=objective.metric,
            mode=objective.mode,
            save_on_train_epoch_end=not validated,
            enable_version_counter=False,
        )
        # A monitored ModelCheckpoint does not update save_last on every missed
        # improvement. An unmonitored checkpoint records the latest trained epoch.
        last = _LatestCheckpoint(
            self.artifacts.last_checkpoint, objective.metric, validated, epoch_callback
        )
        callbacks = [_FiniteLoss(), best, last]
        if self.early_stopping is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=objective.metric, mode=objective.mode, **self.early_stopping
                )
            )
        trainer = L.Trainer(
            default_root_dir=self.artifacts.directory,
            logger=self._loggers(mlflow_run_id),
            callbacks=callbacks,
            **self.trainer,
        )
        trainer.fit(
            task, train_dataloaders=train_loader, val_dataloaders=validation_loader
        )
        if best.best_model_score is None or not last.best_model_path:
            raise RuntimeError(
                "Training ended without a completed best and last checkpoint"
            )
        best_epoch = torch.load(
            self.artifacts.best_checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )["epoch"]
        if self.artifacts.native_checkpoint is not None:
            self._write_native(task)
        return ExperimentResult(
            output_directory=self.artifacts.directory,
            best_checkpoint=self.artifacts.best_checkpoint,
            last_checkpoint=self.artifacts.last_checkpoint,
            best_epoch=best_epoch + 1,
            tracked_metric=objective.metric,
            tracked_mode=objective.mode,
            best_value=float(best.best_model_score),
            validation_dataset=validation_dataset,
            mlflow_run_id=mlflow_run_id,
        )

    def _write_native(self, task: L.LightningModule) -> None:
        """Save the best epoch's native model next to the best checkpoint.

        The payload follows Ultralytics' checkpoint layout, so ``YOLO(path)``
        can load it; the model comes from the encoder's ``to_native``.
        """
        best = type(task).load_from_checkpoint(
            self.artifacts.best_checkpoint, map_location="cpu"
        )
        torch.save(
            {"model": best.native_model(), "train_args": {"task": "classify"}},
            self.artifacts.native_checkpoint,
        )

    def _loggers(self, mlflow_run_id: str | None) -> list:
        """Write the CSV files to the run directory and attach tracking when active."""
        loggers = [CSVLogger(str(self.artifacts.directory), name="", version="")]
        if mlflow_run_id is not None:
            run = active_run()
            if run is None or run.info.run_id != mlflow_run_id:
                raise ValueError(
                    "Stage tracking requires the caller's active MLflow run"
                )
            loggers.append(
                _ActiveRunLogger(
                    run_id=mlflow_run_id,
                    tracking_uri=get_tracking_uri(),
                    log_model=False,
                    synchronous=True,
                )
            )
        return loggers


class _ActiveRunLogger(MLFlowLogger):
    """Log through Lightning while the caller owns MLflow run termination."""

    def finalize(self, status: str = "success") -> None:
        """Leave the run open for export and publication after `fit` returns."""


class _FiniteLoss(Callback):
    """Stop on a nonfinite batch loss before an epoch can checkpoint it."""

    @staticmethod
    def _check(outputs, split: str) -> None:
        """Accept the standard Lightning loss tensor or a result dict containing it."""
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite {split} loss")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Reject a nonfinite training batch loss."""
        self._check(outputs, "training")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Reject a nonfinite validation batch loss."""
        self._check(outputs, "validation")


class _LatestCheckpoint(ModelCheckpoint):
    """Tell an epoch observer only after Lightning writes the latest checkpoint."""

    def __init__(
        self, path: Path, metric: str, validated: bool, observe: Callable | None
    ):
        super().__init__(
            dirpath=path.parent,
            filename=path.stem,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
        self._metric = metric
        self._validated = validated
        self._observe = observe
        self._validation_epoch = None

    def on_validation_end(self, trainer, pl_module) -> None:
        """Remember whether the current epoch has a real validation score."""
        super().on_validation_end(trainer, pl_module)
        if not trainer.sanity_checking:
            self._validation_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Save the epoch and report its score, allowing pruning to propagate."""
        super().on_train_epoch_end(trainer, pl_module)
        if self._observe is not None and (
            not self._validated or self._validation_epoch == trainer.current_epoch
        ):
            value = trainer.callback_metrics[self._metric]
            self._observe(trainer.current_epoch + 1, float(value))
