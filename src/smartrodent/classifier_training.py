"""Standalone cross-entropy training for portable YOLO classifiers."""

import hashlib
import math
import random
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from pydantic import ConfigDict, StrictBool
from pydantic.dataclasses import dataclass

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_model import YoloClassifierModel
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import (
    CheckpointStore,
    EarlyStopping,
    ImageTransforms,
    MlflowConfiguration,
    ModelPublication,
    SchedulerConfiguration,
    _git_commit,
    resolve_validation_metrics,
    validate_stopping,
)
from smartrodent.training_validation import (
    CheckpointPath, ClassNames, Count, EncoderBatchnorm, InitializationSettings, MappingInput, Mode, Name,
    NonnegativeInt, OptimizerSettings, Seed, StoppingSettings, TransformSettings, ValidationMetrics,
    parse_settings, settings_dict, validate_output_paths,
)
from smartrodent.model_card import load_card_template, write_model_card
from smartrodent.datasets import ImageDataset


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class ClassifierConfiguration:
    """Configure a fresh CE experiment independently of SupCon sampling.

    Args:
        train_dataset: Root containing one training directory per class.
        output_directory: Destination outside both input splits.
        class_names: Unique species names, assigned alphabetical IDs.
        initialization: Explicit encoder source and replacement-head settings.
        transforms: Shared image preprocessing and augmentation settings.
        optimizer: Native AdamW parameters, including a positive learning rate.
        validation_dataset: Optional existing validation split.
        freeze_encoder: Disable updates before the selected classifier boundary.
        encoder_batchnorm: Stored statistics (eval) or updating statistics (adapt).
        epochs: Maximum complete training epochs.
        batch_size: Ordinary single-view image batch size, retaining the tail.
        workers: DataLoader subprocess count.
        device: PyTorch device string.
        seed: Seed for initialization, augmentations, and shuffled epochs.
        tracked_metric: CE loss or a configured validation metric.
        tracked_mode: Explicit min/max direction, or the metric default.
        validation_metrics: Optional named sklearn label metrics.
        scheduler: Optional shared validation-driven scheduler settings.
        early_stopping: Optional shared early-stopping settings.
        mlflow: Optional metric tracking; registry publication is deferred.
        model_card_template: YAML template for the selected checkpoint's card.
    """

    train_dataset: Path
    output_directory: Path
    class_names: ClassNames
    initialization: InitializationSettings
    transforms: TransformSettings
    optimizer: OptimizerSettings
    validation_dataset: Path | None = None
    freeze_encoder: StrictBool = True
    encoder_batchnorm: EncoderBatchnorm = "eval"
    epochs: Count = 1
    batch_size: Count = 32
    workers: NonnegativeInt = 0
    device: Name = "cpu"
    seed: Seed = 42
    tracked_metric: Name = "ce_loss_mean"
    tracked_mode: Mode | None = None
    validation_metrics: ValidationMetrics | None = None
    scheduler: MappingInput[SchedulerConfiguration] | None = None
    early_stopping: StoppingSettings | None = None
    mlflow: MappingInput[MlflowConfiguration] | None = None
    model_card_template: CheckpointPath = "configs/model_card_template.yaml"

    def __post_init__(self):
        """Validate settings before any model I/O or output creation."""
        if "ce_loss_mean" in (self.validation_metrics or {}):
            raise ValueError("validation_metrics cannot override ce_loss_mean")
        if self.tracked_metric not in {
            "ce_loss_mean",
            *(self.validation_metrics or {}),
        }:
            raise ValueError(f"tracked_metric '{self.tracked_metric}' is not available")
        if self.validation_dataset is None and (
            self.validation_metrics
            or self.tracked_metric != "ce_loss_mean"
            or self.scheduler is not None
            or self.early_stopping is not None
        ):
            raise ValueError(
                "Validation objectives, scheduling, and stopping require validation_dataset"
            )
        validate_stopping(self, self.effective_tracked_mode)
        validate_output_paths(
            [self.output_directory], (self.train_dataset, self.validation_dataset),
            [self.model_card_template, *(node.get("checkpoint") for node in self.initialization.values())],
        )

    @property
    def effective_tracked_mode(self) -> str:
        """Return explicit direction or loss-min/score-max defaults."""
        return self.tracked_mode or (
            "min" if self.tracked_metric == "ce_loss_mean" else "max"
        )

    @property
    def tracked_metric_definition(self) -> dict:
        """Return the scoring function/options used for objective comparability.

        Returns:
            A copy of the configured sklearn definition, or the fixed native CE
            objective. Metric display names alone do not identify an objective.
        """
        if self.tracked_metric == "ce_loss_mean":
            return {"function": "torch.nn.functional.cross_entropy", "reduction": "mean"}
        else:
            return deepcopy(dict(self.validation_metrics[self.tracked_metric]))

    @classmethod
    def from_mapping(cls, values: Mapping) -> "ClassifierConfiguration":
        """Parse one concrete configuration mapping.

        Args:
            values: YAML-compatible standalone classifier settings.

        Returns:
            Validated classifier settings.
        """
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassifierConfiguration":
        """Read a standalone YAML configuration.

        Args:
            path: Configuration filename.

        Returns:
            Validated settings for exactly one CE experiment.
        """
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """Return independent YAML-compatible settings.

        Returns:
            All settings, preserving configured rather than inferred direction.
        """
        return settings_dict(self)


class ClassifierExperiment:
    """Train a CE classifier using shared infrastructure by composition.

    Args:
        configuration: Validated standalone classifier configuration.
        encoder_provenance: Optional selected-run metadata supplied by orchestration.
            The source path/checksum are always calculated from the actual input.
    """

    def __init__(self, configuration: ClassifierConfiguration, *, encoder_provenance: Mapping | None = None):
        """Prepare scorers and transforms without modifying inputs or creating outputs."""
        self.configuration = configuration
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        self.transforms = ImageTransforms(configuration.transforms)
        self.scorers = resolve_validation_metrics(configuration.validation_metrics)
        self.card_template = load_card_template(Path(configuration.model_card_template))
        self.document = configuration.to_dict()
        self.training_dataset = self._dataset(configuration.train_dataset, training=True)
        self.validation_dataset = (
            self._dataset(configuration.validation_dataset, training=False)
            if configuration.validation_dataset is not None else None
        )

    def _dataset(self, path: Path, *, training: bool) -> ImageDataset:
        """Reuse the existing class mapping and single-view dataset, rejecting empty classes."""
        dataset = ImageDataset(
            Path(path),
            self.configuration.class_names if training else self.training_dataset.label_names,
            transform=self.transforms.training()
            if training
            else self.transforms.validation(),
        )
        if any(count == 0 for count in dataset.class_counts.values()):
            raise ValueError("Every requested dataset class must be nonempty")
        return dataset

    def _build_loader(self) -> tuple[ImageDataset, DataLoader]:
        """Build ordinary shuffled, single-view epochs retaining every source image."""
        config = self.configuration
        dataset = self.training_dataset
        generator = torch.Generator().manual_seed(config.seed)
        return dataset, DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            generator=generator,
            drop_last=False,
        )

    def _build_validation_loader(self) -> DataLoader | None:
        """Traverse the complete existing validation split in dataset order."""
        config = self.configuration
        if self.validation_dataset is None:
            return None
        return DataLoader(
            self.validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            drop_last=False,
            generator=torch.Generator().manual_seed(config.seed),
        )

    def _validate(self, model: torch.nn.Module, loader: DataLoader) -> dict[str, float]:
        """Compute sample-weighted raw-logit CE and full-split label metrics."""
        incoming_mode = model.training
        model.eval()
        loss_sum, count = 0.0, 0
        targets, predictions = [], []
        try:
            with torch.no_grad():
                for labels, images in loader:
                    labels = labels.to(self.configuration.device)
                    logits = model(images.to(self.configuration.device))
                    loss_sum += torch.nn.functional.cross_entropy(
                        logits, labels, reduction="sum"
                    ).item()
                    count += len(labels)
                    targets.append(labels.cpu())
                    predictions.append(logits.argmax(dim=1).cpu())
            if count == 0:
                raise ValueError("Validation dataset must be nonempty")
            actual, predicted = (
                torch.cat(targets).numpy(),
                torch.cat(predictions).numpy(),
            )
            return {
                "ce_loss_mean": loss_sum / count,
                **{
                    name: float(scorer(actual, predicted))
                    for name, scorer in self.scorers.items()
                },
            }
        finally:
            model.train(incoming_mode)

    def run(self) -> Path:
        """Train a fresh classifier.

        Returns:
            The final completed epoch's last.pt checkpoint.
        """
        return self.run_with_result().last_checkpoint

    def run_with_result(
        self, *, parent_run_id: str | None = None, defer_promotion: bool = False
    ) -> ExperimentResult:
        """Train with configured metric selection and optional MLflow tracking.

        Args:
            parent_run_id: Explicit active pipeline parent for a tracked child run.
            defer_promotion: Publish without changing the alias; the sweep owner
                compares its winner only after every classifier has succeeded.

        Returns:
            Best/last checkpoint metadata with the original configured objective.

        Raises:
            FloatingPointError: Training loss or the tracked metric is nonfinite.
            ValueError: Required data or the configured tracked metric is missing.
        """
        config = self.configuration
        context = (
            config.mlflow.start_run(Path(config.output_directory).name, parent_run_id=parent_run_id)
            if config.mlflow is not None else nullcontext()
        )
        with context as run:
            run_id = run.info.run_id if run is not None else None
            if config.mlflow is not None:
                mlflow.set_tags(config.mlflow.run_tags())
                mlflow.log_params(
                    {
                        "tracked_metric": config.tracked_metric,
                        "tracked_mode": config.effective_tracked_mode,
                    }
                )
            result = self._run_training(run_id)
            snapshot = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)
            with result.best_checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            self._write_model_card(result.best_checkpoint, snapshot, run_id, checksum)
            if config.mlflow is not None:
                prefix = (
                    "best_validation_"
                    if config.validation_dataset is not None
                    else "best_training_"
                )
                mlflow.log_metric(
                    prefix + config.tracked_metric,
                    result.best_value,
                    step=result.best_epoch,
                )
                mlflow.log_metric("best_epoch", result.best_epoch)
                if config.mlflow.registered_model_name is not None:
                    version = self._publish_best_model(result, snapshot, checksum)
                    result = replace(result, registered_model_version=version)
                    if not defer_promotion:
                        ClassifierChampionRegistry(config.mlflow, config.tracked_metric_definition, config.class_names).promote(result)
                else:
                    config.mlflow.log_best_artifacts(result.best_checkpoint)
            return result

    def _publish_best_model(self, result: ExperimentResult, snapshot: Mapping, checksum: str) -> str:
        """Restore best.pt independently of the final epoch and publish that trained classifier."""
        config = self.configuration
        checkpoint = result.best_checkpoint
        model = YoloClassifierModel.from_checkpoint_state(snapshot)

        def write_card(version, identity):
            """Complete local and run-level cards before registry alias promotion."""
            paths = self._write_model_card(
                checkpoint, snapshot, result.mlflow_run_id, checksum, version=version, provenance=identity,
            )
            config.mlflow.log_best_artifacts(checkpoint)
            return paths

        registry = ClassifierChampionRegistry(config.mlflow, config.tracked_metric_definition, config.class_names)
        return ModelPublication(config.mlflow).publish(
            model, best_epoch=result.best_epoch,
            validated=config.validation_dataset is not None,
            tags={**snapshot["provenance"], **config.mlflow.run_tags(), **registry.comparison_tags(result)},
            write_card=write_card, promote=False,
        )

    def _train_epoch(
        self, model: YoloClassifierModel, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Train one complete single-view epoch and return sample-weighted mean CE."""
        device = self.configuration.device
        model.train()
        total, count = 0.0, 0
        for labels, images in loader:
            labels, images = labels.to(device), images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training CE loss must be finite")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
        return total / count

    def _tracked_value(self, row: Mapping) -> float:
        """Read the configured objective without substituting training CE for validation."""
        metric = self.configuration.tracked_metric
        if "validation" in row:
            if metric not in row["validation"]:
                raise ValueError(f"Missing tracked validation metric '{metric}'")
            value = float(row["validation"][metric])
        else:
            value = row["mean"]
        if not math.isfinite(value):
            raise FloatingPointError("Tracked validation/training metric must be finite")
        return value

    def _run_training(self, run_id: str | None) -> ExperimentResult:
        """Own fresh optimizer state and atomically publish each complete epoch."""
        config = self.configuration
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        dataset, loader = self._build_loader()
        validation_loader = self._build_validation_loader()
        source = Path(config.initialization["encoder"]["checkpoint"]).resolve()
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance = {
            **self.encoder_provenance,
            "encoder_checkpoint": str(source),
            "encoder_checkpoint_sha256": checksum,
        }
        model = YoloClassifierModel.initialize(
            config.initialization,
            label_names=dataset.label_names,
            freeze_encoder=config.freeze_encoder,
            encoder_batchnorm=config.encoder_batchnorm,
            device=config.device,
        )
        try:
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), **config.optimizer
            )
            scheduler = config.scheduler.build(optimizer) if config.scheduler else None
            stopper = (
                EarlyStopping.from_mapping(config.early_stopping)
                if config.early_stopping
                else None
            )
            store = CheckpointStore(config.output_directory)
            document = self.document
            (store.output_directory / "config.yaml").write_text(
                yaml.safe_dump(document), encoding="utf-8"
            )
            history, best_value, best_epoch = [], None, None
            for epoch in range(1, config.epochs + 1):
                row = {
                    "epoch": epoch,
                    "mean": self._train_epoch(
                        model, tqdm(loader, desc=f"Epoch {epoch}"), optimizer
                    ),
                }
                if validation_loader is not None:
                    row["validation"] = self._validate(model, validation_loader)
                value = self._tracked_value(row)
                improved = best_value is None or (
                    value < best_value
                    if config.effective_tracked_mode == "min"
                    else value > best_value
                )
                if improved:
                    best_value, best_epoch = value, epoch
                if scheduler is not None:
                    scheduler.step(value)
                stopping = stopper(value) if stopper is not None else False
                row["next_learning_rate"] = optimizer.param_groups[0]["lr"]
                history.append(row)
                store.save(
                    {
                        **model.checkpoint_state(),
                        "epoch": epoch,
                        "history": history,
                        "configuration": document,
                        "provenance": provenance,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                        if scheduler
                        else None,
                    },
                    is_best=improved,
                )
                (store.output_directory / "history.yaml").write_text(
                    yaml.safe_dump(history), encoding="utf-8"
                )
                if run_id is not None:
                    mlflow.log_metric("training/ce_loss_mean", row["mean"], step=epoch)
                    for name, metric in row.get("validation", {}).items():
                        mlflow.log_metric(f"validation/{name}", metric, step=epoch)
                if stopping:
                    break
            best = store.output_directory / "best.pt"
            return ExperimentResult(
                output_directory=store.output_directory,
                last_checkpoint=store.output_directory / "last.pt",
                best_checkpoint=best,
                best_epoch=best_epoch,
                best_value=best_value,
                tracked_metric=config.tracked_metric,
                tracked_mode=config.effective_tracked_mode,
                validation_dataset=config.validation_dataset,
                mlflow_run_id=run_id,
            )
        finally:
            model.eval()

    def _write_model_card(
        self, checkpoint: Path, snapshot: Mapping, run_id: str | None, checksum: str,
        *, version: str | None = None, provenance: Mapping | None = None,
    ) -> tuple[Path, Path]:
        """Describe the selected snapshot rather than the last completed epoch."""
        config = self.configuration
        card = deepcopy(self.card_template)
        card["model"].update(
            name=(config.mlflow.registered_model_name if config.mlflow else None) or Path(config.output_directory).name,
            registry_version=version,
            base_model=str(config.initialization["encoder"]["checkpoint"]),
            task="image-classification",
        )
        card["classes"] = {
            index: name for name, index in snapshot["label_names"].items()
        }
        metrics = snapshot["history"][-1].get("validation", {})
        card["validation"].update(
            status="evaluated" if metrics else "not_validated",
            dataset=str(config.validation_dataset)
            if config.validation_dataset
            else None,
            epoch=snapshot["epoch"],
            metrics=metrics,
        )
        card["provenance"].update(
            **snapshot["provenance"],
            checkpoint=checkpoint.name,
            checkpoint_sha256=checksum,
            epoch=snapshot["epoch"],
            run_id=run_id,
            git_commit=_git_commit(),
            dataset_revision=config.mlflow.dataset_revision if config.mlflow else None,
        )
        card["provenance"].update(provenance or {})
        return write_model_card(card, checkpoint)
