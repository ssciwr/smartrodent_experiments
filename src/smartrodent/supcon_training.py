"""Configure, prepare, and run standalone multiview SupCon experiments."""

import hashlib
import random
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from copy import deepcopy
from numbers import Real
from operator import gt, lt
from pathlib import Path
from typing import Annotated, ClassVar

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from smartrodent import config_utils
from smartrodent.datasets import (
    ImageDataset,
    MultiViewImageDataset,
    collate_views,
)
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
from smartrodent.model_card import (
    load_training_template,
    write_training_card,
)
from smartrodent.onnx_export import OnnxExporter
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.training_validation import (
    ClassNames,
    Count,
    MappingInput,
    Mode,
    Name,
    NonnegativeInt,
    OnnxExportSettings,
    OptimizerSettings,
    PathSetting,
    PositiveNumber,
    Probability,
    Seed,
    StoppingSettings,
    TransformSettings,
    ValidationMetrics,
    parse_settings,
    settings_dict,
    validate_output_paths,
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TrainingConfiguration:
    """Configure the standalone, full-precision contrastive training experiment.

    Args:
        train_dataset: Directory containing one subdirectory per species.
        output_directory: Separate directory for configuration, history, checkpoints.
        class_names: Species directories to include.
        transforms: Image size and torchvision transform parameters.
        optimizer: AdamW parameters.
        model_name: Pretrained classification checkpoint or model YAML.
        epochs: Number of training epochs.
        classes_per_batch: Distinct classes selected per source-image batch.
        samples_per_class: Source images per selected class.
        samples_per_epoch: Source-image selections per epoch, rounded down to batches.
        workers: CPU DataLoader worker count.
        prefetch_factor: Batches prefetched per worker when workers are enabled.
        seed: Python, NumPy, and PyTorch random seed.
        device: PyTorch training device.
        dropout: Dropout probability in the projection head.
        output_features: Contrastive embedding dimension.
        temperature: SupCon loss temperature.
        freeze: Backbone layer indices to freeze after enabling gradients.
        log_every: Number of steps between progress loss updates.
        mlflow: Optional MLflow tracking settings.
        validation_dataset: Optional directory containing validation class folders.
        validation_metrics: Scikit-learn metrics computed from validation predictions.
        tracked_metric: Validation metric used to select the best model.
        tracked_mode: Optional ``min``/``max`` direction; omitted preserves legacy defaults.
        early_stopping: Optional patience, tolerance, and comparison settings.
        scheduler: Optional validation-driven learning-rate scheduler settings.
        model_card_template: YAML template used for the selected checkpoint's card.
        onnx_export: Optional ONNX export of the best checkpoint's pooled encoder features;
            omitted/null exports nothing.
    """

    _native_metrics: ClassVar[tuple[str, ...]] = (
        "supcon_loss_min",
        "supcon_loss_mean",
        "supcon_loss_std",
        "supcon_loss_max",
    )

    train_dataset: Path
    output_directory: Path
    class_names: ClassNames
    transforms: TransformSettings
    optimizer: OptimizerSettings
    model_name: PathSetting = "yolo26x-cls.pt"
    epochs: Count = 5
    classes_per_batch: Annotated[Count, Field(ge=2)] = 4
    samples_per_class: Count = 6
    samples_per_epoch: Count = 100000
    workers: NonnegativeInt = 20
    prefetch_factor: Count = 4
    seed: Seed = 42
    device: Name = "cuda"
    dropout: Probability = 0.5
    output_features: Count = 128
    temperature: PositiveNumber = 0.1
    freeze: tuple[NonnegativeInt, ...] = ()
    log_every: Count = 50
    mlflow: MappingInput[MlflowConfiguration] | None = None
    validation_dataset: Path | None = None
    validation_metrics: ValidationMetrics | None = None
    tracked_metric: Name | None = None
    early_stopping: StoppingSettings | None = None
    scheduler: MappingInput[SchedulerConfiguration] | None = None
    model_card_template: PathSetting = "configs/model_card_template.yaml"
    tracked_mode: Mode | None = None
    onnx_export: MappingInput[OnnxExportSettings] | None = None

    def __post_init__(self):
        """Reject invalid settings before building datasets or writing outputs."""
        collisions = set(self.validation_metrics or {}) & set(self._native_metrics)
        if collisions:
            raise ValueError(
                f"validation_metrics cannot override native SupCon losses: {sorted(collisions)}"
            )
        if self.classes_per_batch > len(self.class_names):
            raise ValueError(
                "classes_per_batch must be between 2 and the number of species"
            )
        if self.samples_per_epoch < self.batch_size:
            raise ValueError("samples_per_epoch must contain at least one full batch")
        validate_output_paths(
            [self.output_directory],
            (self.train_dataset, self.validation_dataset),
            (self.model_name, self.model_card_template),
        )
        if self.validation_dataset is None:
            if (
                self.validation_metrics is not None
                or self.tracked_metric is not None
                or self.tracked_mode is not None
                or self.early_stopping is not None
                or self.scheduler is not None
            ):
                raise ValueError(
                    "validation_metrics, tracked_metric, tracked_mode, early_stopping, and scheduler require "
                    "validation_dataset"
                )
        elif self.tracked_metric is not None:
            available_metrics = {
                *self._native_metrics,
                *(self.validation_metrics or {}),
            }
            if self.tracked_metric not in available_metrics:
                raise ValueError(
                    f"tracked_metric '{self.tracked_metric}' is not available"
                )
        validate_stopping(self, self.tracked_mode)

    @property
    def batch_size(self) -> int:
        """Return the number of source images per batch, before doubling views."""
        return self.classes_per_batch * self.samples_per_class

    @property
    def effective_tracked_metric(self) -> str:
        """Return the configured metric or the mean validation SupCon loss."""
        return self.tracked_metric or "supcon_loss_mean"

    @property
    def effective_tracked_mode(self) -> str:
        """Return explicit direction or preserve the legacy loss/score defaults."""
        if self.tracked_mode is not None:
            return self.tracked_mode
        return "min" if self.effective_tracked_metric in self._native_metrics else "max"

    @property
    def tracked_metric_definition(self) -> dict:
        """Return the effective scoring definition used for sweep comparability.

        Returns:
            The native loss/statistic definition or the configured scorer and options.
        """
        metric = self.effective_tracked_metric
        if metric in self._native_metrics:
            return {
                "function": "pytorch_metric_learning.losses.SupConLoss",
                "temperature": self.temperature,
                "statistic": metric,
            }
        else:
            return deepcopy(self.validation_metrics[metric])

    @classmethod
    def from_mapping(cls, values: Mapping) -> "TrainingConfiguration":
        """Build validated settings from one concrete configuration mapping.

        Args:
            values: Configuration values after expanding custom YAML tags.

        Returns:
            Validated experiment settings.

        Raises:
            ValueError: The mapping contains an invalid class-name collection.
        """
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfiguration":
        """Read YAML that expands to exactly one training configuration.

        Args:
            path: Training configuration file.

        Returns:
            Validated experiment settings.

        Raises:
            ValueError: The document expands to more than one run.
        """
        run_mappings = _load_run_mappings(path)
        if len(run_mappings) != 1:
            raise ValueError(
                "TrainingConfiguration.from_yaml requires exactly one run; "
                "use load_training_configurations for sweep configurations"
            )
        return cls.from_mapping(run_mappings[0])

    def to_dict(self) -> dict:
        """Return plain, checkpoint-safe settings suitable for YAML output."""
        return settings_dict(self)

    def mlflow_parameters(self) -> dict[str, str | int | float]:
        """Return concrete hyperparameters suitable for MLflow parameter logging."""
        parameters = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "transforms": str(dict(self.transforms)),
            "optimizer": str(dict(self.optimizer)),
            "classes_per_batch": self.classes_per_batch,
            "samples_per_class": self.samples_per_class,
            "samples_per_epoch": self.samples_per_epoch,
            "workers": self.workers,
            "prefetch_factor": self.prefetch_factor,
            "seed": self.seed,
            "device": self.device,
            "dropout": self.dropout,
            "output_features": self.output_features,
            "temperature": self.temperature,
            "freeze": ",".join(str(index) for index in self.freeze),
        }
        if self.validation_dataset is not None:
            parameters["validation_dataset"] = str(self.validation_dataset)
            parameters["tracked_metric"] = self.effective_tracked_metric
            parameters["tracked_mode"] = self.effective_tracked_mode
        if self.early_stopping is not None:
            parameters.update(
                {
                    f"early_stopping_{name}": value
                    for name, value in self.early_stopping.items()
                }
            )
        if self.scheduler is not None:
            parameters["scheduler_name"] = self.scheduler.name
            parameters.update(
                {
                    f"scheduler_{name}": value
                    for name, value in self.scheduler.parameters.items()
                }
            )
        return parameters


class SupConExperiment:
    """Train the independent notebook experiment and save portable checkpoints.

    Args:
        configuration: Dataset, training, and output settings.
    """

    def __init__(self, configuration: TrainingConfiguration):
        """Prepare verified transforms, scorers, templates, and dataset indices once."""
        self.configuration = configuration
        self.document = configuration.to_dict()
        self.validation_metrics = resolve_validation_metrics(
            configuration.validation_metrics
        )
        # Fail on malformed templates before spending time on training.
        self.card_template = load_training_template(
            Path(configuration.model_card_template)
        )
        self.transforms = ImageTransforms(configuration.transforms)
        self.exporter = (
            OnnxExporter(
                configuration.onnx_export, configuration.transforms["image_size"]
            )
            if configuration.onnx_export is not None
            else None
        )
        self.training_dataset = MultiViewImageDataset(
            configuration.train_dataset,
            labels=configuration.class_names,
            transform=self.transforms.training(),
        )
        if (
            sum(count > 0 for count in self.training_dataset.class_counts.values())
            < configuration.classes_per_batch
        ):
            raise ValueError(
                "Training dataset has too few nonempty classes for the requested batch"
            )
        self.validation_dataset = (
            ImageDataset(
                configuration.validation_dataset,
                labels=self.training_dataset.label_names,
                transform=self.transforms.validation(),
            )
            if configuration.validation_dataset is not None
            else None
        )
        if self.validation_dataset is not None:
            if len(self.validation_dataset) < configuration.batch_size:
                raise ValueError(
                    "Validation dataset must contain at least one full batch"
                )
            if (
                sum(
                    count > 0 for count in self.validation_dataset.class_counts.values()
                )
                < configuration.classes_per_batch
            ):
                raise ValueError(
                    "Validation dataset has too few nonempty classes for the requested batch"
                )

    def _register_best_model(
        self,
        model: YoloSupConModel,
        checkpoint: Path,
        client: MlflowClient,
        source_tags: Mapping[str, str],
        best_epoch: int,
    ) -> str:
        """Register the best trained model and promote validated versions."""
        config = self.configuration
        mlflow_config = config.mlflow
        if mlflow_config is None or mlflow_config.registered_model_name is None:
            raise RuntimeError("Model registration requires a registered model name")

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.to("cpu")
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        version = ModelPublication(mlflow_config, client).publish(
            model,
            best_epoch=best_epoch,
            validated=config.validation_dataset is not None,
            tags=source_tags,
            write_card=lambda version, identity: self._write_model_card(
                checkpoint,
                checkpoint_data,
                version,
                provenance={**source_tags, **identity},
            ),
        )
        if config.validation_dataset is not None:
            print(
                f"Promoted {mlflow_config.registered_model_name} version {version} "
                f"to @{mlflow_config.champion_alias}"
            )
        return version

    def _build_loader(self) -> tuple[MultiViewImageDataset, DataLoader]:
        """Build label-aligned P-K batches with two views per selected image."""
        config = self.configuration
        dataset = self.training_dataset
        sampler = MPerClassSampler(
            labels=dataset.labels,
            m=config.samples_per_class,
            batch_size=config.batch_size,
            length_before_new_iter=config.samples_per_epoch,
        )
        options = {
            "dataset": dataset,
            "sampler": sampler,
            "batch_size": config.batch_size,
            "num_workers": config.workers,
            "pin_memory": torch.device(config.device).type == "cuda",
            "drop_last": True,
            "collate_fn": collate_views,
            "generator": torch.Generator().manual_seed(config.seed),
            "persistent_workers": config.workers > 0,
        }
        if config.workers > 0:
            options["prefetch_factor"] = config.prefetch_factor
        return dataset, DataLoader(**options)

    def _build_validation_loader(self) -> DataLoader | None:
        """Build a deterministic single-view validation loader when configured."""
        config = self.configuration
        dataset = self.validation_dataset
        if dataset is None:
            return None
        validation_length = len(dataset) - len(dataset) % config.batch_size
        sampler = MPerClassSampler(
            labels=dataset.labels,
            m=config.samples_per_class,
            batch_size=config.batch_size,
            length_before_new_iter=validation_length,
        )
        # PML draws from NumPy's global RNG. Materialize one seeded cohort and
        # restore that RNG immediately, so validation neither resamples each epoch
        # nor changes the training sampler's future draws. Reuse PML's P–K policy.
        numpy_state = np.random.get_state()
        try:
            np.random.seed(config.seed)
            indices = list(sampler)
        finally:
            np.random.set_state(numpy_state)
        options = {
            "dataset": dataset,
            "sampler": indices,
            "batch_size": config.batch_size,
            "num_workers": config.workers,
            "pin_memory": torch.device(config.device).type == "cuda",
            "drop_last": True,
            "generator": torch.Generator().manual_seed(config.seed),
            "persistent_workers": config.workers > 0,
        }
        if config.workers > 0:
            options["prefetch_factor"] = config.prefetch_factor
        return DataLoader(**options)

    def _validate(
        self,
        model: YoloSupConModel,
        loader: DataLoader,
        criterion: SupConLoss,
    ) -> dict[str, float]:
        """Predict the validation set and compute every configured metric."""
        true_labels = []
        predicted_labels = []
        losses = []
        model.train(False)
        try:
            with torch.inference_mode():
                for labels, images in loader:
                    images = images.to(self.configuration.device, non_blocking=True)
                    device_labels = labels.to(
                        self.configuration.device, non_blocking=True
                    )
                    classifier_output, embeddings = model(images)
                    losses.append(criterion(embeddings, device_labels).detach())
                    scores = (
                        classifier_output[0]
                        if isinstance(classifier_output, tuple)
                        else classifier_output
                    )
                    true_labels.extend(labels.tolist())
                    predicted_labels.extend(scores.argmax(dim=1).cpu().tolist())
        finally:
            model.embeddings = None
            model.train(True)

        targets = np.asarray(true_labels)
        predictions = np.asarray(predicted_labels)
        loss_values = torch.stack(losses).float().cpu()
        results = {
            "supcon_loss_min": loss_values.min().item(),
            "supcon_loss_mean": loss_values.mean().item(),
            "supcon_loss_std": loss_values.std(
                correction=1 if len(loss_values) > 1 else 0
            ).item(),
            "supcon_loss_max": loss_values.max().item(),
        }
        results.update(
            {
                name: float(scorer(targets, predictions))
                for name, scorer in self.validation_metrics.items()
            }
        )
        nonfinite = [name for name, value in results.items() if not np.isfinite(value)]
        if nonfinite:
            raise FloatingPointError(
                f"Nonfinite validation metrics: {', '.join(nonfinite)}"
            )
        return results

    def _save_checkpoint(
        self, model, optimizer, epoch, history, scheduler=None, *, is_best=False
    ) -> Path:
        """Assemble SupCon state and publish it through the shared checkpoint store."""
        return self.checkpoint_store.save(
            {
                "epoch": epoch,
                "architecture": deepcopy(model.full_model.model.yaml),
                "model_configuration": self.model_configuration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "label_names": dict(model.label_names),
                "configuration": self.document,
                "history": list(history),
            },
            is_best=is_best,
        )

    def _train_epoch(self, model, loader, optimizer, criterion, epoch: int) -> dict:
        """Train one multiview epoch and return its finite loss statistics."""
        config = self.configuration
        learning_rate = optimizer.param_groups[0]["lr"]
        losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}")
        for step, (labels, images) in enumerate(progress):
            labels = labels.to(config.device, non_blocking=True)
            images = images.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            model.embeddings = None
            losses.append(loss.detach())
            if step % config.log_every == 0:
                progress.set_postfix(sc_loss=f"{loss.item():.4f}")
        values = torch.stack(losses).float().cpu()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite loss in epoch {epoch}")
        return {
            "epoch": epoch,
            "min": values.min().item(),
            "mean": values.mean().item(),
            "std": values.std(correction=1 if len(values) > 1 else 0).item(),
            "max": values.max().item(),
            "learning_rate": learning_rate,
        }

    def _report_epoch(self, statistics: Mapping) -> None:
        """Report a completed epoch to the console and optional tracking run."""
        learning_rate = statistics["learning_rate"]
        next_learning_rate = statistics["next_learning_rate"]
        validation = statistics.get("validation", {})
        print(
            f"Epoch: {statistics['epoch']}, {statistics['min']:.6f}, "
            f"{statistics['mean']:.6f} +/- {statistics['std']:.6f}, "
            f"{statistics['max']:.6f}; lr={learning_rate:.6g}"
        )
        if next_learning_rate < learning_rate:
            print(
                f"Learning rate reduced: {learning_rate:.6g} -> {next_learning_rate:.6g} for the next epoch"
            )
        if validation:
            print(
                "Validation: "
                + ", ".join(f"{name}={value:.6f}" for name, value in validation.items())
            )
        if self.configuration.mlflow is not None:
            metrics = {
                f"supcon_loss_{name}": statistics[name]
                for name in ("min", "mean", "std", "max")
            }
            metrics.update(
                learning_rate=learning_rate, next_learning_rate=next_learning_rate
            )
            metrics.update(
                {f"validation/{name}": value for name, value in validation.items()}
            )
            mlflow.log_metrics(metrics, step=statistics["epoch"])

    def _run_training(
        self, epoch_callback: Callable[[int, float], None] | None = None
    ) -> tuple[Path, Path, float, int, YoloSupConModel]:
        """Train and return checkpoints, tracked-metric results, and the live model.

        The optional epoch callback observes each saved epoch's tracked value; any
        exception it raises (e.g. a pruning request) stops training unchanged.
        """
        config = self.configuration
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        dataset, loader = self._build_loader()
        validation_loader = self._build_validation_loader()
        self.model_configuration = {
            "dropout": config.dropout,
            "output_features": config.output_features,
            "num_classes": len(dataset.label_names),
            "freeze": list(config.freeze),
        }
        model = YoloSupConModel(config.model_name, **self.model_configuration)
        return_live_model = False
        try:
            model.to(config.device)
            model.train(True)
            model.label_names = dict(dataset.label_names)
            model.training_configuration = self.document
            model.full_model.model.names = {
                index: name for name, index in model.label_names.items()
            }
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), **config.optimizer
            )
            scheduler = (
                config.scheduler.build(optimizer)
                if config.scheduler is not None
                else None
            )
            criterion = SupConLoss(temperature=config.temperature)
            early_stopping = (
                EarlyStopping.from_mapping(config.early_stopping)
                if config.early_stopping is not None
                else None
            )
            self.checkpoint_store = CheckpointStore(config.output_directory)
            output = self.checkpoint_store.output_directory
            (output / "config.yaml").write_text(
                yaml.safe_dump(self.document, sort_keys=False)
            )
            print(
                f"{len(loader)} batches/epoch; {config.batch_size} source images and "
                f"{2 * config.batch_size} views/batch on {config.device}"
            )
            history = []
            best_checkpoint = output / "best.pt"
            tracked_metric = config.effective_tracked_metric
            tracked_mode = config.effective_tracked_mode
            is_improvement = lt if tracked_mode == "min" else gt
            best_metric_value = float("inf") if tracked_mode == "min" else float("-inf")
            best_epoch = 0
            for epoch in range(1, config.epochs + 1):
                statistics = self._train_epoch(
                    model, loader, optimizer, criterion, epoch
                )
                validation = (
                    self._validate(model, validation_loader, criterion)
                    if validation_loader is not None
                    else {}
                )
                if validation_loader is not None:
                    if tracked_metric not in validation:
                        raise ValueError(
                            f"Validation is missing tracked metric '{tracked_metric}'"
                        )
                    statistics["validation"] = validation
                    tracked_value = validation[tracked_metric]
                else:
                    tracked_value = statistics["mean"]
                if not isinstance(tracked_value, Real) or not np.isfinite(
                    tracked_value
                ):
                    raise FloatingPointError(
                        f"Nonfinite tracked validation metric '{tracked_metric}'"
                    )
                if scheduler is not None:
                    # Step after validation, before saving both optimizer and scheduler:
                    # the checkpoint then holds the LR ready for the next epoch.
                    scheduler.step(tracked_value)
                statistics["next_learning_rate"] = optimizer.param_groups[0]["lr"]
                history.append(statistics)
                is_best = is_improvement(tracked_value, best_metric_value)
                checkpoint = self._save_checkpoint(
                    model, optimizer, epoch, history, scheduler, is_best=is_best
                )
                (output / "history.yaml").write_text(
                    yaml.safe_dump(history, sort_keys=False)
                )
                self._report_epoch(statistics)
                if is_best:
                    best_metric_value = tracked_value
                    best_epoch = epoch
                if epoch_callback is not None:
                    epoch_callback(epoch, float(tracked_value))
                if early_stopping is not None and early_stopping(tracked_value):
                    print(
                        f"Stopping early: {tracked_metric} did not improve for "
                        f"{early_stopping.patience} epochs"
                    )
                    break

            return_live_model = True
            return checkpoint, best_checkpoint, best_metric_value, best_epoch, model
        finally:
            model.train(False)
            if not return_live_model:
                model.finalize()

    def run(self) -> Path:
        """Train in FP32, preserving the standalone checkpoint-path API.

        Returns:
            Path of the final epoch checkpoint.

        Raises:
            ValueError: Batch composition or configured tracked metrics are invalid.
            FloatingPointError: Training or tracked validation values are nonfinite.
        """
        return self.run_with_result().last_checkpoint

    def run_with_result(
        self,
        *,
        parent_run_id: str | None = None,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> ExperimentResult:
        """Train once and report the best epoch selected by the configured metric.

        Args:
            parent_run_id: Explicit active pipeline parent for a tracked child run.
            epoch_callback: Optional observer called as ``(epoch, tracked_value)``
                after each epoch's checkpoint is saved. Exceptions it raises stop
                training and propagate unchanged, before any publication.

        Returns:
            Completed run metadata, including best/last checkpoints and an optional
            MLflow run ID. Training-only runs remain valid standalone results.

        Raises:
            ValueError: Batch composition or configured tracked metrics are invalid.
            FloatingPointError: Training or tracked validation values are nonfinite.
        """
        config = self.configuration
        context = (
            config.mlflow.start_run(
                config.output_directory.name, parent_run_id=parent_run_id
            )
            if config.mlflow is not None
            else nullcontext()
        )
        model = None
        with context as run:
            run_id = run.info.run_id if run is not None else None
            source_tags = (
                config.mlflow.run_tags()
                if config.mlflow is not None
                else {"git_commit": _git_commit()}
            )
            try:
                if config.mlflow is not None:
                    mlflow.set_tags(source_tags)
                    mlflow.log_params(config.mlflow_parameters())
                final_checkpoint, best_checkpoint, best_value, best_epoch, model = (
                    self._run_training(epoch_callback)
                )
                # Export before any publication: a failed export must leave nothing published.
                if self.exporter is not None:
                    onnx_model = self.exporter.encoder(best_checkpoint)
                else:
                    onnx_model = None
                if config.mlflow is not None:
                    metric = (
                        f"best_validation_{config.effective_tracked_metric}"
                        if config.validation_dataset is not None
                        else "best_supcon_loss_mean"
                    )
                    mlflow.log_metric(metric, best_value, step=best_epoch)
                    mlflow.log_metric("best_epoch", best_epoch)
                    mlflow.log_artifact(
                        str(best_checkpoint), artifact_path="best_model"
                    )
                    if onnx_model is not None:
                        mlflow.log_artifact(str(onnx_model), artifact_path="best_model")
                if (
                    config.mlflow is not None
                    and config.mlflow.registered_model_name is not None
                ):
                    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
                    self._register_best_model(
                        model, best_checkpoint, client, source_tags, best_epoch
                    )
                else:
                    checkpoint_data = torch.load(
                        best_checkpoint, map_location="cpu", weights_only=True
                    )
                    cards = self._write_model_card(
                        best_checkpoint,
                        checkpoint_data,
                        provenance={**source_tags, "run_id": run_id},
                    )
                    if config.mlflow is not None:
                        for path in cards:
                            mlflow.log_artifact(str(path), artifact_path="best_model")
                return self._build_result(
                    final_checkpoint,
                    best_checkpoint,
                    best_value,
                    best_epoch,
                    run_id,
                    onnx_model,
                )
            finally:
                if model is not None:
                    model.finalize()

    def _build_result(
        self,
        last_checkpoint: Path,
        best_checkpoint: Path,
        best_value: float,
        best_epoch: int,
        mlflow_run_id: str | None = None,
        onnx_model: Path | None = None,
    ) -> ExperimentResult:
        """Report the same configured metric and snapshots used by the training loop."""
        config = self.configuration
        return ExperimentResult(
            output_directory=Path(config.output_directory),
            last_checkpoint=last_checkpoint,
            best_checkpoint=best_checkpoint,
            best_epoch=best_epoch,
            tracked_metric=config.effective_tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            best_value=best_value,
            validation_dataset=config.validation_dataset,
            mlflow_run_id=mlflow_run_id,
            onnx_model=onnx_model,
        )

    def _write_model_card(
        self,
        checkpoint: Path,
        checkpoint_data: dict,
        version: str | None = None,
        *,
        provenance: Mapping[str, str] | None = None,
    ) -> tuple[Path, Path]:
        """Fill the template with the selected snapshot's metadata and save it."""
        config = self.configuration
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return write_training_card(
            self.card_template,
            checkpoint,
            checkpoint_data,
            model_details={
                "name": (config.mlflow.registered_model_name if config.mlflow else None)
                or config.output_directory.name,
                "registry_version": version,
                "base_model": str(config.model_name),
                "task": "contrastive-encoder",
            },
            validation_dataset=config.validation_dataset,
            provenance={
                "checkpoint_sha256": checksum,
                "dataset_revision": (
                    config.mlflow.dataset_revision if config.mlflow else None
                ),
                **(provenance or {}),
            },
        )


def _load_run_mappings(config_path: Path) -> list[dict]:
    """Load custom YAML tags and expand them into concrete run mappings."""
    configuration = config_utils.load_yaml(config_path)
    if not isinstance(configuration, Mapping):
        raise ValueError("Training configuration must be a YAML mapping")
    return config_utils.ConfigHandler(dict(configuration)).run_configs


def load_training_configurations(config_path: Path) -> list[TrainingConfiguration]:
    """Load, expand, and validate every configured training run.

    Sweep runs receive stable subdirectories so that checkpoints and histories
    from later runs cannot overwrite earlier results.

    Args:
        config_path: YAML configuration with optional custom sweep tags.

    Returns:
        Concrete, validated training configurations in sweep-product order.
    """
    run_mappings = _load_run_mappings(config_path)
    multiple_runs = len(run_mappings) > 1
    configurations = []
    for run_number, run_mapping in enumerate(run_mappings, start=1):
        values = dict(run_mapping)
        if multiple_runs:
            output_directory = Path(values["output_directory"])
            values["output_directory"] = output_directory / f"run_{run_number:03d}"
        configurations.append(TrainingConfiguration.from_mapping(values))
    if len({config.temperature for config in configurations}) > 1:
        raise ValueError("SupCon temperature must remain fixed within a sweep")
    return configurations


def run_training(config_path: Path) -> list[Path]:
    """Train every experiment expanded from a YAML configuration.

    Args:
        config_path: YAML configuration; data paths are relative to the current directory.

    Returns:
        Final checkpoint path for each expanded run.
    """
    experiments = [
        SupConExperiment(config) for config in load_training_configurations(config_path)
    ]
    checkpoints = []
    for run_number, experiment in enumerate(experiments, start=1):
        print(
            f"Starting run {run_number}/{len(experiments)} in "
            f"{experiment.configuration.output_directory}"
        )
        checkpoints.append(experiment.run())
    return checkpoints
