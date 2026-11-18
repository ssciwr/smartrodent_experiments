"""Run the standalone multiview SupCon experiment from a YAML configuration."""

import argparse
import hashlib
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Real
from operator import gt, lt
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from smartrodent import config_utils
from smartrodent.experiment_results import ExperimentResult
# Re-export the existing configuration names for callers importing this script.
from smartrodent.experiment_support import (
    CheckpointStore,
    EarlyStopping,
    ImageTransforms,
    MlflowConfiguration,
    ModelPublication,
    SchedulerConfiguration,
    _git_commit,
    resolve_validation_metrics,
)
from smartrodent.model_card import (
    load_card_template,
    write_model_card,
)
from smartrodent.supcon_datasets import (
    ImageDataset,
    MultiViewImageDataset,
    collate_views,
)
from smartrodent.supcon_model import YoloSupConModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
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
    """

    train_dataset: Path
    output_directory: Path
    class_names: tuple[str, ...]
    transforms: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    model_name: str = "yolo26x-cls.pt"
    epochs: int = 5
    classes_per_batch: int = 4
    samples_per_class: int = 6
    samples_per_epoch: int = 100000
    workers: int = 20
    prefetch_factor: int = 4
    seed: int = 42
    device: str = "cuda"
    dropout: float = 0.5
    output_features: int = 128
    temperature: float = 0.1
    freeze: tuple[int, ...] = ()
    log_every: int = 50
    mlflow: MlflowConfiguration | None = None
    validation_dataset: Path | None = None
    validation_metrics: Mapping[str, Mapping[str, Any]] | None = None
    tracked_metric: str | None = None
    early_stopping: Mapping[str, Any] | None = None
    scheduler: SchedulerConfiguration | None = None
    model_card_template: str = "configs/model_card_template.yaml"
    tracked_mode: str | None = None

    def __post_init__(self):
        """Reject invalid settings before building datasets or writing outputs."""
        integers = (
            self.epochs,
            self.classes_per_batch,
            self.samples_per_class,
            self.samples_per_epoch,
            self.prefetch_factor,
            self.output_features,
            self.log_every,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in integers
        ):
            raise ValueError(
                "Training sizes, counts, and log_every must be positive integers"
            )
        if not isinstance(self.workers, int) or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if isinstance(self.class_names, str) or not self.class_names:
            raise ValueError("class_names must be a nonempty sequence of species names")
        if any(not isinstance(name, str) or not name for name in self.class_names):
            raise ValueError("class_names must contain nonempty strings")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if not 2 <= self.classes_per_batch <= len(self.class_names):
            raise ValueError(
                "classes_per_batch must be between 2 and the number of species"
            )
        if self.samples_per_epoch < self.batch_size:
            raise ValueError("samples_per_epoch must contain at least one full batch")
        if not isinstance(self.transforms, Mapping):
            raise TypeError("transforms must be a mapping")
        required_transforms = {
            "image_size",
            "resize",
            "random_adjust_sharpness",
            "random_grayscale",
            "random_perspective",
            "random_horizontal_flip",
            "random_resized_crop",
        }
        missing_transforms = required_transforms.difference(self.transforms)
        if missing_transforms:
            raise ValueError(
                "transforms is missing: " + ", ".join(sorted(missing_transforms))
            )
        image_size = self.transforms["image_size"]
        if (
            not isinstance(image_size, int)
            or isinstance(image_size, bool)
            or image_size < 1
        ):
            raise ValueError("transforms.image_size must be a positive integer")
        if not isinstance(self.optimizer, Mapping):
            raise TypeError("optimizer must be a mapping")
        learning_rate = self.optimizer.get("lr")
        if not isinstance(learning_rate, (int, float)) or not np.isfinite(
            learning_rate
        ):
            raise ValueError("optimizer.lr must be finite and positive")
        if learning_rate <= 0:
            raise ValueError("optimizer.lr must be finite and positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between zero and one")
        if any(not isinstance(index, int) or index < 0 for index in self.freeze):
            raise ValueError("freeze must contain nonnegative layer indices")
        if self.tracked_mode is not None and self.tracked_mode not in ("min", "max"):
            raise ValueError("tracked_mode must be 'min', 'max', or null")
        output = Path(self.output_directory).resolve()
        source = Path(self.train_dataset).resolve()
        if output.is_relative_to(source):
            raise ValueError("output_directory must be outside the training dataset")
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
            if not isinstance(self.tracked_metric, str) or not self.tracked_metric:
                raise ValueError("tracked_metric must be a nonempty metric name")
            available_metrics = {
                "supcon_loss_min",
                "supcon_loss_mean",
                "supcon_loss_std",
                "supcon_loss_max",
                *(self.validation_metrics or {}),
            }
            if self.tracked_metric not in available_metrics:
                raise ValueError(
                    f"tracked_metric '{self.tracked_metric}' is not available"
                )
        stopping_mode = None
        if self.early_stopping is not None:
            stopper = EarlyStopping.from_mapping(self.early_stopping)
            stopping_mode = "min" if stopper.comparison == "less" else "max"
        if self.scheduler is not None:
            if not isinstance(self.scheduler, SchedulerConfiguration):
                raise TypeError("scheduler must be a SchedulerConfiguration or null")
            if self.scheduler.parameters.get("min_lr", 0.0) > learning_rate:
                raise ValueError(
                    "scheduler.parameters.min_lr must not exceed optimizer.lr"
                )
            if stopping_mode is not None and self.scheduler.mode != stopping_mode:
                raise ValueError(
                    "scheduler mode must agree with early_stopping.comparison"
                )
        if self.tracked_mode is not None:
            if self.scheduler is not None and self.scheduler.mode != self.tracked_mode:
                raise ValueError("scheduler direction must agree with tracked_mode")
            if stopping_mode is not None and stopping_mode != self.tracked_mode:
                raise ValueError("early_stopping direction must agree with tracked_mode")

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
        return "min" if self.effective_tracked_metric.startswith("supcon_loss_") else "max"

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
        if not isinstance(values, Mapping):
            raise ValueError("Training configuration must be a YAML mapping")
        values = dict(values)
        values["train_dataset"] = Path(values["train_dataset"])
        values["output_directory"] = Path(values["output_directory"])
        if not isinstance(values["class_names"], list):
            raise ValueError("class_names must be a YAML list")
        values["class_names"] = tuple(values["class_names"])
        values["freeze"] = tuple(values.get("freeze", ()))
        for name in ("transforms", "optimizer"):
            if not isinstance(values.get(name), Mapping):
                raise TypeError(f"{name} must be a mapping")
            values[name] = dict(values[name])
        values["mlflow"] = MlflowConfiguration.from_mapping(values.get("mlflow"))
        values["scheduler"] = SchedulerConfiguration.from_mapping(
            values.get("scheduler")
        )
        validation_dataset = values.get("validation_dataset")
        if validation_dataset is not None:
            values["validation_dataset"] = Path(validation_dataset)
        early_stopping = values.get("early_stopping")
        if early_stopping is not None:
            if not isinstance(early_stopping, Mapping):
                raise TypeError("early_stopping must be a mapping")
            values["early_stopping"] = dict(early_stopping)
        return cls(**values)

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
        values = asdict(self)
        values["train_dataset"] = str(self.train_dataset)
        values["output_directory"] = str(self.output_directory)
        values["class_names"] = list(self.class_names)
        values["freeze"] = list(self.freeze)
        if self.validation_dataset is not None:
            values["validation_dataset"] = str(self.validation_dataset)
        return values

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
        self.configuration = configuration
        self.validation_metrics = resolve_validation_metrics(
            configuration.validation_metrics
        )
        # Fail on malformed templates before spending time on training.
        self.card_template = load_card_template(Path(configuration.model_card_template))

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
            model, best_epoch=best_epoch,
            validated=config.validation_dataset is not None, tags=source_tags,
            write_card=lambda version, identity: self._write_model_card(
                checkpoint, checkpoint_data, version,
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
        dataset = MultiViewImageDataset(
            Path(config.train_dataset),
            labels=config.class_names,
            transform=ImageTransforms(config.transforms).training(),
            size=config.transforms["image_size"],
        )
        labels = [dataset.label_names[path.parent.name] for path in dataset.images]
        if len(set(labels)) < config.classes_per_batch:
            raise ValueError(
                "Training dataset has too few nonempty classes for the requested batch"
            )
        sampler = MPerClassSampler(
            labels=labels,
            m=config.samples_per_class,
            batch_size=config.batch_size,
            length_before_new_iter=config.samples_per_epoch,
        )
        options = dict(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.batch_size,
            num_workers=config.workers,
            pin_memory=torch.device(config.device).type == "cuda",
            drop_last=True,
            collate_fn=collate_views,
            generator=torch.Generator().manual_seed(config.seed),
            persistent_workers=config.workers > 0,
        )
        if config.workers > 0:
            options["prefetch_factor"] = config.prefetch_factor
        return dataset, DataLoader(**options)

    def _build_validation_loader(self) -> DataLoader | None:
        """Build a deterministic single-view validation loader when configured."""
        config = self.configuration
        if config.validation_dataset is None:
            return None
        transform = ImageTransforms(config.transforms).validation()
        dataset = ImageDataset(
            config.validation_dataset,
            labels=config.class_names,
            transform=transform,
            size=config.transforms["image_size"],
        )
        if len(dataset) < config.batch_size:
            raise ValueError("Validation dataset must contain at least one full batch")
        labels = [dataset.label_names[path.parent.name] for path in dataset.images]
        validation_length = len(dataset) - len(dataset) % config.batch_size
        sampler = MPerClassSampler(
            labels=labels,
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
        options = dict(
            dataset=dataset,
            sampler=indices,
            batch_size=config.batch_size,
            num_workers=config.workers,
            pin_memory=torch.device(config.device).type == "cuda",
            drop_last=True,
            generator=torch.Generator().manual_seed(config.seed),
            persistent_workers=config.workers > 0,
        )
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
        config = self.configuration
        return CheckpointStore(config.output_directory).save(
            {
                "epoch": epoch,
                "architecture": deepcopy(model.full_model.model.yaml),
                "model_configuration": {
                    "dropout": config.dropout,
                    "output_features": config.output_features,
                    "num_classes": len(config.class_names),
                    "freeze": list(config.freeze),
                },
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "label_names": dict(model.label_names),
                "configuration": config.to_dict(),
                "history": list(history),
            },
            is_best=is_best,
        )

    def _run_training(
        self,
    ) -> tuple[Path, Path, float, int, YoloSupConModel]:
        """Train and return checkpoints, tracked-metric results, and the live model."""
        config = self.configuration
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        dataset, loader = self._build_loader()
        validation_loader = self._build_validation_loader()
        model = YoloSupConModel(
            config.model_name,
            dropout=config.dropout,
            output_features=config.output_features,
            num_classes=len(config.class_names),
        ).to(config.device)
        model.requires_grad_(True)
        for index in config.freeze:
            model.full_model.model.model[index].requires_grad_(False)
        model.train(True)
        model.label_names = dict(dataset.label_names)
        model.training_configuration = config.to_dict()
        model.full_model.model.names = {
            index: name for name, index in model.label_names.items()
        }
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), **config.optimizer
        )
        scheduler = (
            config.scheduler.build(optimizer) if config.scheduler is not None else None
        )
        criterion = SupConLoss(temperature=config.temperature)
        early_stopping = (
            EarlyStopping.from_mapping(config.early_stopping)
            if config.early_stopping is not None
            else None
        )
        output = Path(config.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False)
        )
        print(
            f"{len(loader)} batches/epoch; {config.batch_size} source images and "
            f"{2 * config.batch_size} views/batch on {config.device}"
        )
        history = []
        best_checkpoint = None
        tracked_metric = config.effective_tracked_metric
        tracked_mode = config.effective_tracked_mode
        is_improvement = lt if tracked_mode == "min" else gt
        best_metric_value = float("inf") if tracked_mode == "min" else float("-inf")
        best_epoch = 0
        return_live_model = False
        try:
            for epoch in range(1, config.epochs + 1):
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
                statistics = {
                    "epoch": epoch,
                    "min": values.min().item(),
                    "mean": values.mean().item(),
                    "std": values.std(correction=1 if len(values) > 1 else 0).item(),
                    "max": values.max().item(),
                    "learning_rate": learning_rate,
                }
                validation = (
                    self._validate(model, validation_loader, criterion)
                    if validation_loader is not None
                    else {}
                )
                if validation_loader is not None:
                    if tracked_metric not in validation:
                        raise ValueError(f"Validation is missing tracked metric '{tracked_metric}'")
                    statistics["validation"] = validation
                    tracked_value = validation[tracked_metric]
                else:
                    tracked_value = statistics["mean"]
                if not isinstance(tracked_value, Real) or not np.isfinite(tracked_value):
                    raise FloatingPointError(f"Nonfinite tracked validation metric '{tracked_metric}'")
                if scheduler is not None:
                    # Step after validation, before saving both optimizer and scheduler:
                    # the checkpoint then holds the LR ready for the next epoch.
                    scheduler.step(tracked_value)
                next_learning_rate = optimizer.param_groups[0]["lr"]
                statistics["next_learning_rate"] = next_learning_rate
                history.append(statistics)
                print(
                    f"Epoch: {epoch}, {statistics['min']:.6f}, "
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
                        + ", ".join(
                            f"{name}={value:.6f}" for name, value in validation.items()
                        )
                    )
                is_best = is_improvement(tracked_value, best_metric_value)
                checkpoint = self._save_checkpoint(
                    model, optimizer, epoch, history, scheduler, is_best=is_best
                )
                (output / "history.yaml").write_text(
                    yaml.safe_dump(history, sort_keys=False)
                )
                if config.mlflow is not None:
                    mlflow_metrics = {
                        "supcon_loss_min": statistics["min"],
                        "supcon_loss_mean": statistics["mean"],
                        "supcon_loss_std": statistics["std"],
                        "supcon_loss_max": statistics["max"],
                        "learning_rate": learning_rate,
                        "next_learning_rate": next_learning_rate,
                    }
                    mlflow_metrics.update(
                        {
                            f"validation/{name}": value
                            for name, value in validation.items()
                        }
                    )
                    mlflow.log_metrics(mlflow_metrics, step=epoch)
                if is_best:
                    best_metric_value = tracked_value
                    best_checkpoint = checkpoint.with_name("best.pt")
                    best_epoch = epoch
                if early_stopping is not None and early_stopping(tracked_value):
                    print(
                        f"Stopping early: {tracked_metric} did not improve for "
                        f"{early_stopping.patience} epochs"
                    )
                    break

            if best_checkpoint is None:
                raise RuntimeError("Training completed without producing a checkpoint")
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

    def run_with_result(self, *, parent_run_id: str | None = None) -> ExperimentResult:
        """Train once and report the best epoch selected by the configured metric.

        Args:
            parent_run_id: Explicit active pipeline parent for a tracked child run.

        Returns:
            Completed run metadata, including best/last checkpoints and an optional
            MLflow run ID. Training-only runs remain valid standalone results.

        Raises:
            ValueError: Batch composition or configured tracked metrics are invalid.
            FloatingPointError: Training or tracked validation values are nonfinite.
        """
        config = self.configuration
        if config.mlflow is None:
            model = None
            try:
                final_checkpoint, best_checkpoint, best_value, best_epoch, model = (
                    self._run_training()
                )
                checkpoint_data = torch.load(
                    best_checkpoint, map_location="cpu", weights_only=True
                )
                self._write_model_card(best_checkpoint, checkpoint_data)
                return self._build_result(final_checkpoint, best_checkpoint, best_value, best_epoch)
            finally:
                if model is not None:
                    model.finalize()

        source_tags = config.mlflow.run_tags()
        model = None
        with config.mlflow.start_run(Path(config.output_directory).name, parent_run_id=parent_run_id) as run:
            client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
            try:
                mlflow.set_tags(source_tags)
                mlflow.log_params(config.mlflow_parameters())
                final_checkpoint, best_checkpoint, best_value, best_epoch, model = (
                    self._run_training()
                )
                if config.validation_dataset is not None:
                    mlflow.log_metric(
                        f"best_validation_{config.effective_tracked_metric}",
                        best_value,
                        step=best_epoch,
                    )
                else:
                    mlflow.log_metric(
                        "best_supcon_loss_mean", best_value, step=best_epoch
                    )
                mlflow.log_metric("best_epoch", best_epoch)
                mlflow.log_artifact(str(best_checkpoint), artifact_path="best_model")
                if config.mlflow.registered_model_name is not None:
                    self._register_best_model(
                        model,
                        best_checkpoint,
                        client,
                        source_tags,
                        best_epoch,
                    )
                else:
                    checkpoint_data = torch.load(
                        best_checkpoint, map_location="cpu", weights_only=True
                    )
                    for path in self._write_model_card(
                        best_checkpoint,
                        checkpoint_data,
                        provenance={**source_tags, "run_id": run.info.run_id},
                    ):
                        mlflow.log_artifact(str(path), artifact_path="best_model")
            finally:
                if model is not None:
                    model.finalize()
        return self._build_result(
            final_checkpoint, best_checkpoint, best_value, best_epoch, run.info.run_id
        )

    def _build_result(
        self,
        last_checkpoint: Path,
        best_checkpoint: Path,
        best_value: float,
        best_epoch: int,
        mlflow_run_id: str | None = None,
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
        card = deepcopy(self.card_template)
        name = config.mlflow.registered_model_name if config.mlflow else None
        card["model"].update(
            name=name or Path(config.output_directory).name,
            registry_version=version,
            base_model=config.model_name,
            task="contrastive-encoder",
        )
        card["classes"] = dict(
            sorted(
                (
                    (index, name)
                    for name, index in checkpoint_data["label_names"].items()
                )
            )
        )
        # The snapshot's last history row belongs to that snapshot, not the final epoch.
        metrics = checkpoint_data["history"][-1].get("validation", {})
        card["validation"].update(
            status="evaluated" if metrics else "not_validated",
            dataset=(
                str(config.validation_dataset) if config.validation_dataset else None
            ),
            epoch=checkpoint_data["epoch"],
            metrics=metrics,
        )
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        card["provenance"].update(
            checkpoint=checkpoint.name,
            checkpoint_sha256=checksum,
            epoch=checkpoint_data["epoch"],
            run_id=None,
            model_id=None,
            git_commit=_git_commit(),
            dataset_revision=config.mlflow.dataset_revision if config.mlflow else None,
        )
        card["provenance"].update(provenance or {})
        return write_model_card(card, checkpoint)


def _load_run_mappings(config_path: Path) -> list[dict]:
    """Load custom YAML tags and expand them into concrete run mappings."""
    configuration = yaml.load(
        config_path.read_text(encoding="utf-8"),
        Loader=config_utils.get_loader(),
    )
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


def main(config_path: Path) -> list[Path]:
    """Train every experiment expanded from a YAML configuration.

    Args:
        config_path: YAML configuration; data paths are relative to the current directory.

    Returns:
        Final checkpoint path for each expanded run.
    """
    configurations = load_training_configurations(config_path)
    checkpoints = []
    for run_number, configuration in enumerate(configurations, start=1):
        print(
            f"Starting run {run_number}/{len(configurations)} in "
            f"{configuration.output_directory}"
        )
        checkpoints.append(SupConExperiment(configuration).run())
    return checkpoints


def _parse_args() -> argparse.Namespace:
    """Parse the positional experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to a standalone SupCon YAML configuration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    for final_checkpoint in main(_parse_args().config):
        print("Final checkpoint:", final_checkpoint)
