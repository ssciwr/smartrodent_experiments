"""Run the standalone multiview SupCon experiment from a YAML configuration."""

import argparse
import functools
import random
import subprocess
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import sklearn.metrics
import torch
import yaml
from mlflow import MlflowClient
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from smartrodent import config_utils
from smartrodent.supcon_datasets import (
    ImageDataset,
    MultiViewImageDataset,
    collate_views,
)
from smartrodent.supcon_model import YoloSupConModel
from torch.utils.data import DataLoader
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
from tqdm.auto import tqdm


@dataclass(frozen=True)
class MlflowConfiguration:
    """Configure MLflow tracking for the standalone SupCon experiment.

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

    def run_tags(self) -> dict[str, str]:
        """Return source revisions that make a tracked run reproducible."""
        tags = {"git_commit": _git_commit()}
        if self.dataset_revision is not None:
            tags["dataset_revision"] = self.dataset_revision
        return tags


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
        tracked_metric: Validation metric whose highest value selects the best model.
        early_stopping: Optional patience, tolerance, and comparison settings.
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
        output = Path(self.output_directory).resolve()
        source = Path(self.train_dataset).resolve()
        if output.is_relative_to(source):
            raise ValueError("output_directory must be outside the training dataset")
        if self.validation_dataset is None:
            if (
                self.validation_metrics is not None
                or self.tracked_metric is not None
                or self.early_stopping is not None
            ):
                raise ValueError(
                    "validation_metrics, tracked_metric, and early_stopping require "
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
        if self.early_stopping is not None:
            EarlyStopping.from_mapping(self.early_stopping)

    @property
    def batch_size(self) -> int:
        """Return the number of source images per batch, before doubling views."""
        return self.classes_per_batch * self.samples_per_class

    @property
    def effective_tracked_metric(self) -> str:
        """Return the configured metric or the mean validation SupCon loss."""
        return self.tracked_metric or "supcon_loss_mean"

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
        if self.early_stopping is not None:
            parameters.update(
                {
                    f"early_stopping_{name}": value
                    for name, value in self.early_stopping.items()
                }
            )
        return parameters


class EarlyStopping:
    """Stop after repeated failures to improve beyond a tolerance threshold.

    The tolerance is subtracted from the previous value before applying the
    configured comparison. For ``less``, a value therefore improves only when
    it is lower than ``previous_value - tolerance``.
    """

    def __init__(self, patience: int, tolerance: float, comparison: str):
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
        self.compare = (
            (lambda x, y: x > y) if comparison == "greater" else (lambda x, y: x < y)
        )
        self.history: list[float] = []

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EarlyStopping":
        """Build an early stopper from its YAML-compatible configuration."""
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

    def __call__(self, value: float) -> bool:
        """Record a value and report whether patience has been exhausted."""
        if not np.isfinite(value):
            raise ValueError("Early-stopping values must be finite")
        if not self.history:
            self.history.append(value)
            return False

        if self.compare(value, self.history[-1] - self.tol):
            self.current_patience = self.patience
        else:
            self.current_patience -= 1

        self.history.append(value)
        return self.current_patience == 0


class SupConExperiment:
    """Train the independent notebook experiment and save portable checkpoints.

    Args:
        configuration: Dataset, training, and output settings.
    """

    def __init__(self, configuration: TrainingConfiguration):
        self.configuration = configuration
        self.validation_metrics = self._resolve_validation_metrics()

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
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=mlflow_config.model_artifact_name,
            registered_model_name=mlflow_config.registered_model_name,
            serialization_format="pickle",
        )

        version = model_info.registered_model_version
        if version is None:
            raise RuntimeError("MLflow did not return a registered model version")
        version = str(version)
        validation_status = (
            "passed" if config.validation_dataset is not None else "not_validated"
        )
        version_tags = {
            **source_tags,
            "validation_status": validation_status,
            "best_epoch": str(best_epoch),
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                mlflow_config.registered_model_name,
                version,
                key,
                value,
            )

        if validation_status == "passed":
            client.set_registered_model_alias(
                mlflow_config.registered_model_name,
                mlflow_config.champion_alias,
                version,
            )
            print(
                f"Promoted {mlflow_config.registered_model_name} version {version} "
                f"to @{mlflow_config.champion_alias}"
            )
        return version

    def _resolve_validation_metrics(
        self,
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        """Resolve configured sklearn metrics and reject invalid specs early."""
        specs = self.configuration.validation_metrics
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

    def _build_transform(self) -> Compose:
        """Reproduce the notebook's independently sampled augmentation pipeline."""
        settings = self.configuration.transforms
        size = settings["image_size"]
        return Compose(
            [
                PILToTensor(),
                ToDtype(torch.float32, scale=True),
                Resize((size, size), **settings["resize"]),
                RandomAdjustSharpness(**settings["random_adjust_sharpness"]),
                RandomGrayscale(**settings["random_grayscale"]),
                RandomPerspective(**settings["random_perspective"]),
                RandomHorizontalFlip(**settings["random_horizontal_flip"]),
                RandomResizedCrop(
                    (size, size),
                    **settings["random_resized_crop"],
                ),
            ]
        )

    def _build_loader(self) -> tuple[MultiViewImageDataset, DataLoader]:
        """Build label-aligned P-K batches with two views per selected image."""
        config = self.configuration
        dataset = MultiViewImageDataset(
            Path(config.train_dataset),
            labels=config.class_names,
            transform=self._build_transform(),
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
        transform = Compose(
            [
                PILToTensor(),
                ToDtype(torch.float32, scale=True),
                Resize(
                    (config.transforms["image_size"], config.transforms["image_size"]),
                    **config.transforms["resize"],
                ),
            ]
        )
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
        options = dict(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.batch_size,
            num_workers=config.workers,
            pin_memory=torch.device(config.device).type == "cuda",
            drop_last=True,
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

    def _save_checkpoint(self, model, optimizer, epoch, history) -> Path:
        """Save weights, architecture, label mapping, optimizer, and scalar history."""
        config = self.configuration
        path = Path(config.output_directory) / f"epoch_{epoch:04d}.pt"
        torch.save(
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
                "label_names": dict(model.label_names),
                "configuration": config.to_dict(),
                "history": list(history),
            },
            path,
        )
        return path

    def _run_training(
        self,
    ) -> tuple[Path, Path, float, int, YoloSupConModel]:
        """Train and return checkpoints, selection data, and the live model."""
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
        best_selection_value = float("-inf")
        best_epoch = 0
        return_live_model = False
        try:
            for epoch in range(1, config.epochs + 1):
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
                }
                validation = (
                    self._validate(model, validation_loader, criterion)
                    if validation_loader is not None
                    else {}
                )
                if validation:
                    statistics["validation"] = validation
                history.append(statistics)
                print(
                    f"Epoch: {epoch}, {statistics['min']:.6f}, "
                    f"{statistics['mean']:.6f} +/- {statistics['std']:.6f}, "
                    f"{statistics['max']:.6f}"
                )
                if validation:
                    print(
                        "Validation: "
                        + ", ".join(
                            f"{name}={value:.6f}" for name, value in validation.items()
                        )
                    )
                checkpoint = self._save_checkpoint(model, optimizer, epoch, history)
                (output / "history.yaml").write_text(
                    yaml.safe_dump(history, sort_keys=False)
                )
                if config.mlflow is not None:
                    mlflow_metrics = {
                        "supcon_loss_min": statistics["min"],
                        "supcon_loss_mean": statistics["mean"],
                        "supcon_loss_std": statistics["std"],
                        "supcon_loss_max": statistics["max"],
                    }
                    mlflow_metrics.update(
                        {
                            f"validation/{name}": value
                            for name, value in validation.items()
                        }
                    )
                    mlflow.log_metrics(mlflow_metrics, step=epoch)
                if validation:
                    tracked_metric = config.effective_tracked_metric
                    tracked_value = validation[tracked_metric]
                    selection_value = (
                        -tracked_value
                        if tracked_metric.startswith("supcon_loss_")
                        else tracked_value
                    )

                else:
                    tracked_value = statistics["mean"]
                    selection_value = -tracked_value
                if selection_value > best_selection_value:
                    best_selection_value = selection_value
                    best_metric_value = tracked_value
                    best_checkpoint = checkpoint
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
        """Train in FP32 and optionally track the run with MLflow.

        Returns:
            Path of the final epoch checkpoint.

        Raises:
            ValueError: Batch composition is incompatible with the source dataset.
            FloatingPointError: An epoch contains nonfinite losses.
        """
        config = self.configuration
        if config.mlflow is None:
            model = None
            try:
                final_checkpoint, _, _, _, model = self._run_training()
                return final_checkpoint
            finally:
                if model is not None:
                    model.finalize()

        config.mlflow.configure()
        client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
        source_tags = config.mlflow.run_tags()
        model = None
        with mlflow.start_run(run_name=Path(config.output_directory).name):
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
            finally:
                if model is not None:
                    model.finalize()
        return final_checkpoint


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
