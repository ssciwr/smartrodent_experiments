"""Standalone cross-entropy training for portable YOLO classifiers."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

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
)
from smartrodent.model_card import load_card_template, write_model_card
from smartrodent.supcon_datasets import ImageDataset


@dataclass(frozen=True, kw_only=True)
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
    class_names: tuple[str, ...]
    initialization: Mapping
    transforms: Mapping
    optimizer: Mapping
    validation_dataset: Path | None = None
    freeze_encoder: bool = True
    encoder_batchnorm: str = "eval"
    epochs: int = 1
    batch_size: int = 32
    workers: int = 0
    device: str = "cpu"
    seed: int = 42
    tracked_metric: str = "ce_loss_mean"
    tracked_mode: str | None = None
    validation_metrics: Mapping | None = None
    scheduler: SchedulerConfiguration | None = None
    early_stopping: Mapping | None = None
    mlflow: MlflowConfiguration | None = None
    model_card_template: str = "configs/model_card_template.yaml"

    def __post_init__(self):
        """Validate settings before any model I/O or output creation."""
        for name, minimum in (
            ("epochs", 1),
            ("batch_size", 1),
            ("workers", 0),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        YoloClassifierModel._label_mapping(self.class_names)
        if not isinstance(self.freeze_encoder, bool):
            raise ValueError("freeze_encoder must be a boolean")
        if self.encoder_batchnorm not in ("eval", "adapt"):
            raise ValueError("encoder_batchnorm must be eval or adapt")
        self._validate_initialization()
        if not isinstance(self.transforms, Mapping):
            raise ValueError("transforms must be a mapping")
        image_size = self.transforms.get("image_size")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size < 1
        ):
            raise ValueError("transforms.image_size must be a positive integer")
        # Constructing transforms validates native options without consuming images.
        ImageTransforms(self.transforms).training()
        if not isinstance(self.optimizer, Mapping):
            raise ValueError("optimizer must be an AdamW parameter mapping")
        lr = self.optimizer.get("lr")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or lr <= 0
        ):
            raise ValueError("optimizer.lr must be finite and positive")
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], **self.optimizer)
        resolve_validation_metrics(self.validation_metrics)
        if "ce_loss_mean" in (self.validation_metrics or {}):
            raise ValueError("validation_metrics cannot override ce_loss_mean")
        if self.tracked_metric not in {
            "ce_loss_mean",
            *(self.validation_metrics or {}),
        }:
            raise ValueError(f"tracked_metric '{self.tracked_metric}' is not available")
        if self.tracked_mode is not None and self.tracked_mode not in ("min", "max"):
            raise ValueError("tracked_mode must be min, max, or null")
        if self.validation_dataset is None and (
            self.validation_metrics
            or self.tracked_metric != "ce_loss_mean"
            or self.scheduler is not None
            or self.early_stopping is not None
        ):
            raise ValueError(
                "Validation objectives, scheduling, and stopping require validation_dataset"
            )
        if self.scheduler is not None:
            if self.scheduler.mode != self.effective_tracked_mode:
                raise ValueError("scheduler direction must agree with tracked_mode")
            if self.scheduler.parameters.get("min_lr", 0) > lr:
                raise ValueError("scheduler min_lr must not exceed optimizer.lr")
        if self.early_stopping is not None:
            stopper = EarlyStopping.from_mapping(self.early_stopping)
            mode = "min" if stopper.comparison == "less" else "max"
            if mode != self.effective_tracked_mode:
                raise ValueError(
                    "early_stopping direction must agree with tracked_mode"
                )
        output = Path(self.output_directory).resolve()
        for source in (self.train_dataset, self.validation_dataset):
            if source is not None and output.is_relative_to(Path(source).resolve()):
                raise ValueError("output_directory must be outside both input datasets")
        self._validate_source_collisions(output)

    def _validate_source_collisions(self, output: Path) -> None:
        """Protect source bytes from artifact writes, temporary cleanup, and aliases."""
        names = (
            "last.pt",
            "best.pt",
            ".last.pt.tmp",
            ".best.pt.tmp",
            "config.yaml",
            "history.yaml",
            "best.model_card.yaml",
            "best.model_card.md",
        )
        for node in self.initialization.values():
            if node.get("checkpoint") is None:
                continue
            source = Path(node["checkpoint"]).resolve()
            for name in names:
                artifact = output / name
                same_file = (
                    source.exists() and artifact.exists() and source.samefile(artifact)
                )
                if source == artifact.resolve() or same_file:
                    raise ValueError(
                        f"Output artifact {artifact} collides with source checkpoint {source}"
                    )

    def _validate_initialization(self) -> None:
        """Reject ambiguous or misspelled source/head nodes before loading weights."""
        nodes = self.initialization
        if not isinstance(nodes, Mapping) or set(nodes) != {"encoder", "head"}:
            raise ValueError("initialization requires exactly encoder and head nodes")
        encoder, head = nodes["encoder"], nodes["head"]
        if not isinstance(encoder, Mapping) or set(encoder) != {"source", "checkpoint"}:
            raise ValueError("encoder requires source and checkpoint")
        if encoder["source"] not in ("supcon", "ultralytics", "classifier"):
            raise ValueError("Invalid encoder source")
        if (
            not isinstance(encoder["checkpoint"], (str, Path))
            or not str(encoder["checkpoint"]).strip()
        ):
            raise ValueError("encoder checkpoint must be a local path")
        if (
            not isinstance(head, Mapping)
            or not {"replace", "source"}.issubset(head)
            or set(head) - {"replace", "source", "checkpoint"}
        ):
            raise ValueError(
                "head requires replace/source and optional checkpoint; unexpected head keys"
            )
        if head["replace"] not in ("linear", "full"):
            raise ValueError("Invalid head replace boundary")
        if head["source"] not in ("random", "yolo_checkpoint"):
            raise ValueError("Invalid head source")
        if head["source"] == "yolo_checkpoint":
            if (
                not isinstance(head.get("checkpoint"), (str, Path))
                or not str(head["checkpoint"]).strip()
            ):
                raise ValueError("head checkpoint is required")
        elif head.get("checkpoint") is not None:
            raise ValueError("Random head must not specify a checkpoint")

    @property
    def effective_tracked_mode(self) -> str:
        """Return explicit direction or loss-min/score-max defaults."""
        return self.tracked_mode or (
            "min" if self.tracked_metric == "ce_loss_mean" else "max"
        )

    @classmethod
    def from_mapping(cls, values: Mapping) -> "ClassifierConfiguration":
        """Parse one concrete configuration mapping.

        Args:
            values: YAML-compatible standalone classifier settings.

        Returns:
            Validated classifier settings.
        """
        if not isinstance(values, Mapping):
            raise ValueError("Classifier configuration must be a mapping")
        values = deepcopy(dict(values))
        for name in ("train_dataset", "validation_dataset", "output_directory"):
            if values.get(name) is not None:
                values[name] = Path(values[name])
        names = values.get("class_names")
        if isinstance(names, list):
            values["class_names"] = tuple(names)
        values["scheduler"] = SchedulerConfiguration.from_mapping(
            values.get("scheduler")
        )
        values["mlflow"] = MlflowConfiguration.from_mapping(values.get("mlflow"))
        return cls(**values)

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
        values = asdict(self)
        for name in ("train_dataset", "validation_dataset", "output_directory"):
            if values[name] is not None:
                values[name] = str(values[name])
        values["class_names"] = list(self.class_names)
        for node in values["initialization"].values():
            if node.get("checkpoint") is not None:
                node["checkpoint"] = str(node["checkpoint"])
        values["model_card_template"] = str(values["model_card_template"])
        return values


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

    def _dataset(self, path: Path, *, training: bool) -> ImageDataset:
        """Reuse the existing class mapping and single-view dataset, rejecting empty classes."""
        dataset = ImageDataset(
            Path(path),
            self.configuration.class_names,
            transform=self.transforms.training()
            if training
            else self.transforms.validation(),
        )
        counts = Counter(image.parent.name for image in dataset.images)
        if any(counts[name] == 0 for name in self.configuration.class_names):
            raise ValueError("Every requested dataset class must be nonempty")
        return dataset

    def _build_loader(self) -> tuple[ImageDataset, DataLoader]:
        """Build ordinary shuffled, single-view epochs retaining every source image."""
        config = self.configuration
        dataset = self._dataset(config.train_dataset, training=True)
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
        if config.validation_dataset is None:
            return None
        dataset = self._dataset(config.validation_dataset, training=False)
        return DataLoader(
            dataset,
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

    def run_with_result(self, *, parent_run_id: str | None = None) -> ExperimentResult:
        """Train with configured metric selection and optional MLflow tracking.

        Args:
            parent_run_id: Explicit active pipeline parent for a tracked child run.

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
                    self._publish_best_model(result)
                else:
                    config.mlflow.log_best_artifacts(result.best_checkpoint)
            return result

    def _publish_best_model(self, result: ExperimentResult) -> None:
        """Restore best.pt independently of the final epoch and publish that trained classifier."""
        config = self.configuration
        checkpoint = result.best_checkpoint
        model = YoloClassifierModel.from_checkpoint(checkpoint)
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)

        def write_card(version, identity):
            """Complete local and run-level cards before registry alias promotion."""
            paths = self._write_model_card(checkpoint, result.mlflow_run_id, version=version, provenance=identity)
            config.mlflow.log_best_artifacts(checkpoint)
            return paths

        ModelPublication(config.mlflow).publish(
            model, best_epoch=result.best_epoch,
            validated=config.validation_dataset is not None,
            tags={**snapshot["provenance"], **config.mlflow.run_tags()},
            write_card=write_card,
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
        _, loader = self._build_loader()
        validation_loader = self._build_validation_loader()
        encoder, head = config.initialization["encoder"], config.initialization["head"]
        source = Path(encoder["checkpoint"]).resolve()
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance = {
            **self.encoder_provenance,
            "encoder_checkpoint": str(source),
            "encoder_checkpoint_sha256": checksum,
        }
        model = YoloClassifierModel.initialize(
            source,
            source_format=encoder["source"],
            class_names=config.class_names,
            head_replace=head["replace"],
            head_source=head["source"],
            head_checkpoint=head.get("checkpoint"),
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
            document = config.to_dict()
            (store.output_directory / "config.yaml").write_text(
                yaml.safe_dump(document), encoding="utf-8"
            )
            history, best_value, best_epoch = [], None, None
            for epoch in range(1, config.epochs + 1):
                row = {
                    "epoch": epoch,
                    "mean": self._train_epoch(model, loader, optimizer),
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
            self._write_model_card(best, run_id)
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
        self, checkpoint: Path, run_id: str | None, *, version: str | None = None,
        provenance: Mapping | None = None,
    ) -> tuple[Path, Path]:
        """Describe the selected snapshot rather than the last completed epoch."""
        config = self.configuration
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
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
