"""Frozen-encoder YOLO head experiments using Ultralytics classification training."""

import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import field, replace
from functools import cached_property
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils.torch_utils import strip_optimizer

from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_model import YoloClassifierModel
from smartrodent.datasets import ImageDataset
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import ImageTransforms, MlflowConfiguration, ModelPublication, _git_commit, resolve_validation_metrics
from smartrodent.model_card import load_training_template, write_training_card
from smartrodent.training_validation import (
    CheckpointPath, ClassifierTrainingSettings, ClassNames, Count, EncoderBatchnorm,
    InitializationSettings, MappingInput, Mode, Name, NonnegativeInt, Seed,
    TransformSettings, ValidationMetrics, parse_settings, settings_dict, validate_output_paths,
)


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class ClassifierConfiguration:
    """Configure native YOLO head training with independently owned shared settings.

    Args:
        train_dataset: Training split containing species directories.
        output_directory: Experiment destination outside input splits.
        class_names: Species selected for alphabetical dataset IDs.
        initialization: Encoder format/path and head replacement/source policy.
        transforms: Prepared torchvision preprocessing/augmentation options.
        train_kwargs: Native optimizer, learning-rate schedule, stopping and other options.
        validation_dataset: Optional existing validation split.
        encoder_batchnorm: Fixed statistics (eval) or statistics adaptation (adapt).
        epochs: Maximum epochs.
        batch_size: Source images per ordinary single-view batch.
        workers: Loader worker count.
        device: Single PyTorch device.
        seed: Initialization/training seed.
        tracked_metric: Full-split CE or configured sklearn alias.
        tracked_mode: Explicit min/max or inferred loss-min/score-max direction.
        validation_metrics: Optional sklearn label metric specifications.
        mlflow: Optional owned tracking/registry settings.
        model_card_template: Template used for the selected best checkpoint.
    """

    train_dataset: Path
    output_directory: Path
    class_names: ClassNames
    initialization: InitializationSettings
    transforms: TransformSettings
    train_kwargs: MappingInput[ClassifierTrainingSettings] = field(default_factory=dict)
    validation_dataset: Path | None = None
    encoder_batchnorm: EncoderBatchnorm = "eval"
    epochs: Count = 1
    batch_size: Count = 32
    workers: NonnegativeInt = 0
    device: Name = "cpu"
    seed: Seed = 42
    tracked_metric: Name = "ce_loss_mean"
    tracked_mode: Mode | None = None
    validation_metrics: ValidationMetrics | None = None
    mlflow: MappingInput[MlflowConfiguration] | None = None
    model_card_template: CheckpointPath = "configs/model_card_template.yaml"

    def __post_init__(self):
        """Validate external options once before preparing native resources or outputs."""
        owned = {"model", "data", "task", "mode", "project", "name", "save_dir", "exist_ok", "freeze", "resume",
                 "pretrained", "save", "epochs", "batch", "imgsz", "workers", "device", "seed", "val", "split",
                 "classes", "single_cls", "cache", "augmentations", "auto_augment", "erasing", "scale", "crop_fraction",
                 "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr", "multi_scale", "distill_model", "quantize", "compile",
                 "fraction", "tile"}
        conflicts = owned & self.train_kwargs.keys()
        if conflicts:
            raise ValueError(f"train_kwargs cannot override owned or unsupported settings: {sorted(conflicts)}")
        get_cfg(overrides=dict(self.train_kwargs))
        torch.device(self.device)  # Only a single explicit device, not a DDP subprocess/backend.
        if self.initialization["head"]["replace"] == "linear" and self.train_kwargs.get("dropout", 0):
            raise ValueError("dropout requires replace=full; the pre-linear encoder is frozen")
        reserved = {"ce_loss_mean", "fitness", "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5",
                    "train/loss", "epoch", "time"}
        metric_conflicts = {name for name in (self.validation_metrics or {})
                            if name in reserved or (name.startswith("lr/pg") and name[5:].isdigit())}
        if metric_conflicts:
            raise ValueError(f"validation_metrics cannot override reserved names: {sorted(metric_conflicts)}")
        if self.tracked_metric not in {"ce_loss_mean", *(self.validation_metrics or {})}:
            raise ValueError(f"tracked_metric '{self.tracked_metric}' is not available")
        if self.validation_dataset is None and (self.validation_metrics or self.tracked_metric != "ce_loss_mean"):
            raise ValueError("Validation objectives require validation_dataset")
        validate_output_paths(
            [self.output_directory], (self.train_dataset, self.validation_dataset),
            [self.model_card_template, *(node.get("checkpoint") for node in self.initialization.values())],
            native_directories=[self.output_directory],
        )

    @property
    def effective_tracked_mode(self) -> str:
        """Return explicit direction or the loss-min/score-max default."""
        return self.tracked_mode or ("min" if self.tracked_metric == "ce_loss_mean" else "max")

    @property
    def tracked_metric_definition(self) -> dict:
        """Return the exact diagnostic/scorer definition used for comparison."""
        if self.tracked_metric == "ce_loss_mean":
            return {"function": "torch.nn.functional.cross_entropy", "reduction": "mean"}
        else:
            return deepcopy(dict(self.validation_metrics[self.tracked_metric]))

    @classmethod
    def from_mapping(cls, values: Mapping) -> "ClassifierConfiguration":
        """Parse a concrete configuration.

        Args:
            values: Standalone classifier settings.

        Returns:
            Independently owned, validated settings.
        """
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassifierConfiguration":
        """Read a concrete YAML configuration.

        Args:
            path: Configuration filename; use the sweep loader for custom tags.

        Returns:
            Settings for one experiment.
        """
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        """Return an independent YAML-compatible configuration dictionary."""
        return settings_dict(self)


class _ClassificationData(torch.utils.data.Dataset):
    """Adapt an already indexed image dataset without rescanning or writing native caches."""

    def __init__(self, dataset):
        """Retain the prepared dataset without copying its index or transforms."""
        self.dataset = dataset

    def __len__(self):
        """Return the number of indexed images."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Return the native classification batch fields for one image."""
        label, image = self.dataset[index]
        return {"img": image, "cls": label, "im_file": str(self.dataset.images[index])}


class _HeadValidator(ClassificationValidator):
    """Add configured diagnostics to the native validation loop, not another scoring pass."""

    def __init__(self, experiment, *args, **kwargs):
        """Bind the experiment's already resolved scorers to native validation."""
        self.experiment = experiment
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        """Initialize native metrics and full-split unweighted CE accumulation."""
        super().init_metrics(model)
        self.ce_total = torch.zeros((), device=self.device)

    def postprocess(self, preds):
        """Keep raw logits until the matching batch targets reach update_metrics."""
        self.raw_logits = preds[1]
        return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        """Reuse native prediction collection and accumulate the configured CE diagnostic."""
        super().update_metrics(preds, batch)
        # Native val/loss averages batch means (and can use class weights).
        # Preserve ce_loss_mean's existing, unweighted whole-split definition.
        self.ce_total += torch.nn.functional.cross_entropy(self.raw_logits, batch["cls"], reduction="sum")
        self.raw_logits = None

    def get_stats(self):
        """Return native statistics plus full-split CE and sklearn label metrics."""
        metrics = super().get_stats()
        actual = torch.cat(self.targets).numpy()
        predicted = torch.cat(self.pred)[:, 0].numpy()
        return {**metrics, "ce_loss_mean": self.ce_total.item() / len(actual),
                **{name: float(scorer(actual, predicted)) for name, scorer in self.experiment.scorers.items()}}


class _HeadTrainer(ClassificationTrainer):
    """Supply prepared model/data and selection policy to the native training lifecycle."""

    def __init__(self, experiment, model):
        """Configure native execution around the supplied model without reloading it."""
        self.experiment, self.prepared_model = experiment, model
        config = experiment.configuration
        device = torch.device(config.device)
        output = config.output_directory.resolve()
        last_layer = len(model.network.model) - 1
        frozen = list(range(last_layer))
        if config.initialization["head"]["replace"] == "linear":
            # Native freeze accepts module-name suffixes as well as whole-layer indices.
            frozen.append(f"{last_layer}.conv")
        overrides = {
            **config.train_kwargs, "model": str(config.initialization["encoder"]["checkpoint"]), "data": str(config.train_dataset.parent),
            "project": str(output.parent), "name": output.name,
            "exist_ok": True, "epochs": config.epochs, "batch": config.batch_size, "workers": config.workers,
            "device": str(device.index or 0) if device.type == "cuda" else str(device), "seed": config.seed,
            # Native's val flag schedules validate(), including training-only selection.
            "imgsz": config.transforms["image_size"], "freeze": frozen, "val": True,
        }
        super().__init__(overrides=overrides)
        # Native setup_model recognizes an existing nn.Module: no second factory,
        # checkpoint load or head reset can replace the selected SupCon encoder.
        self.model = model.network
        self.model.args = vars(self.args)
        self.model.transforms = experiment.validation_transform
        if config.initialization["head"]["replace"] == "full":
            self.model.model[-1].drop.p = self.args.dropout
        for event, callbacks in self.callbacks.items():
            self.callbacks[event] = [fn for fn in callbacks if fn.__module__ != "ultralytics.utils.callbacks.mlflow"]
        self.best_epoch, self.best_metrics = 0, {}
        self.add_callback("on_fit_epoch_end", self._log_epoch)

    def get_dataset(self):
        """Use shared, verified splits instead of native discovery/downloads."""
        config = self.experiment.configuration
        names = {index: name for name, index in self.experiment.training_dataset.label_names.items()}
        return {"train": str(config.train_dataset), "val": config.validation_dataset, "names": names,
                "nc": len(names), "channels": 3}

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Use prepared torchvision transforms and deterministic, complete validation."""
        dataset = self.experiment.training_dataset if mode == "train" else self.experiment.validation_dataset
        if dataset is None:
            # Native progress reporting reads test_loader.dataset even with val=False.
            return torch.utils.data.DataLoader(())
        return build_dataloader(_ClassificationData(dataset), batch_size, self.args.workers, shuffle=mode == "train",
                                rank=-1, drop_last=False, device=self.device)

    def get_class_counts(self):
        """Return the established dataset's counts for native optional class weighting."""
        return np.asarray(list(self.experiment.training_dataset.class_counts.values()), dtype=np.float32)

    def get_validator(self):
        """Build the native validator extension with the existing prepared scorers."""
        return _HeadValidator(self.experiment, self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _model_train(self):
        """Keep frozen encoder Dropout/BN policy when native training changes modes."""
        self.prepared_model.train()

    def validate(self):
        """Select native fitness from the configured objective, never native top-1 defaults."""
        config = self.experiment.configuration
        metrics = self.validator(self) if config.validation_dataset is not None else {"ce_loss_mean": float(self.tloss["loss"])}
        if config.tracked_metric not in metrics:
            raise ValueError(f"Missing tracked validation metric '{config.tracked_metric}'")
        value = float(metrics[config.tracked_metric])
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite tracked validation metric '{config.tracked_metric}'")
        fitness = value if config.effective_tracked_mode == "max" else -value
        metrics.pop("fitness", None)
        # Match native checkpoint publication, which also retains the latest epoch on an exact tie.
        if self.best_fitness is None or fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch, self.best_metrics = self.epoch + 1, dict(metrics)
        return metrics, fitness

    def _log_epoch(self, trainer):
        """Log only this owned run's completed native epochs."""
        if self.experiment.configuration.mlflow is not None:
            metrics = {"training/ce_loss_mean": float(self.tloss["loss"])}
            if self.experiment.validation_dataset is not None:
                metrics.update({f"validation/{name}": value for name, value in self.metrics.items()})
            mlflow.log_metrics(metrics, step=self.epoch + 1)

    def final_eval(self):
        """Finalize native files without rescoring or rediscovering shared dataset paths."""
        last = strip_optimizer(self.last)
        best = strip_optimizer(self.best, updates={"train_results": last.get("train_results")})
        if not best:
            raise ValueError(f"Native best checkpoint could not be finalized: {self.best}")
        model = best["model"]
        self.best_metadata = {"label_names": {name: index for index, name in model.names.items()},
                              "provenance": model.provenance}
        self.metrics = self.best_metrics


class ClassifierExperiment:
    """Own prepared inputs and publication around native head training.

    Args:
        configuration: Validated classifier settings.
        encoder_provenance: Optional encoder-selection metadata from orchestration.
        head_cache: Optional sweep-owned read-only head checkpoint cache.
    """

    def __init__(self, configuration: ClassifierConfiguration, *, encoder_provenance: Mapping | None = None, head_cache=None):
        """Prepare known inputs without creating run outputs or initializing an encoder."""
        self.configuration = configuration
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        self.transforms = ImageTransforms(configuration.transforms)
        self.validation_transform = self.transforms.validation()
        self.scorers = resolve_validation_metrics(configuration.validation_metrics)
        self.card_template = load_training_template(Path(configuration.model_card_template))
        self.training_dataset = self._dataset(configuration.train_dataset, training=True)
        self.validation_dataset = self._dataset(configuration.validation_dataset, training=False) if configuration.validation_dataset is not None else None
        self.head_donor = YoloClassifierModel.prepare_head(configuration.initialization["head"], self.training_dataset.label_names, cache=head_cache)

    @cached_property
    def champion_registry(self) -> ClassifierChampionRegistry | None:
        """Return the owned registry/client, or None when registration is disabled."""
        config = self.configuration
        if config.mlflow is None or config.mlflow.registered_model_name is None:
            return None
        else:
            return ClassifierChampionRegistry(config.mlflow, config.tracked_metric_definition, config.class_names)

    def _dataset(self, path: Path, *, training: bool) -> ImageDataset:
        """Index inputs once and share the established class mapping with validation."""
        dataset = ImageDataset(
            Path(path), self.configuration.class_names if training else self.training_dataset.label_names,
            transform=self.transforms.training() if training else self.validation_transform,
        )
        if any(count == 0 for count in dataset.class_counts.values()):
            raise ValueError("Every requested dataset class must be nonempty")
        return dataset

    def run(self) -> Path:
        """Train the head and return the native final weights/last.pt checkpoint."""
        return self.run_with_result().last_checkpoint

    def run_with_result(self, *, parent_run_id: str | None = None, defer_promotion: bool = False) -> ExperimentResult:
        """Train once, publish the selected native model, and optionally compare its champion.

        Args:
            parent_run_id: Explicit active pipeline parent for a child run.
            defer_promotion: Leave final alias comparison to the sweep owner.

        Returns:
            Selected best/final checkpoint metadata.
        """
        config = self.configuration
        context = config.mlflow.start_run(config.output_directory.name, parent_run_id=parent_run_id) if config.mlflow else nullcontext()
        with context as run:
            run_id = run.info.run_id if run is not None else None
            source_tags = config.mlflow.run_tags() if config.mlflow else {"git_commit": _git_commit()}
            if config.mlflow:
                mlflow.set_tags(source_tags)
                mlflow.log_params({"tracked_metric": config.tracked_metric, "tracked_mode": config.effective_tracked_mode})
            result, snapshot = self._run_training(run_id)
            with result.best_checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            self._write_model_card(result, snapshot, checksum, source_tags)
            if config.mlflow:
                prefix = "best_validation_" if config.validation_dataset is not None else "best_training_"
                mlflow.log_metric(prefix + config.tracked_metric, result.best_value, step=result.best_epoch)
                mlflow.log_metric("best_epoch", result.best_epoch)
                if self.champion_registry is not None:
                    version = self._publish_best_model(result, snapshot, checksum, source_tags)
                    result = replace(result, registered_model_version=version)
                    if not defer_promotion:
                        self.champion_registry.promote(result)
                else:
                    config.mlflow.log_best_artifacts(result.best_checkpoint)
            return result

    def _run_training(self, run_id):
        """Prepare the selected native model once; Ultralytics owns all optimization/checkpointing."""
        config = self.configuration
        source = Path(config.initialization["encoder"]["checkpoint"]).resolve()
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        torch.manual_seed(config.seed)
        model = YoloClassifierModel.initialize(
            config.initialization, label_names=self.training_dataset.label_names,
            encoder_batchnorm=config.encoder_batchnorm, head_donor=self.head_donor,
        )
        model.network.provenance = {**self.encoder_provenance, "encoder_checkpoint": str(source), "encoder_checkpoint_sha256": checksum}
        try:
            trainer = _HeadTrainer(self, model)
            (config.output_directory / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
            trainer.train()
        finally:
            model.eval()
        result = ExperimentResult(
            output_directory=config.output_directory,
            last_checkpoint=config.output_directory / "weights/last.pt", best_checkpoint=config.output_directory / "weights/best.pt",
            best_epoch=trainer.best_epoch, best_value=float(trainer.best_metrics[config.tracked_metric]),
            tracked_metric=config.tracked_metric, tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset, mlflow_run_id=run_id,
        )
        # Native finalization already read the selected file; retain metadata, not another model.
        return result, {"epoch": result.best_epoch, **trainer.best_metadata,
                        "history": [{"validation": trainer.best_metrics if config.validation_dataset is not None else {}}]}

    def _publish_best_model(self, result, snapshot, checksum, source_tags):
        """Restore best weights only for registry publication, without premature promotion."""
        config = self.configuration
        model = YoloClassifierModel._load_native(result.best_checkpoint.resolve()).eval()

        def write_card(version, identity):
            self._write_model_card(result, snapshot, checksum, {**source_tags, **identity}, version=version)
            config.mlflow.log_best_artifacts(result.best_checkpoint)
            return result.best_checkpoint.with_suffix(".model_card.yaml"), result.best_checkpoint.with_suffix(".model_card.md")

        return ModelPublication(config.mlflow, self.champion_registry.client).publish(
            model, best_epoch=result.best_epoch, validated=config.validation_dataset is not None,
            tags={**snapshot["provenance"], **source_tags, **self.champion_registry.comparison_tags(result)},
            write_card=write_card, promote=False,
        )

    def _write_model_card(self, result, snapshot, checksum, source_tags, *, version=None):
        """Describe the exact selected native snapshot using the shared template writer."""
        config = self.configuration
        return write_training_card(
            self.card_template, result.best_checkpoint, snapshot,
            model_details={"name": (config.mlflow.registered_model_name if config.mlflow else None) or config.output_directory.name,
                           "registry_version": version, "base_model": str(config.initialization["encoder"]["checkpoint"]), "task": "image-classification"},
            validation_dataset=config.validation_dataset,
            provenance={**snapshot["provenance"], **source_tags, "run_id": result.mlflow_run_id, "checkpoint_sha256": checksum,
                        "dataset_revision": config.mlflow.dataset_revision if config.mlflow else None},
        )
