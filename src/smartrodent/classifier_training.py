"""YOLO classifier experiments with configurable layer freezing, using Ultralytics classification training."""

import hashlib
import math
from collections.abc import Callable, Mapping
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
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils.torch_utils import strip_optimizer

from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_model import YoloClassifierBuilder
from smartrodent.datasets import ImageDataset
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import (
    ImageTransforms,
    MlflowConfiguration,
    ModelPublication,
    _git_commit,
    resolve_validation_metrics,
)
from smartrodent.model_card import load_training_template, write_training_card
from smartrodent.onnx_export import OnnxExporter
from smartrodent.training_validation import (
    CheckpointPath,
    ClassifierTrainingSettings,
    ClassNames,
    Count,
    EncoderBatchnorm,
    FreezeLayers,
    InitializationSettings,
    MappingInput,
    Mode,
    Name,
    NonnegativeInt,
    OnnxExportSettings,
    Seed,
    TransformSettings,
    ValidationMetrics,
    parse_settings,
    settings_dict,
    validate_output_paths,
)


class FreezeConfigurationError(ValueError):
    """The freeze setting does not fit the network's depth and head replacement.

    It depends on the encoder network, so it can only be detected when the network is
    known; studies treat it as an invalid sampled combination.
    """


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class ClassifierConfiguration:
    """Configure native YOLO classifier training with independently owned shared settings.

    Args:
        train_dataset: Training split containing species directories.
        output_directory: Experiment destination outside input splits.
        class_names: Species selected for alphabetical dataset IDs.
        initialization: Encoder format/path and head replacement/source policy.
        freeze: Layer count N (layers 0..N-1) or unique layer indices to freeze; 0 or []
            trains everything. The head index is only valid with ``replace: linear``
            and freezes the head's convolution, so Linear always trains.
        transforms: Prepared torchvision preprocessing/augmentation options.
        train_kwargs: Native optimizer, learning-rate schedule, stopping and other options.
        validation_dataset: Optional existing validation split.
        encoder_batchnorm: Frozen layers' BatchNorm statistics: fixed (eval) or adapted
            (adapt). Required when layers are frozen, must be omitted/null otherwise.
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
        onnx_export: Optional ONNX export of the best checkpoint; omitted/null exports nothing.
    """

    train_dataset: Path
    output_directory: Path
    class_names: ClassNames
    initialization: InitializationSettings
    freeze: FreezeLayers
    transforms: TransformSettings
    train_kwargs: MappingInput[ClassifierTrainingSettings] = field(default_factory=dict)
    validation_dataset: Path | None = None
    encoder_batchnorm: EncoderBatchnorm | None = None
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
    onnx_export: MappingInput[OnnxExportSettings] | None = None

    def __post_init__(self):
        """Validate external options once before preparing native resources or outputs."""
        self._validate_train_kwargs()
        torch.device(
            self.device
        )  # Only a single explicit device, not a DDP subprocess/backend.
        self._validate_batchnorm_policy()
        self._validate_metrics()
        validate_output_paths(
            [self.output_directory],
            (self.train_dataset, self.validation_dataset),
            [
                self.model_card_template,
                *(node.get("checkpoint") for node in self.initialization.values()),
            ],
            native_directories=[self.output_directory],
        )

    def _validate_train_kwargs(self) -> None:
        """Reject native options this configuration owns, then let Ultralytics check the rest."""
        owned = {
            "model",
            "data",
            "task",
            "mode",
            "project",
            "name",
            "save_dir",
            "exist_ok",
            "freeze",
            "resume",
            "pretrained",
            "save",
            "epochs",
            "batch",
            "imgsz",
            "workers",
            "device",
            "seed",
            "val",
            "split",
            "classes",
            "single_cls",
            "cache",
            "augmentations",
            "auto_augment",
            "erasing",
            "scale",
            "crop_fraction",
            "fliplr",
            "flipud",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "bgr",
            "multi_scale",
            "distill_model",
            "quantize",
            "compile",
            "fraction",
            "tile",
            "save_period",
        }
        conflicts = owned & self.train_kwargs.keys()
        if conflicts:
            raise ValueError(
                f"train_kwargs cannot override owned or unsupported settings: {sorted(conflicts)}"
            )
        get_cfg(overrides=dict(self.train_kwargs))

    def _validate_batchnorm_policy(self) -> None:
        """Demand the BatchNorm policy exactly when it acts, i.e. when layers are frozen."""
        nothing_frozen = self.freeze in (0, [])
        if nothing_frozen and self.encoder_batchnorm is not None:
            raise ValueError(
                "encoder_batchnorm has no effect without frozen layers (freeze: 0 or []); omit it"
            )
        elif not nothing_frozen and self.encoder_batchnorm is None:
            raise ValueError(
                "encoder_batchnorm (eval | adapt) is required when freeze selects layers"
            )

    def _validate_metrics(self) -> None:
        """Keep custom metric names distinct from native ones and require data for validation objectives."""
        reserved = {
            "ce_loss_mean",
            "fitness",
            "val/loss",
            "metrics/accuracy_top1",
            "metrics/accuracy_top5",
            "train/loss",
            "epoch",
            "time",
        }
        metric_conflicts = {
            name
            for name in (self.validation_metrics or {})
            if name in reserved or (name.startswith("lr/pg") and name[5:].isdigit())
        }
        if metric_conflicts:
            raise ValueError(
                f"validation_metrics cannot override reserved names: {sorted(metric_conflicts)}"
            )
        if self.tracked_metric not in {
            "ce_loss_mean",
            *(self.validation_metrics or {}),
        }:
            raise ValueError(f"tracked_metric '{self.tracked_metric}' is not available")
        if self.validation_dataset is None and (
            self.validation_metrics or self.tracked_metric != "ce_loss_mean"
        ):
            raise ValueError("Validation objectives require validation_dataset")

    def native_freeze(self, layer_count: int) -> list[int | str]:
        """Translate ``freeze`` into Ultralytics' freeze list for a concrete network.

        Args:
            layer_count: Number of top-level layers, the last being the Classify head.

        Returns:
            Layer indices; a frozen linear-probe head becomes ``"<head>.conv"``.

        Raises:
            FreezeConfigurationError: A layer is outside the network or inside the trained head.
            ValueError: The head replacement is unsupported.
        """
        head = layer_count - 1
        if self.initialization["head"]["replace"] == "linear":
            last_freezable = head
        elif self.initialization["head"]["replace"] == "full":
            last_freezable = head - 1
        else:
            raise ValueError(
                f"Unsupported head replacement: {self.initialization['head']['replace']}"
            )
        layers = (
            list(range(self.freeze))
            if isinstance(self.freeze, int)
            else list(self.freeze)
        )
        out_of_range = [layer for layer in layers if layer > last_freezable]
        if out_of_range:
            raise FreezeConfigurationError(
                f"freeze={self.freeze} selects layer {out_of_range[0]}, but only layers 0..{last_freezable} "
                f"can be frozen with replace={self.initialization['head']['replace']} and {layer_count} layers"
            )
        # Native freeze accepts module-name suffixes as well as whole-layer indices.
        return [f"{layer}.conv" if layer == head else layer for layer in layers]

    def native_overrides(self, freeze: list[int | str]) -> dict:
        """Return Ultralytics trainer arguments for this run.

        Args:
            freeze: Native freeze list from ``native_freeze``.

        Returns:
            ``train_kwargs`` plus every option this configuration owns.
        """
        device = torch.device(self.device)
        output = self.output_directory.resolve()
        return {
            **self.train_kwargs,
            "model": str(self.initialization["encoder"]["checkpoint"]),
            "data": str(self.train_dataset.parent),
            "project": str(output.parent),
            "name": output.name,
            "exist_ok": True,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "workers": self.workers,
            "seed": self.seed,
            "device": str(device.index or 0) if device.type == "cuda" else str(device),
            "imgsz": self.transforms["image_size"],
            "freeze": freeze,
            # Native's val flag schedules validate(), including training-only selection.
            "val": True,
        }

    @property
    def effective_tracked_mode(self) -> str:
        """Return explicit direction or the loss-min/score-max default."""
        return self.tracked_mode or (
            "min" if self.tracked_metric == "ce_loss_mean" else "max"
        )

    @property
    def tracked_metric_definition(self) -> dict:
        """Return the exact diagnostic/scorer definition used for comparison."""
        if self.tracked_metric == "ce_loss_mean":
            return {
                "function": "torch.nn.functional.cross_entropy",
                "reduction": "mean",
            }
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


class _ClassifierValidator(ClassificationValidator):
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
        self.ce_total += torch.nn.functional.cross_entropy(
            self.raw_logits, batch["cls"], reduction="sum"
        )
        self.raw_logits = None

    def get_stats(self):
        """Return native statistics plus full-split CE and sklearn label metrics."""
        metrics = super().get_stats()
        actual = torch.cat(self.targets).numpy()
        predicted = torch.cat(self.pred)[:, 0].numpy()
        return {
            **metrics,
            "ce_loss_mean": self.ce_total.item() / len(actual),
            **{
                name: float(scorer(actual, predicted))
                for name, scorer in self.experiment.scorers.items()
            },
        }


class _ClassifierTrainer(ClassificationTrainer):
    """Supply prepared model/data and selection policy to the native training lifecycle."""

    def __init__(
        self,
        experiment,
        network: ClassificationModel,
        freeze: list[int | str],
        epoch_callback: Callable[[int, float], None] | None = None,
    ):
        """Configure native execution around the supplied network without reloading it."""
        self.experiment = experiment
        self.epoch_callback = epoch_callback
        super().__init__(overrides=experiment.configuration.native_overrides(freeze))
        # Native setup_model recognizes an existing nn.Module: no second factory,
        # checkpoint load or head reset can replace the selected encoder.
        self.model = network
        self.model.args = vars(self.args)
        self.model.transforms = experiment.validation_transform
        # Only the head's dropout before Linear, unlike native get_model(), which
        # rewrites every Dropout including those of a frozen encoder.
        self.model.model[-1].drop.p = self.args.dropout
        # Trailing dots keep layer 1 from matching layer 10 (native's own matching).
        self.frozen_prefixes = tuple(f"model.{layer}." for layer in freeze)
        self.adapt_frozen_batchnorm = (
            experiment.configuration.encoder_batchnorm == "adapt"
        )
        for event, callbacks in self.callbacks.items():
            self.callbacks[event] = [
                fn
                for fn in callbacks
                if fn.__module__ != "ultralytics.utils.callbacks.mlflow"
            ]
        self.best_epoch, self.best_metrics = 0, {}
        self.add_callback("on_fit_epoch_end", self._log_epoch)
        if epoch_callback is not None:
            # Native on_fit_epoch_end runs after validation and save_model, once per
            # trained epoch; final_eval below does not re-trigger it.
            self.add_callback("on_fit_epoch_end", self._report_epoch)

    def get_dataset(self):
        """Use shared, verified splits instead of native discovery/downloads."""
        config = self.experiment.configuration
        names = {
            index: name
            for name, index in self.experiment.training_dataset.label_names.items()
        }
        return {
            "train": str(config.train_dataset),
            "val": config.validation_dataset,
            "names": names,
            "nc": len(names),
            "channels": 3,
        }

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Use prepared torchvision transforms and deterministic, complete validation."""
        dataset = (
            self.experiment.training_dataset
            if mode == "train"
            else self.experiment.validation_dataset
        )
        if dataset is None:
            # Native progress reporting reads test_loader.dataset even with val=False.
            return torch.utils.data.DataLoader(())
        return build_dataloader(
            _ClassificationData(dataset),
            batch_size,
            self.args.workers,
            shuffle=mode == "train",
            rank=-1,
            drop_last=False,
            device=self.device,
        )

    def get_class_counts(self):
        """Return the established dataset's counts for native optional class weighting."""
        return np.asarray(
            list(self.experiment.training_dataset.class_counts.values()),
            dtype=np.float32,
        )

    def get_validator(self):
        """Build the native validator extension with the existing prepared scorers."""
        return _ClassifierValidator(
            self.experiment,
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _model_train(self):
        """Train unfrozen modules; frozen layers stay in eval unless their BatchNorm adapts.

        Native _model_train only fixes BatchNorm in frozen layers and would leave
        their Dropout active, which changes frozen features during training.
        """
        self.model.train()
        # Parents precede children in named_modules(), so BatchNorm can re-enable
        # training after its frozen layer was switched to eval as a whole. _BatchNorm
        # (unlike native's BatchNorm2d check) covers every BatchNorm dimensionality.
        for name, module in self.model.named_modules():
            if any(f"{name}.".startswith(prefix) for prefix in self.frozen_prefixes):
                module.train(
                    self.adapt_frozen_batchnorm
                    and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                )

    def validate(self):
        """Select native fitness from the configured objective, never native top-1 defaults."""
        config = self.experiment.configuration
        metrics = (
            self.validator(self)
            if config.validation_dataset is not None
            else {"ce_loss_mean": float(self.tloss["loss"])}
        )
        if config.tracked_metric not in metrics:
            raise ValueError(
                f"Missing tracked validation metric '{config.tracked_metric}'"
            )
        value = float(metrics[config.tracked_metric])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Nonfinite tracked validation metric '{config.tracked_metric}'"
            )
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
                metrics.update(
                    {
                        f"validation/{name}": value
                        for name, value in self.metrics.items()
                    }
                )
            mlflow.log_metrics(metrics, step=self.epoch + 1)

    def _report_epoch(self, trainer):
        """Pass the configured tracked value of the completed epoch to the observer."""
        tracked_metric = self.experiment.configuration.tracked_metric
        self.epoch_callback(self.epoch + 1, float(self.metrics[tracked_metric]))

    def final_eval(self):
        """Finalize native files without rescoring or rediscovering shared dataset paths."""
        last = strip_optimizer(self.last)
        best = strip_optimizer(
            self.best, updates={"train_results": last.get("train_results")}
        )
        if not best:
            raise ValueError(
                f"Native best checkpoint could not be finalized: {self.best}"
            )
        model = best["model"]
        self.best_metadata = {
            "label_names": {name: index for index, name in model.names.items()},
            "provenance": model.provenance,
        }
        self.metrics = self.best_metrics


class ClassifierExperiment:
    """Own prepared inputs and publication around native classifier training.

    Args:
        configuration: Validated classifier settings.
        encoder_provenance: Optional encoder-selection metadata from orchestration.
        head_cache: Optional sweep-owned read-only head checkpoint cache.
    """

    def __init__(
        self,
        configuration: ClassifierConfiguration,
        *,
        encoder_provenance: Mapping | None = None,
        head_cache=None,
    ):
        """Prepare known inputs without creating run outputs or initializing an encoder."""
        self.configuration = configuration
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        self.transforms = ImageTransforms(configuration.transforms)
        self.validation_transform = self.transforms.validation()
        self.scorers = resolve_validation_metrics(configuration.validation_metrics)
        self.card_template = load_training_template(
            Path(configuration.model_card_template)
        )
        self.training_dataset = self._dataset(
            configuration.train_dataset, training=True
        )
        self.validation_dataset = (
            self._dataset(configuration.validation_dataset, training=False)
            if configuration.validation_dataset is not None
            else None
        )
        self.head_donor = YoloClassifierBuilder.prepare_head(
            configuration.initialization["head"],
            self.training_dataset.label_names,
            cache=head_cache,
        )
        self.exporter = (
            OnnxExporter(
                configuration.onnx_export, configuration.transforms["image_size"]
            )
            if configuration.onnx_export is not None
            else None
        )

    @cached_property
    def champion_registry(self) -> ClassifierChampionRegistry | None:
        """Return the owned registry/client, or None when registration is disabled."""
        config = self.configuration
        if config.mlflow is None or config.mlflow.registered_model_name is None:
            return None
        else:
            return ClassifierChampionRegistry(
                config.mlflow, config.tracked_metric_definition, config.class_names
            )

    def _dataset(self, path: Path, *, training: bool) -> ImageDataset:
        """Index inputs once and share the established class mapping with validation."""
        dataset = ImageDataset(
            Path(path),
            (
                self.configuration.class_names
                if training
                else self.training_dataset.label_names
            ),
            transform=(
                self.transforms.training() if training else self.validation_transform
            ),
        )
        if any(count == 0 for count in dataset.class_counts.values()):
            raise ValueError("Every requested dataset class must be nonempty")
        return dataset

    def run(self) -> Path:
        """Train the classifier and return the native final weights/last.pt checkpoint."""
        return self.run_with_result().last_checkpoint

    def run_with_result(
        self,
        *,
        parent_run_id: str | None = None,
        defer_promotion: bool = False,
        epoch_callback: Callable[[int, float], None] | None = None,
    ) -> ExperimentResult:
        """Train once, publish the selected native model, and optionally compare its champion.

        Args:
            parent_run_id: Explicit active pipeline parent for a child run.
            defer_promotion: Leave final alias comparison to the sweep owner.
            epoch_callback: Optional observer called as ``(epoch, tracked_value)``
                after each native epoch is validated and saved. Exceptions it raises
                stop training and propagate unchanged, before any publication.

        Returns:
            Selected best/final checkpoint metadata.
        """
        config = self.configuration
        # Prepare before tracking: an invalid encoder or freeze leaves no MLflow run behind.
        network = self._prepare_network()
        freeze = config.native_freeze(len(network.model))
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
                else {"git_commit": _git_commit()}
            )
            if config.mlflow:
                mlflow.set_tags(source_tags)
                mlflow.log_params(
                    {
                        "tracked_metric": config.tracked_metric,
                        "tracked_mode": config.effective_tracked_mode,
                    }
                )
            result, snapshot = self._train(network, freeze, run_id, epoch_callback)
            # Export before any publication: a failed export must leave nothing published.
            if self.exporter is not None:
                result = replace(
                    result, onnx_model=self.exporter.classifier(result.best_checkpoint)
                )
            with result.best_checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            self._write_model_card(result, snapshot, checksum, source_tags)
            if config.mlflow:
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
                if self.champion_registry is not None:
                    version = self._publish_best_model(
                        result, snapshot, checksum, source_tags
                    )
                    result = replace(result, registered_model_version=version)
                    if not defer_promotion:
                        self.champion_registry.promote(result)
                else:
                    config.mlflow.log_best_artifacts(
                        result.best_checkpoint, result.onnx_model
                    )
            return result

    def _prepare_network(self) -> ClassificationModel:
        """Build the seeded network once and record which encoder file it came from."""
        config = self.configuration
        source = Path(config.initialization["encoder"]["checkpoint"]).resolve()
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        torch.manual_seed(config.seed)
        network = YoloClassifierBuilder.initialize(
            config.initialization,
            label_names=self.training_dataset.label_names,
            head_donor=self.head_donor,
        )
        network.provenance = {
            **self.encoder_provenance,
            "encoder_checkpoint": str(source),
            "encoder_checkpoint_sha256": checksum,
        }
        return network

    def _train(
        self,
        network: ClassificationModel,
        freeze: list[int | str],
        run_id: str | None,
        epoch_callback: Callable[[int, float], None] | None = None,
    ):
        """Run native training; Ultralytics owns all optimization and checkpointing."""
        config = self.configuration
        try:
            trainer = _ClassifierTrainer(self, network, freeze, epoch_callback)
            (config.output_directory / "config.yaml").write_text(
                yaml.safe_dump(config.to_dict()), encoding="utf-8"
            )
            trainer.train()
        finally:
            network.eval()
        result = ExperimentResult(
            output_directory=config.output_directory,
            last_checkpoint=config.output_directory / "weights/last.pt",
            best_checkpoint=config.output_directory / "weights/best.pt",
            best_epoch=trainer.best_epoch,
            best_value=float(trainer.best_metrics[config.tracked_metric]),
            tracked_metric=config.tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
            mlflow_run_id=run_id,
        )
        # Native finalization already read the selected file; retain metadata, not another model.
        return result, {
            "epoch": result.best_epoch,
            **trainer.best_metadata,
            "history": [
                {
                    "validation": (
                        trainer.best_metrics
                        if config.validation_dataset is not None
                        else {}
                    )
                }
            ],
        }

    def _publish_best_model(self, result, snapshot, checksum, source_tags):
        """Restore best weights only for registry publication, without premature promotion."""
        config = self.configuration
        model = YoloClassifierBuilder.load_native(result.best_checkpoint).eval()

        def write_card(version, identity):
            self._write_model_card(
                result, snapshot, checksum, {**source_tags, **identity}, version=version
            )
            config.mlflow.log_best_artifacts(result.best_checkpoint, result.onnx_model)
            return result.best_checkpoint.with_suffix(
                ".model_card.yaml"
            ), result.best_checkpoint.with_suffix(".model_card.md")

        return ModelPublication(config.mlflow, self.champion_registry.client).publish(
            model,
            best_epoch=result.best_epoch,
            validated=config.validation_dataset is not None,
            tags={
                **snapshot["provenance"],
                **source_tags,
                **self.champion_registry.comparison_tags(result),
            },
            write_card=write_card,
            promote=False,
        )

    def _write_model_card(
        self, result, snapshot, checksum, source_tags, *, version=None
    ):
        """Describe the exact selected native snapshot using the shared template writer."""
        config = self.configuration
        return write_training_card(
            self.card_template,
            result.best_checkpoint,
            snapshot,
            model_details={
                "name": (config.mlflow.registered_model_name if config.mlflow else None)
                or config.output_directory.name,
                "registry_version": version,
                "base_model": str(config.initialization["encoder"]["checkpoint"]),
                "task": "image-classification",
            },
            validation_dataset=config.validation_dataset,
            provenance={
                **snapshot["provenance"],
                **source_tags,
                "run_id": result.mlflow_run_id,
                "checkpoint_sha256": checksum,
                "dataset_revision": (
                    config.mlflow.dataset_revision if config.mlflow else None
                ),
            },
        )
