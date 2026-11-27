"""Validated configurations of the SupCon and classifier stages."""

from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from lightning.pytorch.callbacks import EarlyStopping
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from smartrodent import config_loading
from smartrodent.classifier_task import CE_METRIC
from smartrodent.components import ComponentSpec
from smartrodent.encoders import FreezePolicy, Reinitialization
from smartrodent.experiment_results import Objective
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.metrics import ValidationMetrics
from smartrodent.output_paths import RunArtifacts, validate_output_paths
from smartrodent.search_space import reject_search_tags
from smartrodent.stage_runner import StageRunner
from smartrodent.supcon_task import SupConTask
from smartrodent.training_optimization import SchedulerPhase
from smartrodent.training_validation import (
    ClassNames,
    Count,
    MappingInput,
    Mode,
    Nonblank,
    NonnegativeInt,
    OnnxExportSettings,
    PathSetting,
    PositiveNumber,
    Probability,
    Seed,
    parse_settings,
    settings_dict,
)
from smartrodent.transforms import TransformSettings


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CheckpointEncoder:
    """Name a Lightning checkpoint that an encoder is restored from.

    Args:
        checkpoint: SupCon or classifier ``.ckpt`` file.
    """

    checkpoint: PathSetting


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StageConfiguration:
    """Settings every training stage shares, and the guard of its output paths.

    Args:
        train_dataset: Training split with one directory per species.
        output_directory: Directory of this stage's artifacts, outside the inputs.
        class_names: Species selected for alphabetical dataset IDs.
        transforms: Image size, resize options and training augmentations.
        optimizer: PyTorch optimizer and its options.
        trainer: Native ``L.Trainer`` arguments; ``max_epochs`` is required.
        scheduler: Learning-rate phases; empty keeps the rate constant.
        early_stopping: Native Lightning ``EarlyStopping`` options, or None.
        validation_dataset: Optional validation split.
        freeze: Encoder parameters to freeze and the frozen BatchNorm policy.
        workers: Loader worker count.
        seed: Seed for initialization and shuffling.
        mlflow: Optional tracking and registry settings.
        model_card_template: Template of the selected checkpoint's model card.
        onnx_export: Optional ONNX export of the best checkpoint.
    """

    # Settings that define what trials are compared on, or where they read and write.
    protected: ClassVar[tuple[str, ...]] = (
        "tracked_metric",
        "tracked_mode",
        "validation_metrics",
        "train_dataset",
        "validation_dataset",
        "class_names",
        "output_directory",
        "mlflow",
    )

    train_dataset: Path
    output_directory: Path
    class_names: ClassNames
    transforms: MappingInput[TransformSettings]
    optimizer: MappingInput[ComponentSpec]
    trainer: dict[str, Any]
    scheduler: tuple[MappingInput[SchedulerPhase], ...] = ()
    early_stopping: dict[str, Any] | None = None
    validation_dataset: Path | None = None
    freeze: MappingInput[FreezePolicy] = field(default_factory=FreezePolicy)
    workers: NonnegativeInt = 0
    seed: Seed = 42
    mlflow: MappingInput[MlflowConfiguration] | None = None
    model_card_template: PathSetting = "configs/model_card_template.yaml"
    onnx_export: MappingInput[OnnxExportSettings] | None = None

    def __post_init__(self):
        """Validate the shared sections, then protect the inputs from the outputs."""
        reserved = StageRunner.reserved_trainer_keys & self.trainer.keys()
        if reserved:
            raise ValueError(
                f"trainer settings {sorted(reserved)} are owned by the stage runner"
            )
        if "max_epochs" not in self.trainer:
            raise ValueError("trainer must state max_epochs explicitly")
        if self.early_stopping is not None:
            # The monitor is supplied by the runner; any value checks the options.
            EarlyStopping(monitor="metric", **self.early_stopping)
        validate_output_paths(
            [self.artifacts],
            (self.train_dataset, self.validation_dataset),
            self.input_files(),
        )

    @property
    def artifacts(self) -> RunArtifacts:
        """Return the paths this stage writes."""
        return RunArtifacts(
            self.output_directory, onnx=self.onnx_export is not None, native=False
        )

    def input_files(self) -> tuple:
        """Return the files the stage reads, which no output may overwrite."""
        return (self.model_card_template,)

    @classmethod
    def from_mapping(cls, values: Mapping):
        """Parse one concrete configuration mapping."""
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Read one run's configuration, rejecting search tags that need a study.

        Args:
            path: Configuration file.

        Raises:
            ValueError: The settings are invalid or contain search tags.
        """
        document = config_loading.load_yaml(path)
        reject_search_tags(document)
        return cls.from_mapping(document)

    def to_dict(self) -> dict:
        """Return an independent YAML-compatible dictionary."""
        return settings_dict(self)


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class SupConConfiguration(StageConfiguration):
    """Configure supervised contrastive encoder training.

    Args:
        encoder: Component spec of the encoder to train.
        classes_per_batch: Distinct classes per batch.
        samples_per_class: Source images per selected class.
        samples_per_epoch: Source-image selections per epoch, rounded to batches.
        prefetch_factor: Batches prefetched per worker.
        dropout: Dropout probability before the projection.
        output_features: Projected embedding width.
        temperature: SupCon loss temperature.
    """

    protected: ClassVar[tuple[str, ...]] = (
        *StageConfiguration.protected,
        "temperature",
    )

    encoder: MappingInput[ComponentSpec]
    classes_per_batch: Annotated[Count, Field(ge=2)] = 4
    samples_per_class: Count = 6
    samples_per_epoch: Count = 100000
    prefetch_factor: Count = 4
    dropout: Probability = 0.5
    output_features: Count = 128
    temperature: PositiveNumber = 0.1

    def __post_init__(self):
        """Check the batch layout before building any dataset."""
        if self.classes_per_batch > len(self.class_names):
            raise ValueError(
                "classes_per_batch must be between 2 and the number of species"
            )
        if self.samples_per_epoch < self.batch_size:
            raise ValueError("samples_per_epoch must contain at least one full batch")
        super().__post_init__()

    @property
    def objective(self) -> Objective:
        """Return the SupCon loss mean, which is always minimized."""
        return SupConTask.objective

    @property
    def batch_size(self) -> int:
        """Return the number of source images per batch, before doubling views."""
        return self.classes_per_batch * self.samples_per_class


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class ClassifierConfiguration(StageConfiguration):
    """Configure cross-entropy classifier training.

    Args:
        encoder: Component spec, or a checkpoint the encoder is restored from.
        batch_size: Source images per batch.
        head: ``linear`` adds only a new final layer; ``full`` also renews the
            encoder's post-pooling part.
        reinitialization: Optional random reset of encoder modules.
        native_checkpoint: Whether to export a native ``best.pt`` beside ``best.ckpt``.
        validation_metrics: Optional sklearn scorers by alias.
        tracked_metric: ``ce_loss_mean`` or a metric alias.
        tracked_mode: Optional direction; defaults to loss ``min``, score ``max``.
    """

    encoder: MappingInput[ComponentSpec] | MappingInput[CheckpointEncoder]
    batch_size: Count = 32
    head: Literal["linear", "full"] = "linear"
    reinitialization: MappingInput[Reinitialization] | None = None
    native_checkpoint: bool = False
    validation_metrics: dict[Nonblank, MappingInput[ComponentSpec]] | None = None
    tracked_metric: Nonblank = CE_METRIC
    tracked_mode: Mode | None = None

    def __post_init__(self):
        """Require validation data for validation objectives."""
        if self.validation_dataset is None and (
            self.validation_metrics or self.tracked_metric != CE_METRIC
        ):
            raise ValueError("Validation objectives require validation_dataset")
        super().__post_init__()

    @property
    def objective(self) -> Objective:
        """Return the tracked metric and its direction."""
        return Objective(
            self.tracked_metric,
            self.tracked_mode or ("min" if self.tracked_metric == CE_METRIC else "max"),
        )

    @property
    def metric_definition(self) -> dict:
        """Return the registry identity of the tracked metric (unchanged format)."""
        if self.tracked_metric == CE_METRIC:
            return {
                "function": "torch.nn.functional.cross_entropy",
                "reduction": "mean",
            }
        return ValidationMetrics(self.validation_metrics).definition(
            self.tracked_metric
        )

    @property
    def artifacts(self) -> RunArtifacts:
        """Return the paths this stage writes, including an optional native export."""
        return RunArtifacts(
            self.output_directory,
            onnx=self.onnx_export is not None,
            native=self.native_checkpoint,
        )

    def input_files(self) -> tuple:
        """Return the template and encoder files the run reads."""
        files = [self.model_card_template]
        if isinstance(self.encoder, CheckpointEncoder):
            files.append(self.encoder.checkpoint)
        return tuple(files)
