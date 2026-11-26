"""Declarative settings types for all configurable components, and filesystem safety."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    InstanceOf,
    StringConstraints,
    TypeAdapter,
    with_config,
)

type Flag = Annotated[bool, Field(strict=True)]
type Count = Annotated[int, Field(strict=True, gt=0)]
type NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
type Seed = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
type FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type PositiveNumber = Annotated[FiniteNumber, Field(gt=0)]
type NonnegativeNumber = Annotated[FiniteNumber, Field(ge=0)]
type Probability = Annotated[FiniteNumber, Field(ge=0, le=1)]
type Name = Annotated[str, StringConstraints(strict=True, min_length=1)]
type Nonblank = Annotated[Name, StringConstraints(pattern=r"\S")]
# A path setting: a Path, or a nonblank string (a blank string would mean the cwd).
type PathSetting = Nonblank | InstanceOf[Path]
type Mode = Literal["min", "max"]
type EncoderSource = Literal["supcon", "ultralytics"]
type HeadReplacement = Literal["linear", "full"]
type EncoderBatchnorm = Literal["eval", "adapt"]


def _mapping_input(value):
    """Normalize Mapping inputs at root and nested dataclass boundaries."""
    return dict(value) if isinstance(value, Mapping) else value


type MappingInput[T] = Annotated[T, BeforeValidator(_mapping_input)]


def unique(values):
    """Reject repeated entries instead of silently deduplicating them.

    Args:
        values: Validated sequence.

    Returns:
        ``values`` unchanged.

    Raises:
        ValueError: If an entry occurs more than once.
    """
    if len(set(values)) != len(values):
        raise ValueError("entries must be unique")
    return values


# Sequence, not tuple/list: pydantic's lax tuples and lists also accept sets, which
# would make the order (class indices, column order) differ between runs.
type ColumnNames = Annotated[
    Sequence[Name], AfterValidator(unique), AfterValidator(tuple)
]
type ClassNames = Annotated[
    Sequence[Name], Field(min_length=1), AfterValidator(unique), AfterValidator(tuple)
]


# Ultralytics' convention: a count N freezes layers 0..N-1, a list names layers.
# The value is kept as given (not normalized) so saved configs match their source.
type FreezeLayers = (
    NonnegativeInt | Annotated[list[NonnegativeInt], AfterValidator(unique)]
)


@with_config(ConfigDict(extra="allow"))
class OptimizerSettings(TypedDict):
    """Require a valid learning rate while preserving native AdamW options."""

    lr: PositiveNumber


@with_config(ConfigDict(extra="allow"))
class ClassifierTrainingSettings(TypedDict):
    """Pass native training options through, retaining the positive learning-rate contract."""

    lr0: NotRequired[PositiveNumber]


@with_config(ConfigDict(extra="allow"))
class MetricSettings(TypedDict):
    """Require a scorer name while preserving its native sklearn keyword options."""

    function: Name


type ValidationMetrics = dict[Name, MetricSettings]


@with_config(ConfigDict(extra="allow"))
class TransformSettings(TypedDict):
    """Require pipeline sections; torchvision owns their native keyword contracts."""

    image_size: Count
    resize: dict[str, Any]
    random_adjust_sharpness: dict[str, Any]
    random_grayscale: dict[str, Any]
    random_perspective: dict[str, Any]
    random_horizontal_flip: dict[str, Any]
    random_resized_crop: dict[str, Any]


@with_config(ConfigDict(extra="forbid"))
class PlateauSettings(TypedDict):
    """Validate supported plateau options without injecting copies of PyTorch defaults."""

    mode: NotRequired[Mode]
    factor: NotRequired[Annotated[FiniteNumber, Field(gt=0, lt=1)]]
    patience: NotRequired[NonnegativeInt]
    threshold: NotRequired[NonnegativeNumber]
    threshold_mode: NotRequired[Literal["rel", "abs"]]
    cooldown: NotRequired[NonnegativeInt]
    min_lr: NotRequired[NonnegativeNumber]
    eps: NotRequired[NonnegativeNumber]


@with_config(ConfigDict(extra="forbid"))
class StoppingSettings(TypedDict):
    """Describe early-stopping configuration separately from its runtime state."""

    patience: Count
    tolerance: NonnegativeNumber
    comparison: Literal["less", "greater"]


@with_config(ConfigDict(extra="forbid"))
class EncoderSettings(TypedDict):
    """Identify a local standalone encoder without loading the checkpoint."""

    source: EncoderSource
    checkpoint: PathSetting


@with_config(ConfigDict(extra="forbid"))
class RandomHead(TypedDict):
    """Configure a fresh head, permitting an explicitly coupled null donor."""

    replace: HeadReplacement
    source: Literal["random"]
    checkpoint: NotRequired[None]


@with_config(ConfigDict(extra="forbid"))
class CheckpointHead(TypedDict):
    """Configure a head initialized from an explicit local donor."""

    replace: HeadReplacement
    source: Literal["yolo_checkpoint"]
    checkpoint: PathSetting


@with_config(ConfigDict(extra="forbid"))
class InitializationSettings(TypedDict):
    """Discriminate head-source requirements without handwritten key/type checks."""

    encoder: EncoderSettings
    head: Annotated[RandomHead | CheckpointHead, Field(discriminator="source")]


@with_config(ConfigDict(extra="forbid"))
class OnnxExportSettings(TypedDict):
    """Select ONNX export options; an empty mapping exports with Ultralytics' defaults.

    The opset is deliberately not configurable: both exports use Ultralytics'
    default for the installed torch, so classifier and encoder files always agree.
    """

    dynamic: NotRequired[Flag]
    simplify: NotRequired[Flag]


@with_config(ConfigDict(extra="forbid"))
class SharedData(TypedDict):
    """Keep pipeline data concrete while stage settings may still contain sweep tags."""

    train_dataset: Path
    validation_dataset: Path
    class_names: ClassNames


def parse_settings(schema, values):
    """Validate input without sharing caller-owned mutable configuration values.

    Args:
        schema: Target dataclass, TypedDict, or scalar schema.
        values: Python/YAML settings or external metadata.

    Returns:
        The parsed value, raising Pydantic validation errors for invalid input.
    """
    # Normalize Mapping implementations before copying; mappingproxy cannot be pickled.
    parsed = TypeAdapter(schema).validate_python(_mapping_input(values))
    return deepcopy(parsed)


def settings_dict(settings) -> dict:
    """Serialize validated dataclasses to checkpoint/YAML-safe primitive values.

    Args:
        settings: Validated configuration or result.

    Returns:
        Primitive fields with paths and tuples encoded for YAML/checkpoints.
    """
    return TypeAdapter(type(settings)).dump_python(settings, mode="json")


def validate_output_paths(
    directories: Sequence[Path],
    datasets=(),
    sources=(),
    roots=(),
    *,
    native_directories=(),
) -> None:
    """Protect dataset/source bytes against all run artifacts, symlinks, and hard links.

    Args:
        directories: Per-run output directories, which must not overlap.
        datasets: Input splits that no output directory may be inside.
        sources: Model weights and templates that no artifact may overwrite.
        roots: Optional sweep/pipeline roots containing configuration and summaries.
        native_directories: Classifier runs whose native artifacts/weights must be protected.

    Raises:
        ValueError: Outputs overlap, lie inside data, or collide with any input.
    """
    outputs = [Path(directory).resolve() for directory in directories]
    roots = [Path(root).resolve() for root in roots]
    for index, directory in enumerate(outputs):
        if any(
            directory.is_relative_to(other) or other.is_relative_to(directory)
            for other in outputs[index + 1 :]
        ):
            raise ValueError("output_directory paths must not overlap")
    inputs = [Path(dataset).resolve() for dataset in datasets if dataset is not None]
    for dataset in inputs:
        if any(output.is_relative_to(dataset) for output in [*outputs, *roots]):
            raise ValueError("output_directory must be outside both input datasets")
    names = (
        "config.yaml",
        "history.yaml",
        "last.pt",
        "best.pt",
        ".last.pt.tmp",
        ".best.pt.tmp",
        "best.model_card.yaml",
        "best.model_card.md",
    )
    artifacts = [directory / name for directory in outputs for name in names]
    root_artifacts = [
        root / name for root in roots for name in ("config.yaml", "selection.yaml")
    ]
    artifacts.extend(root_artifacts)
    native = [Path(directory).resolve() for directory in native_directories]
    native_names = (
        "config.yaml",
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "weights/last.pt",
        "weights/best.pt",
        "weights/best.model_card.yaml",
        "weights/best.model_card.md",
    )
    native_artifacts = [
        directory / name for directory in native for name in native_names
    ]
    # Native plots/periodic snapshots can have epoch-dependent names. Existing
    # aliases must be checked too, while the weights subtree is wholly output-owned.
    for directory in native:
        native_artifacts.extend(directory.glob("*.jpg"))
        native_artifacts.extend(directory.glob("*.png"))
        native_artifacts.extend(
            path for path in (directory / "weights").glob("*") if path.is_file()
        )
    artifacts.extend(native_artifacts)
    for dataset in inputs:
        if any(artifact.resolve().is_relative_to(dataset) for artifact in artifacts):
            raise ValueError(
                "Output artifacts must remain outside both input datasets, including aliases"
            )
    # Native writers truncate existing files. Reject shared inodes rather than
    # scanning large datasets; SupCon's intentional checkpoint hardlinks remain valid.
    for artifact in [*native_artifacts, *root_artifacts]:
        if artifact.is_file() and artifact.stat().st_nlink > 1:
            raise ValueError(
                f"Output artifact {artifact} is hardlinked and could overwrite an input/source"
            )
    for source in sources:
        if source is None:
            continue
        source = Path(source).resolve()
        if any(
            source.is_relative_to((directory / "weights").resolve())
            for directory in native
        ):
            raise ValueError(
                f"Native output weights collide with source checkpoint/template {source}"
            )
        for artifact in artifacts:
            if source == artifact.resolve() or (
                source.exists() and artifact.exists() and source.samefile(artifact)
            ):
                raise ValueError(
                    f"Output artifact {artifact} collides with source checkpoint/template {source}"
                )
