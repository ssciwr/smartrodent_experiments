"""Declarative settings types for all configurable components, and filesystem safety."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

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


@with_config(ConfigDict(extra="forbid"))
class OnnxExportSettings(TypedDict):
    """Select ONNX export options; an empty mapping exports with the defaults.

    The opset is deliberately not configurable: both exports use Ultralytics'
    default for the installed torch, so classifier and encoder files always agree.
    """

    dynamic: NotRequired[Flag]
    simplify: NotRequired[Flag]


@with_config(ConfigDict(extra="forbid"))
class SharedData(TypedDict):
    """Keep pipeline data concrete while stage settings may still contain search tags."""

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
