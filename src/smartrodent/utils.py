from collections.abc import Sequence
import importlib
from pathlib import Path
import re
from typing import Any

import torch


def path_component(value: str) -> str:
    """Return a readable value that is safe to use as a directory or file name."""
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return value or "unknown"


def assign_at_path(cfg: dict, path: Sequence[Any], value: Any) -> None:
    """Assign a value within a nested mapping or sequence.

    Args:
        cfg: Nested configuration to modify.
        path: Keys and indices leading to the target value.
        value: Value to assign at the target.
    """
    target = cfg
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a possibly-relative data path against the project root.

    Args:
        path: Path to resolve. If already absolute, it is returned unchanged.

    Returns:
        Path: The absolute path, resolved relative to the ``PARENT`` module
        global if set, otherwise relative to the repository root inferred
        from this file's location.
    """
    p = globals().get("PARENT", Path(__file__).resolve().parents[2])
    path = Path(path)
    return path if path.is_absolute() else (p / path).resolve()


def import_class(class_path: str) -> Any:
    """Import an attribute, typically a class, from a dotted import path.

    Args:
        class_path: Full ``package.module.Attribute`` path, as written in configs.

    Returns:
        Any: The imported attribute.

    Raises:
        ValueError: The path has no module part, the module cannot be imported, or
            the module has no such attribute.
    """
    module_name, _, attribute_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"'{class_path}' is not a full module.Class import path")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"Cannot import module '{module_name}'") from error
    if not hasattr(module, attribute_name):
        raise ValueError(f"Module '{module_name}' has no attribute '{attribute_name}'")
    return getattr(module, attribute_name)


def to_json_value(value: torch.Tensor) -> int | float | bool | list:
    """Convert a tensor into a JSON-serializable Python value.

    Args:
        value: Tensor of any shape.

    Returns:
        A Python scalar for 0-dimensional tensors, otherwise a (nested) list.
    """
    if value.ndim == 0:
        return value.item()
    else:
        return value.tolist()
