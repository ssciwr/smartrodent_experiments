from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any


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
