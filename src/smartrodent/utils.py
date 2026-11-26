import re
from collections.abc import Sequence
from pathlib import Path
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


def parse_path(text: str) -> tuple:
    """Parse a dotted configuration path such as ``a.b[0]`` into ``("a", "b", 0)``.

    Args:
        text: Dotted path, with list indices in brackets (the coupled-sweep syntax).

    Returns:
        Keys and indices leading to the addressed value.
    """
    components = []
    for token in text.split("."):
        key, *indices = token.split("[")
        components.append(key)
        components.extend(int(index.rstrip("]")) for index in indices)
    return tuple(components)


def format_path(path: Sequence[Any]) -> str:
    """Format configuration keys and indices as a dotted path such as ``a.b[0]``.

    Args:
        path: Keys and list indices leading to a value.

    Returns:
        The dotted path, the inverse of ``parse_path``.
    """
    text = ""
    for component in path:
        if isinstance(component, int) and not isinstance(component, bool):
            text += f"[{component}]"
        elif text:
            text += f".{component}"
        else:
            text = str(component)
    return text


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
