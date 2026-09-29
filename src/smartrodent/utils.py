from pathlib import Path


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
