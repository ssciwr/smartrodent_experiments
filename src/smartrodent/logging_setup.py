"""Logging set up by the command-line entry points, never by library classes."""

import logging
from pathlib import Path


def log_to_console_and_file(log_file: Path) -> None:
    """Send the package's INFO messages to the console and to ``log_file``.

    The handlers go on the ``smartrodent`` package logger rather than through
    ``logging.basicConfig``, which does nothing once the root logger has handlers.
    Each call adds handlers, so call it once per process.

    Args:
        log_file: File the messages are appended to; its directory must exist.
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("smartrodent")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(log_file), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
