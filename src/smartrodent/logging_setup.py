"""Logging set up by the command-line entry points, never by library classes."""

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass

# Loggers that do not live under the ``smartrodent`` package but whose output belongs
# in the same console and log file: Lightning's loggers (which keep a private handler
# and do not propagate) and the one that Python warnings are redirected to.
_THIRD_PARTY_LOGGERS = ("lightning.pytorch", "lightning.fabric", "py.warnings")


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class LoggingConfiguration:
    """The ``logging`` section of a training configuration.

    Args:
        suppress: Regular expressions; a console message in which any of them is
            found is left off the console but kept in the log file.
    """

    suppress: tuple[str, ...] = ()

    @field_validator("suppress")
    @classmethod
    def _compilable(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an invalid pattern when the configuration loads, not mid-training."""
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"suppress pattern {pattern!r} is not a valid regular expression: "
                    f"{error}"
                ) from error
        return patterns


class ConsoleNoiseFilter(logging.Filter):
    """Drop the records whose message matches any of the given regular expressions.

    Args:
        patterns: Regular expressions searched (not fully matched) in the message.
    """

    def __init__(self, patterns: Sequence[str]):
        super().__init__()
        self._patterns = [re.compile(pattern) for pattern in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for a record that is console noise."""
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self._patterns)


def log_to_console_and_file(log_file: Path, suppress: Sequence[str] = ()) -> None:
    """Send INFO messages to the console and to ``log_file``, minus console noise.

    The handlers go on the ``smartrodent`` package logger and on the third-party
    loggers of ``_THIRD_PARTY_LOGGERS`` rather than through ``logging.basicConfig``,
    which does nothing once the root logger has handlers. Python warnings are
    redirected into logging so that they pass the same filter. These loggers stop
    propagating to the root logger, and a repeated call replaces the earlier handlers.

    Args:
        log_file: File the messages are appended to; its directory must exist.
        suppress: Regular expressions of messages to keep off the console. The file
            always receives everything.
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler, console_handler = (
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    )
    console_handler.addFilter(ConsoleNoiseFilter(suppress))
    handlers = (file_handler, console_handler)
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.captureWarnings(True)
    logging.getLogger("smartrodent").setLevel(logging.INFO)
    for name in ("smartrodent", *_THIRD_PARTY_LOGGERS):
        logger = logging.getLogger(name)
        # Only this function's handlers may write to these named loggers: Lightning's
        # private stderr handler would print the noise unfiltered, and a handler left by
        # an earlier call would write everything twice. Closing releases their files.
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
            existing.close()
        # Without propagation the root logger's handlers cannot print the messages again.
        logger.propagate = False
        for handler in handlers:
            logger.addHandler(handler)

    if len(suppress) > 0:
        logging.getLogger(__name__).info(
            "Suppressing %d console message pattern(s); full output in %s",
            len(suppress),
            log_file,
        )


def log_training_run(output_directory: Path, settings: LoggingConfiguration) -> None:
    """Set up logging for a training script, writing ``training.log`` to its output.

    Args:
        output_directory: Directory of the run's artifacts; created when missing.
        settings: The configuration's ``logging`` section.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    log_to_console_and_file(output_directory / "training.log", settings.suppress)
