from typing import Protocol, runtime_checkable
import pandas as pd
from pathlib import Path


@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects that can be constructed from a YAML config file."""

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Configurable":
        """Build an instance from a YAML config file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configurable: A new instance configured from the file.
        """


@runtime_checkable
class DatasetLoader(Configurable, Protocol):
    """Protocol for objects that execute a download method to get data"""

    def download(self) -> None:
        """Fetch records and download allowed photos for every configured species."""
        ...
