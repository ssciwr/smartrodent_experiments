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
class Filterable(Configurable, Protocol):
    """Protocol for objects that filter a set of images and report results."""

    def filter_data(self) -> pd.DataFrame:
        """Run filtering over the input data.

        Returns:
            pd.DataFrame: One row per processed item describing the outcome.
        """

    def save_results(self, res_df: pd.DataFrame) -> Path:
        """Persist filtering results to disk.

        Args:
            res_df: The results DataFrame returned by :meth:`filter_data`.

        Returns:
            Path: The path the results were written to.
        """

    def collect_image_paths(self) -> list[Path]:
        """Gather the image paths that should be filtered.

        Returns:
            list[Path]: Paths of the images to process.
        """
