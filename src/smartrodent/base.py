"""Protocols shared by the configurable dataset loaders and filters."""

from abc import abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

# The protocol methods are abstract so that an implementation missing one fails when
# it is created, instead of silently inheriting a stub that does nothing.


@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects that can be constructed from a YAML config file."""

    @classmethod
    @abstractmethod
    def from_config(cls, config_path: str | Path) -> "Configurable":
        """Build an instance from a YAML config file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configurable: A new instance configured from the file.
        """
        ...


@runtime_checkable
class Filterable(Configurable, Protocol):
    """Protocol for per-species dataframe filters that only read images."""

    @abstractmethod
    def filter_data(
        self, records_by_species: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Annotate per-species photo records with filter results.

        Args:
            records_by_species: Species names mapped to photo metadata frames.

        Returns:
            Per-species frames with filter-result columns added.
        """
        ...


@runtime_checkable
class DatasetLoader(Configurable, Protocol):
    """Protocol for dataset loaders with separate record and image phases."""

    @abstractmethod
    def retrieve_records(self) -> None:
        """Retrieve and save all species records without downloading images."""
        ...

    @abstractmethod
    def download_images(self) -> None:
        """Download images using saved records, without retrieving records."""
        ...

    @abstractmethod
    def download(self) -> None:
        """Retrieve all species records first, then download their images."""
        ...
