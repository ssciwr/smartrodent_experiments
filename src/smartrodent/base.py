from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


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
        ...


@runtime_checkable
class Filterable(Configurable, Protocol):
    """Protocol for per-species dataframe filters that only read images."""

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

    def retrieve_records(self) -> None:
        """Retrieve and save all species records without downloading images."""
        ...

    def download_images(self) -> None:
        """Download images using saved records, without retrieving records."""
        ...

    def download(self) -> None:
        """Retrieve all species records first, then download their images."""
        ...


class YoloDatasetCreatorBase(Configurable, ABC):
    """Load per-class records with existing split assignments, without sampling.

    Constructors only read metadata. They neither run inference nor create output
    directories. Source row order, additional columns, and oversampled rows remain
    available in ``records_by_species`` for the format conversion workflow.
    """

    def __init__(
        self,
        path_to_image_data: str | Path,
        dataset_output_path: str | Path,
        class_names: list[str] | None = None,
    ):
        """Load the selected species directories' ``records.csv`` files.

        Args:
            path_to_image_data: Input root containing per-species directories.
            dataset_output_path: Destination for the generated dataset.
            class_names: Optional species-directory selection in class-index order.
                None discovers all species directories in sorted order. Relative
                image paths are resolved against the shared dataset root, the parent
                of ``path_to_image_data``.

        Raises:
            ValueError: The input root or selected metadata is invalid.
            FileNotFoundError: Required records or source files are missing.
        """
        self.path_to_image_data = Path(path_to_image_data).resolve()
        self.dataset_output_path = Path(dataset_output_path).resolve()
        if not self.path_to_image_data.is_dir():
            raise ValueError(
                f"Input directory {self.path_to_image_data} does not exist"
            )

        self.class_names = self._select_classes(class_names)
        self.classes = {index: name for index, name in enumerate(self.class_names)}
        self.records_by_species = {
            species: self._load_records(species) for species in self.class_names
        }
        self._validate_split_assignments()

    def _select_classes(self, class_names: list[str] | None) -> list[str]:
        """Discover classes or validate an explicit ordered directory selection."""
        available = {
            path.name for path in self.path_to_image_data.iterdir() if path.is_dir()
        }
        selected = sorted(available) if class_names is None else list(class_names)
        if not selected:
            raise ValueError("At least one class directory must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("class_names must not contain duplicates")
        unknown = set(selected) - available
        if unknown:
            raise ValueError(f"Unknown class directories: {sorted(unknown)}")
        return selected

    def _load_records(self, species: str) -> pd.DataFrame:
        """Read a species frame, keeping row order, multiplicity, and extra columns."""
        records_path = self.path_to_image_data / species / "records.csv"
        if not records_path.is_file():
            raise FileNotFoundError(f"Missing species records: {records_path}")
        try:
            records = pd.read_csv(records_path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{species}: records.csv has no column header") from error

        missing = {"image_path", "dataset_split"} - set(records.columns)
        if missing:
            raise ValueError(f"{species}: missing required columns {sorted(missing)}")
        records["dataset_split"] = records["dataset_split"].replace(
            {"validation": "val"}
        )
        if not records["dataset_split"].isin(["train", "val", "test"]).all():
            raise ValueError(
                f"{species}: dataset_split must be train, validation/val, or test"
            )
        if (
            not records["image_path"]
            .map(lambda path: isinstance(path, str) and bool(path.strip()))
            .all()
        ):
            raise ValueError(
                f"{species}: image_path must contain nonempty path strings"
            )
        records["image_path"] = records["image_path"].map(
            lambda path: self._resolve_image_path(path, self.path_to_image_data.parent)
        )
        return records

    def _validate_split_assignments(self) -> None:
        """Reject images assigned across splits while preserving oversampled rows."""
        assignments = pd.concat(
            [
                records[["image_path", "dataset_split"]]
                for records in self.records_by_species.values()
            ],
            ignore_index=True,
        )
        splits_per_image = assignments.groupby("image_path")["dataset_split"].nunique()
        conflicting = splits_per_image[splits_per_image > 1]
        if not conflicting.empty:
            raise ValueError(
                f"Images assigned to multiple splits: {conflicting.index.tolist()}"
            )

    @staticmethod
    def _resolve_image_path(image_path: str, dataset_root: Path) -> str:
        """Resolve shared-root source paths explicitly and reject missing images."""
        path = Path(image_path)
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image_path does not refer to a file: {path}")
        return str(path)

    @abstractmethod
    def create(self) -> Path:
        """Create the configured dataset.

        Returns:
            The generated dataset's root directory.
        """
        raise NotImplementedError
