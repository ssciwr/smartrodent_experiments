from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch


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
    """Protocol for objects that filter a set of images and report results."""

    def filter_data(self) -> pd.DataFrame:
        """Run filtering over the input data.

        Returns:
            pd.DataFrame: One row per processed item describing the outcome.
        """
        ...

    def save_results(self, res_df: pd.DataFrame) -> Path:
        """Persist filtering results to disk.

        Args:
            res_df: The results DataFrame returned by :meth:`filter_data`.

        Returns:
            Path: The path the results were written to.
        """
        ...

    def collect_image_paths(self) -> list[Path]:
        """Gather the image paths that should be filtered.

        Returns:
            list[Path]: Paths of the images to process.
        """


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


class ImageFilterBase(ABC):
    """Abstract base for image filters that load a model and score images."""

    def __init__(self, model: str, tol: float):
        self.tol = tol
        self._load_model(self.resolve_local_model(model))

    @abstractmethod
    def _load_model(self, model: str):
        """Load the model identified by ``model``."""
        pass

    def resolve_local_model(self, model_name: str | Path) -> str:
        """Resolve bundled model weights relative to this file when present."""
        model_path = Path(model_name)
        if model_path.is_absolute():
            return str(model_path)

        bundled_path = Path(__file__).with_name(str(model_name))
        return str(bundled_path) if bundled_path.exists() else str(model_name)

    @abstractmethod
    def compute_similarity(
        self, images: torch.Tensor | list | np.ndarray
    ) -> torch.Tensor:
        """Compute similarities for the supplied images."""
        pass

    @abstractmethod
    def filter_similarities(
        self,
        similarity: torch.Tensor | list | np.ndarray,
    ) -> tuple | list:
        """Filter similarity scores and return the selected results."""
        pass

    @abstractmethod
    def decisions(self, imgs: list, decided: torch.Tensor | list | np.ndarray) -> dict:
        """Map filtered results back to their input images."""
        pass


class YoloDatasetCreatorBase(ABC):
    """Shared setup and interface for YOLO dataset creators."""

    def __init__(
        self,
        path_to_image_data: str,
        path_to_labels: str,
        dataset_output_path: str,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        img_types=(".jpg", ".jpeg", ".png"),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        IoU_threshold: float = 0.45,
        create_detection_dirs: bool = True,
    ):
        """Store common YOLO dataset creation settings.

        Args:
            path_to_image_data: Root directory for source images or source crops.
            path_to_labels: Root directory for label metadata.
            dataset_output_path: Destination root for the generated YOLO dataset.
            class_names: Class names in the order they should appear in YOLO metadata.
            train_val_test_split: Fractions for train, validation, and test splits.
            img_types: Image suffixes accepted when scanning source folders.
            rng_seed: Seed used for reproducible train/val/test splits.
            confidence_threshold: Minimum detector confidence to keep a label.
            IoU_threshold: NMS overlap threshold for duplicate detections.
            create_detection_dirs: Whether to create YOLO detection image/label dirs.
        """
        self.path_to_image_data = path_to_image_data
        if not Path(self.path_to_image_data).exists():
            raise ValueError(
                f"Path to image data {self.path_to_image_data} does not exist"
            )

        self.path_to_labels = path_to_labels
        self.confidence_threshold = confidence_threshold
        self.IoU_threshold = IoU_threshold
        self.dataset_output_path = dataset_output_path
        self.class_names = class_names
        self.train_frac, self.val_frac, self.test_frac = train_val_test_split
        self.labels = None

        if not np.isclose(self.train_frac + self.val_frac + self.test_frac, 1.0):
            raise ValueError(
                "train_val_test_split fractions must sum to 1.0, but got "
                f"{self.train_frac + self.val_frac + self.test_frac}"
            )

        self.img_types = img_types
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.classes = {i: name for i, name in enumerate(class_names)}

        output_path = Path(self.dataset_output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if create_detection_dirs:
            (output_path / "labels").mkdir(parents=True, exist_ok=True)
            (output_path / "images").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _filter_by_observation(self) -> dict[str, list[Path]]:
        """Select observations suitable for dataset creation."""
        pass

    @abstractmethod
    def _filter_labels(self, detections: list) -> list:
        """Filter raw detections according to creator settings."""
        pass

    @abstractmethod
    def _split_train_val_test(
        self,
    ) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
        """Assign selected images to the train, validation, and test splits."""
        pass

    @abstractmethod
    def _preprocess_labels(self, raw_labels: dict) -> dict:
        """Convert raw labels into the format expected by the writer."""
        pass

    @abstractmethod
    def _write_labels(
        self,
        paths: dict[str, list[Path]],
        assignments: dict[str, list[Path]],
        preprocessed_labels: dict,
    ) -> Path:
        """Write images and labels into the generated dataset."""
        pass
