"""Shared abstractions for dataset creation."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class YoloDatasetCreatorBase(ABC):
    """Store and validate settings shared by YOLO dataset creators.

    Concrete creators own their output layouts because detection and classification
    datasets use fundamentally different directory structures.
    """

    def __init__(
        self,
        path_to_image_data: str | Path,
        path_to_labels: str | Path,
        dataset_output_path: str | Path,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        img_types: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        rng_seed: int = 42,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialize shared dataset creation settings.

        Args:
            path_to_image_data: Root containing source images or crops.
            path_to_labels: Root containing SpeciesNet prediction files.
            dataset_output_path: Destination directory for the generated dataset.
            class_names: Ordered names used in generated metadata.
            train_val_test_split: Train, validation, and test fractions.
            img_types: Accepted source-image suffixes.
            rng_seed: Seed used for deterministic splitting.
            confidence_threshold: Minimum accepted detection confidence.
            iou_threshold: Maximum Intersection over Union (IoU) allowed between
                retained detections during non-maximum suppression. Values range
                from 0 (no overlap) to 1 (identical boxes); detections overlapping
                by more than this threshold are treated as duplicates.

        Raises:
            ValueError: If an input path is absent, fractions are invalid, or no
                classes are configured.
        """
        self.path_to_image_data = Path(path_to_image_data)
        self.path_to_labels = Path(path_to_labels)
        for path in (self.path_to_image_data, self.path_to_labels):
            if not path.exists():
                raise ValueError(f"Input path {path} does not exist")

        fractions = tuple(train_val_test_split)
        if len(fractions) != 3 or any(fraction < 0 for fraction in fractions):
            raise ValueError(
                "train_val_test_split must contain three non-negative values"
            )
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError(
                "train_val_test_split fractions must sum to 1.0, "
                f"but got {sum(fractions)}"
            )
        if not class_names:
            raise ValueError("class_names must contain at least one class")

        self.dataset_output_path = Path(dataset_output_path)
        self.dataset_output_path.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names)
        self.classes = dict(enumerate(self.class_names))
        self.train_frac, self.val_frac, self.test_frac = fractions
        self.img_types = tuple(suffix.lower() for suffix in img_types)
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    @abstractmethod
    def __call__(self) -> Path:
        """Build the configured dataset and return its output directory."""
