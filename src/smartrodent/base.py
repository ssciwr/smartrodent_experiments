from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch


class ImageFilterBase(ABC):
    def __init__(self, model: str, tol: float):
        self.tol = tol
        self._load_model(self.resolve_local_model(model))

    @abstractmethod
    def _load_model(self, model: str):
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
        pass

    @abstractmethod
    def filter_similarities(
        self,
        similarity: torch.Tensor | list | np.ndarray,
    ) -> tuple | list:
        pass

    @abstractmethod
    def decisions(self, imgs: list, decided: torch.Tensor | list | np.ndarray) -> dict:
        pass


class DataPreprocessorBase(ABC):
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    @abstractmethod
    def filter_raw_data(self):
        pass

    @abstractmethod
    def process_raw_data(self):
        pass

    @abstractmethod
    def finalize(self):
        pass


class YoloDatasetCreatorBase(ABC):
    def __init__(
        self,
        path_to_image_data: str,
        path_to_labels: str,
        dataset_output_path: str,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        img_types=[".jpg", ".jpeg", ".png"],
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
            create_detection_dirs: When true, create ``images/`` and ``labels/`` for
                YOLO detection datasets. Classification datasets use split/class
                folders instead and opt out of these directories.
        """
        self.path_to_image_data = path_to_image_data

        if Path(self.path_to_image_data).exists() is False:
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

        self.classes = dict()
        for i, class_name in enumerate(class_names):
            self.classes[i] = class_name

        p = Path(self.dataset_output_path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

        if create_detection_dirs:
            (p / "labels").mkdir(parents=True, exist_ok=True)
            (p / "images").mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _filter_labels(self, detections: list) -> list:
        pass

    @abstractmethod
    def _split_train_val_test(
        self,
    ) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
        pass

    @abstractmethod
    def _preprocess_labels(self, raw_labels: dict) -> dict:
        pass

    @abstractmethod
    def _write_labels(
        self,
        paths: dict[str, list[Path]],
        assignments: dict[str, list[Path]],
        preprocessed_labels: dict,
    ) -> Path:
        pass

    def __call__(self) -> str | Path:
        paths, assignments = self._split_train_val_test()

        preprocessed_labels = self._preprocess_labels(self.labels)

        return self._write_labels(paths, assignments, preprocessed_labels)
