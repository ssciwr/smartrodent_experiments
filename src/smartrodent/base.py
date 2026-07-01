from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


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
        dataset_output_path: str,
        path_to_labels: str,
        class_names: list[str],
        train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        img_types=[".jpg", ".jpeg", ".png"],
        rng_seed: int = 42,
    ):
        self.path_to_image_data = path_to_image_data

        if Path(self.path_to_image_data).exists() is False:
            raise ValueError(
                f"Path to image data {self.path_to_image_data} does not exist"
            )

        self.dataset_output_path = dataset_output_path
        self.path_to_labels = path_to_labels
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

        (p / "labels").mkdir(parents=True, exist_ok=True)
        (p / "images").mkdir(parents=True, exist_ok=True)

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
