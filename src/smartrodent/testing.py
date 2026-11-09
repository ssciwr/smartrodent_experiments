import json
from pathlib import Path
from typing import Any

import captum  # noqa: F401  # reserved for the upcoming interpretability work
import torch
import yaml
from torchmetrics import Metric, MetricCollection
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

from .base import TesterBase
from .utils import import_class, resolve_data_path, to_json_value


def build_metric_collection(
    metric_configs: dict[str, dict[str, Any]], num_classes: int
) -> MetricCollection:
    """Build a torchmetrics collection from name -> {class, kwargs} config entries.

    Args:
        metric_configs: Metric names mapped to a mapping with ``class``, the full
            import path of a ``torchmetrics.Metric`` subclass, and optional
            ``kwargs`` for its constructor.
        num_classes: Number of classes the tested model predicts. Any configured
            ``num_classes`` kwarg must equal it.

    Returns:
        A collection whose ``compute()`` returns a dict keyed by the config names.

    Raises:
        ValueError: The config is empty or malformed, a class path cannot be
            imported, or a ``num_classes`` kwarg disagrees with the model.
        TypeError: A class path refers to something that is not a torchmetrics
            ``Metric``.
    """
    if not metric_configs:
        raise ValueError("At least one metric must be configured")

    metrics = {}
    for name, metric_config in metric_configs.items():
        if "class" not in metric_config:
            raise ValueError(f"Metric '{name}' is missing its 'class' import path")
        kwargs = metric_config.get("kwargs", {})
        if "num_classes" in kwargs and kwargs["num_classes"] != num_classes:
            raise ValueError(
                f"Metric '{name}' has num_classes={kwargs['num_classes']}, "
                f"but the model predicts {num_classes} classes"
            )
        metric_class = import_class(metric_config["class"])
        if not (isinstance(metric_class, type) and issubclass(metric_class, Metric)):
            raise TypeError(
                f"Metric '{name}': {metric_config['class']} is not a torchmetrics Metric"
            )
        metrics[name] = metric_class(**kwargs)

    # compute_groups lets metrics with identical internal state (e.g. accuracy,
    # precision, recall and F1 with the same settings) share one state and one
    # update per batch instead of each keeping and updating its own copy.
    return MetricCollection(metrics, compute_groups=True)


class YoloTester(TesterBase):
    """Evaluate a trained YOLO classifier on a held-out split with configured metrics.

    The test data is a split directory with one subdirectory per class, e.g.
    ``datasets/stage8_yolo_classifier/test/<species>/*.jpg``. Ground-truth indices
    come from the model's own ``names`` mapping, because Ultralytics derives class
    indices at training time and these need not match any dataset-creator order.
    """

    metrics_file_name = "metrics.json"

    def __init__(
        self,
        model_path: str | Path,
        data_path: str | Path,
        output_path: str | Path,
        metrics: dict[str, dict[str, Any]],
        predict_kwargs: dict[str, Any] | None = None,
    ):
        """Construct a testing system for a given model with a set of configurable metrics.

        Validates paths, loads the model, collects the test images with their
        labels, and builds the metrics.

        Args:
            model_path: Path to trained Ultralytics classification weights (``.pt``).
            data_path: Split directory containing one subdirectory per class.
            output_path: Directory for the metrics JSON; created if missing.
            metrics: Metric names mapped to ``{"class": <torchmetrics class path>,
                "kwargs": {...}}``. See ``build_metric_collection``.
            predict_kwargs: Keyword arguments forwarded to ``YOLO.predict``, such as
                ``imgsz``, ``batch`` and ``device``. None forwards nothing.

        Raises:
            FileNotFoundError: The model file or data directory does not exist.
            ValueError: The model is not a classifier, a class directory is unknown
                to the model, the data contains no images, or the metric config is
                invalid.
            TypeError: A configured metric class is not a torchmetrics ``Metric``.
        """
        self.model_path = resolve_data_path(model_path)
        self.data_path = resolve_data_path(data_path)
        self.output_path = resolve_data_path(output_path)
        self.predict_kwargs = predict_kwargs or {}

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file {self.model_path} does not exist")
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Data directory {self.data_path} does not exist")

        self.model = YOLO(str(self.model_path))
        # predict() results only carry class probabilities for classifiers; the
        # detection case is deliberately left for later.
        if self.model.task != "classify":
            raise ValueError(
                f"Only classification models are supported, got task '{self.model.task}'"
            )

        self.image_paths, self.labels = self._collect_images()
        self.metrics = build_metric_collection(metrics, len(self.model.names))
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _collect_images(self) -> tuple[list[Path], list[int]]:
        """List images per class directory with their model class indices."""
        index_by_name = {name: index for index, name in self.model.names.items()}
        class_dirs = sorted(path for path in self.data_path.iterdir() if path.is_dir())
        unknown = [path.name for path in class_dirs if path.name not in index_by_name]
        if unknown:
            raise ValueError(f"Class directories unknown to the model: {unknown}")

        image_paths, labels = [], []
        for class_dir in class_dirs:
            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix[1:].lower() in IMG_FORMATS
                ):
                    image_paths.append(image_path)
                    labels.append(index_by_name[class_dir.name])
        if not image_paths:
            raise ValueError(
                f"No images found in class directories of {self.data_path}"
            )
        return image_paths, labels

    @classmethod
    def from_config(cls, config_path: str | Path) -> "YoloTester":
        """Build an instance from a YAML config file.

        Args:
            config_path: Path to a YAML file whose top-level keys are the
                constructor arguments: ``model_path``, ``data_path``,
                ``output_path``, ``metrics`` and optional ``predict_kwargs``.

        Returns:
            YoloTester: A new instance configured from the file.
        """
        config = yaml.safe_load(Path(config_path).read_text())
        return cls(**config)

    def test(self) -> dict[str, float | list]:
        """Predict every test image, compute the metrics and write them as JSON.

        Returns:
            Metric names mapped to their values; scalars as floats, per-class
            metrics and confusion matrices as (nested) lists. The same mapping is
            written to ``<output_path>/metrics.json``.
        """
        # Reset so repeated calls evaluate the data once rather than accumulating.
        self.metrics.reset()
        # Ultralytics' predictor already puts the model in eval mode; inference_mode
        # additionally guarantees no autograd state is built during the run.
        with torch.inference_mode():
            results = self.model.predict(
                source=[str(path) for path in self.image_paths],
                stream=True,
                **self.predict_kwargs,
            )
            # Results arrive in source order; strict zip fails loudly if there is
            # not exactly one result per image.
            for result, label in zip(results, self.labels, strict=True):
                probabilities = result.probs.data.float().cpu().unsqueeze(0)
                self.metrics.update(probabilities, torch.tensor([label]))

        computed = {
            name: to_json_value(value) for name, value in self.metrics.compute().items()
        }
        metrics_path = self.output_path / self.metrics_file_name
        metrics_path.write_text(json.dumps(computed, indent=2))
        return computed
