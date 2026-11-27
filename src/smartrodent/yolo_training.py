"""Training helpers for SmartRodent YOLO experiments."""

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import mlflow
import pandas as pd
import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from ultralytics import YOLO, settings

from .training_validation import Nonblank, parse_settings


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MlflowTrackingConfiguration:
    """Configure local MLflow tracking for training and tuning operations.

    Args:
        tracking_uri: MLflow tracking URI, e.g. ``sqlite:///mlflow.db``.
        train_experiment_name: Experiment that training runs are logged to.
        tune_experiment_name: Experiment that tuning runs are logged to.
    """

    tracking_uri: Nonblank
    train_experiment_name: Nonblank
    tune_experiment_name: Nonblank

    def configure(self, operation: Literal["train", "tune"]) -> None:
        """Configure MLflow and Ultralytics for one tracked operation.

        Args:
            operation: Whether the imminent operation is training or tuning.
        """
        experiment_name = (
            self.train_experiment_name
            if operation == "train"
            else self.tune_experiment_name
        )
        # Ultralytics' MLflow callback reads these process settings when it starts
        # its run, while the direct calls create/select the configured experiment.
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        settings.update({"mlflow": True})


class YoloDetectionTrainer:
    """Train a YOLO detector and record validation metrics through callbacks.

    This class keeps the training loop itself inside Ultralytics. It only resolves the
    dataset YAML, registers a callback, and stores the validation metrics that
    Ultralytics has already computed at the end of each fit epoch.
    """

    def __init__(
        self,
        train_dataset: str | Path,
        model_name: str | Path = "yolo26n.pt",
        include_macro_f1: bool = True,
        return_format: Literal["dict", "dataframe"] = "dict",
        tune_kwargs: dict[str, Any] | None = None,
        export_kwargs: dict[str, Any] | None = None,
        mlflow_config: Mapping[str, str] | None = None,
        event_callback: list[tuple[str, Callable]] | None = None,
        trainer=None,
        **train_kwargs,
    ):
        """Initialize a YOLO detector training run.

        Args:
            train_dataset: Dataset directory containing ``data.yaml``/``config.yaml``,
                or a direct path to a YOLO data YAML file.
            model_name: Ultralytics model name or local weights path.
            include_macro_f1: Whether to add the mean of available per-class F1 scores
                to each epoch's metric record.
            return_format: Default format returned by ``train`` when no explicit
                format is passed there.
            tune_kwargs: Arguments forwarded to ``YOLO.tune``. Keep tuner-specific
                options here, such as ``iterations`` and ``space``.
            export_kwargs: Kwargs forwarded to ``YOLO.model.export``. Check out the ultralytics export documentation for more.
            mlflow_config: Optional tracking URI and experiment names loaded from YAML.
            event_callback: Optional extra Ultralytics callback as ``(event, fn)``.
                This is registered in addition to the built-in metric recorder.
            trainer: Optional Ultralytics trainer passed through to ``YOLO.train``.
            **train_kwargs: Additional keyword arguments forwarded to ``YOLO.train``.
        """
        self.train_dataset = Path(train_dataset)
        self.model_name = model_name
        self.include_macro_f1 = include_macro_f1
        self.return_format = return_format
        self.tune_kwargs = tune_kwargs or {}
        self.event_callback = event_callback
        self.trainer = trainer
        self.train_kwargs = train_kwargs
        self.history: dict[int, dict[str, Any]] = {}
        self.export_kwargs = export_kwargs
        self.mlflow_config = (
            parse_settings(MlflowTrackingConfiguration, mlflow_config)
            if mlflow_config is not None
            else None
        )
        self.model = YOLO(str(self.model_name))

    @classmethod
    def from_config(cls, config_path: str | Path) -> "YoloDetectionTrainer":
        """Create a detector trainer from a YAML config file.

        Args:
            config_path: Path to a YAML file with top-level trainer settings,
                ``train_kwargs`` for ``YOLO.train``, and optional ``tune_kwargs`` for
                ``YOLO.tune``.

        Returns:
            A configured ``YoloDetectionTrainer`` instance.

        Notes:
            ``event_callback`` and ``trainer`` are intentionally left as Python-side
            options because they are callable/runtime objects and do not map cleanly
            to YAML.
        """
        config_path = Path(config_path)
        config = yaml.safe_load(config_path.read_text())
        train_kwargs = config.pop("train_kwargs", {})
        tune_kwargs = config.pop("tune_kwargs", {})
        export_kwargs = config.pop("export_kwargs", {})
        mlflow_config = config.pop("mlflow", None)
        return cls(
            **config,
            tune_kwargs=tune_kwargs,
            export_kwargs=export_kwargs,
            mlflow_config=mlflow_config,
            **train_kwargs,
        )

    def data_yaml(self) -> Path:
        """Resolve the configured dataset to a YOLO data YAML path.

        Returns:
            The input path itself when ``train_dataset`` is a file; otherwise the first
            matching ``data.yaml`` or ``config.yaml`` inside the dataset directory.

        Raises:
            FileNotFoundError: If a dataset directory does not contain a recognized
                YOLO data YAML.
        """
        if self.train_dataset.is_file():
            return self.train_dataset

        for name in ("data.yaml", "config.yaml"):
            candidate = self.train_dataset / name
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No data.yaml or config.yaml found in {self.train_dataset}"
        )

    def collect_metrics(self, trainer) -> None:
        """Record one epoch of validation metrics from an Ultralytics trainer.

        Args:
            trainer: Ultralytics trainer object passed by the ``on_fit_epoch_end``
                callback. At this point validation metrics are available on
                ``trainer.metrics`` and ``trainer.validator.metrics``.
        """
        # Ultralytics also calls this event during final best-model evaluation. That
        # pass is useful for logging but should not create a duplicate history entry.
        if trainer.epoch >= trainer.epochs:
            return

        metrics = trainer.metrics
        det_metrics = trainer.validator.metrics
        names = det_metrics.names
        class_indices = {name: index for index, name in names.items()}

        # Start with every known class so classes with no validation detections still
        # appear in the returned structure with explicit None metrics; Ultralytics'
        # summary only lists classes that occur in the validation labels.
        classes = {
            name: {
                "class_index": class_index,
                "precision": None,
                "recall": None,
                "f1": None,
                "ap50": None,
                "ap50_95": None,
            }
            for class_index, name in sorted(names.items())
        }
        # summary() rounds to 5 decimals; that precision is ample for comparing runs.
        for row in det_metrics.summary():
            classes[row["Class"]] = {
                "class_index": class_indices[row["Class"]],
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "f1": float(row["Box-F1"]),
                "ap50": float(row["mAP50"]),
                "ap50_95": float(row["mAP50-95"]),
            }

        map50 = float(metrics["metrics/mAP50(B)"])
        map50_95 = float(metrics["metrics/mAP50-95(B)"])
        self.history[trainer.epoch + 1] = {
            "map50": map50,
            "map50_95": map50_95,
            # These are aliases, not separate mean-IoU scalars. They make it explicit
            # that the reported detector quality is AP at IoU thresholds.
            "iou_ap50": map50,
            "iou_map50_95": map50_95,
            "classes": classes,
        }

        if self.include_macro_f1:
            f1s = [c["f1"] for c in classes.values() if c["f1"] is not None]
            self.history[trainer.epoch + 1]["macro_f1"] = (
                sum(f1s) / len(f1s) if f1s else None
            )

    def train(
        self,
        return_format: Literal["dict", "dataframe"] | None = None,
    ) -> dict[int, dict[str, Any]] | pd.DataFrame:
        """Run training and return the recorded validation metric history.

        Args:
            return_format: Optional override for the configured return format.
                ``"dict"`` gives the nested epoch history, while ``"dataframe"``
                gives a long-form table with one row per epoch/class.

        Returns:
            Validation metric history in the requested format.

        Raises:
            ValueError: If ``return_format`` is not ``"dict"`` or ``"dataframe"``.
        """
        return_format = return_format or self.return_format
        if self.mlflow_config is not None:
            self.mlflow_config.configure("train")
        self.model.add_callback("on_fit_epoch_end", self.collect_metrics)
        if self.event_callback is not None:
            for event, callback in self.event_callback:
                self.model.add_callback(event, callback)

        self.model.train(
            trainer=self.trainer,
            data=str(self.data_yaml()),
            **self.train_kwargs,
        )

        if return_format == "dict":
            return self.history
        if return_format == "dataframe":
            return self.dataframe()
        raise ValueError("return_format must be 'dict' or 'dataframe'")

    def tune_space(self) -> dict[str, Any]:
        """Return the configured hyperparameter search space for YOLO tuning.

        Returns:
            Search-space values from ``self.tune_kwargs["space"]`` with JSON lists
            converted to tuples, which is the range format Ultralytics' native tuner
            expects.
        """
        space = self.tune_kwargs.get("space", {})
        return {
            parameter: tuple(values) if isinstance(values, list) else values
            for parameter, values in space.items()
        }

    def tune(self) -> Any:
        """Run Ultralytics' native genetic hyperparameter tuner.

        Returns:
            The result returned by ``YOLO.tune``.

        Notes:
            ``train_kwargs`` provide the normal training defaults, while
            ``tune_kwargs`` can override them for shorter or cheaper tuning runs.
        """
        if self.mlflow_config is not None:
            self.mlflow_config.configure("tune")
        tune_kwargs = self.tune_kwargs.copy()
        if "space" in tune_kwargs:
            tune_kwargs["space"] = self.tune_space()

        # Merge before the call so tune-specific settings naturally override the
        # normal training defaults without passing duplicate keyword arguments.
        kwargs = {**self.train_kwargs, **tune_kwargs}

        return self.model.tune(data=str(self.data_yaml()), **kwargs)

    def dataframe(self) -> pd.DataFrame:
        """Return recorded metrics as a long-form table.

        Returns:
            A DataFrame with one row per epoch/class. Global epoch metrics such as
            ``map50`` and ``map50_95`` are repeated on each class row for convenient
            filtering and plotting.
        """
        rows = []
        for epoch, epoch_metrics in self.history.items():
            global_metrics = {
                key: value for key, value in epoch_metrics.items() if key != "classes"
            }
            for class_name, class_metrics in epoch_metrics["classes"].items():
                rows.append(
                    {
                        "epoch": epoch,
                        "class_name": class_name,
                        **global_metrics,
                        **class_metrics,
                    }
                )
        return pd.DataFrame(rows)

    def export(self) -> str:
        export_kwargs = self.export_kwargs or {}

        return self.model.export(**export_kwargs)
