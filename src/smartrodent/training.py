"""Training helpers for SmartRodent YOLO experiments."""

import json
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import yaml
from ultralytics import YOLO


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
        self.model = YOLO(str(self.model_name))

    @classmethod
    def from_config(cls, config_path: str | Path) -> "YoloDetectionTrainer":
        """Create a detector trainer from a JSON config file.

        Args:
            config_path: Path to a JSON file with top-level trainer settings,
                ``train_kwargs`` for ``YOLO.train``, and optional ``tune_kwargs`` for
                ``YOLO.tune``.

        Returns:
            A configured ``YoloDetectionTrainer`` instance.

        Notes:
            ``event_callback`` and ``trainer`` are intentionally left as Python-side
            options because they are callable/runtime objects and do not map cleanly
            to JSON.
        """
        config_path = Path(config_path)
        config = json.loads(config_path.read_text())
        train_kwargs = config.pop("train_kwargs", {})
        tune_kwargs = config.pop("tune_kwargs", {})
        return cls(**config, tune_kwargs=tune_kwargs, **train_kwargs)

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
        box = det_metrics.box
        names = det_metrics.names
        if isinstance(names, list):
            names = dict(enumerate(names))

        # Start with every known class so classes with no validation detections still
        # appear in the returned structure with explicit None metrics.
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

        # Ultralytics stores per-class arrays in the order of box.ap_class_index, not
        # necessarily in raw class-index order.
        for i, class_index in enumerate(box.ap_class_index):
            class_index = int(class_index)
            class_name = names.get(class_index, str(class_index))
            classes[class_name] = {
                "class_index": class_index,
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "f1": float(box.f1[i]),
                "ap50": float(box.ap50[i]),
                "ap50_95": float(box.ap[i]),
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


class YoloClassificationTrainer(YoloDetectionTrainer):
    """Train a YOLO image classifier and record validation metrics by epoch.

    Ultralytics classification fine-tuning expects a split/class folder layout, for
    example ``train/mouse/*.jpg`` and ``val/rat/*.jpg``. This subclass keeps the
    detection trainer's simple train/tune/export wrapper but changes the dataset
    resolver and metrics collector for classification.
    """

    def __init__(
        self,
        train_dataset: str | Path,
        model_name: str | Path = "yolo26n-cls.pt",
        include_macro_f1: bool = True,
        return_format: Literal["dict", "dataframe"] = "dict",
        tune_kwargs: dict[str, Any] | None = None,
        export_kwargs: dict[str, Any] | None = None,
        event_callback: list[tuple[str, Callable]] | None = None,
        trainer=None,
        **train_kwargs,
    ):
        """Initialize a YOLO classification training run.

        Args:
            train_dataset: Classification dataset root containing ``train/`` and
                usually ``val/`` class folders, or a ``data.yaml`` whose ``path``
                points at that root.
            model_name: Ultralytics classification model name or local weights path.
                Pretrained classifier weights use the ``-cls`` suffix, for example
                ``yolo26n-cls.pt``.
            include_macro_f1: Whether to add a confusion-matrix-derived macro F1
                score to each epoch's metric record.
            return_format: Default format returned by ``train`` when no explicit
                format is passed there.
            tune_kwargs: Arguments forwarded to ``YOLO.tune``.
            export_kwargs: Kwargs forwarded to ``YOLO.export``.
            event_callback: Optional extra Ultralytics callbacks as ``(event, fn)``.
            trainer: Optional Ultralytics trainer passed through to ``YOLO.train``.
            **train_kwargs: Additional keyword arguments forwarded to ``YOLO.train``.
        """
        super().__init__(
            train_dataset=train_dataset,
            model_name=model_name,
            include_macro_f1=include_macro_f1,
            return_format=return_format,
            tune_kwargs=tune_kwargs,
            export_kwargs=export_kwargs,
            event_callback=event_callback,
            trainer=trainer,
            **train_kwargs,
        )
        # Be explicit about the task so custom/local filenames do not have to end in
        # "-cls.pt" for Ultralytics to choose the classification code path.
        self.model = YOLO(str(self.model_name), task="classify")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "YoloClassificationTrainer":
        """Create a classification trainer from a JSON config file.

        Args:
            config_path: Path to a JSON file with top-level trainer settings,
                ``train_kwargs`` for ``YOLO.train``, and optional ``tune_kwargs`` for
                ``YOLO.tune``.

        Returns:
            A configured ``YoloClassificationTrainer`` instance.
        """
        config_path = Path(config_path)
        config = json.loads(config_path.read_text())
        train_kwargs = config.pop("train_kwargs", {})
        tune_kwargs = config.pop("tune_kwargs", {})
        return cls(**config, tune_kwargs=tune_kwargs, **train_kwargs)

    def data_yaml(self) -> Path:
        """Resolve the configured classification dataset root.

        The method keeps the inherited name for compatibility with
        ``YoloDetectionTrainer.train``. For classification, Ultralytics requires the
        dataset directory itself rather than a detection-style YAML file.

        Returns:
            Dataset root containing split folders such as ``train/`` and ``val/``.

        Raises:
            FileNotFoundError: If the dataset root or required ``train`` split is
                missing.
            ValueError: If a YAML file is supplied without a usable ``path`` entry, or
                if a non-YAML file is supplied.
        """
        dataset = self.train_dataset

        if dataset.is_file():
            if dataset.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(
                    "YOLO classification training expects a dataset directory, "
                    "or a data.yaml file with a 'path' entry."
                )

            metadata = yaml.safe_load(dataset.read_text()) or {}
            root = metadata.get("path")
            if root is None:
                raise ValueError(f"No 'path' entry found in {dataset}")

            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = dataset.parent / root_path
            dataset = root_path

        if not dataset.exists():
            raise FileNotFoundError(f"Classification dataset not found: {dataset}")
        if not dataset.is_dir():
            raise ValueError(f"Classification dataset must be a directory: {dataset}")
        if not (dataset / "train").is_dir():
            raise FileNotFoundError(
                f"No train split found at {dataset / 'train'}. "
                "Expected folders like train/<class_name>/*.jpg."
            )

        return dataset

    def collect_metrics(self, trainer) -> None:
        """Record one epoch of classifier validation metrics.

        Args:
            trainer: Ultralytics classification trainer passed by the
                ``on_fit_epoch_end`` callback.
        """
        # Ultralytics also fires this callback during final best-model evaluation.
        # Match the detector trainer by keeping only real training epochs.
        if trainer.epoch >= trainer.epochs:
            return

        metrics = trainer.metrics
        validator = trainer.validator
        names = getattr(validator, "names", getattr(trainer.model, "names", {}))
        if isinstance(names, list):
            names = dict(enumerate(names))

        classes = {
            name: {
                "class_index": class_index,
                "precision": None,
                "recall": None,
                "f1": None,
                "support": None,
                "predictions": None,
            }
            for class_index, name in sorted(names.items())
        }

        confusion_matrix = getattr(
            getattr(validator, "confusion_matrix", None), "matrix", None
        )
        if confusion_matrix is not None:
            # The matrix is indexed as [predicted_class, true_class]. Restrict to the
            # known class block because some Ultralytics versions allocate an extra
            # background row/column even for classifier plots.
            n_classes = len(names)
            matrix = confusion_matrix[:n_classes, :n_classes]

            for class_index, class_name in sorted(names.items()):
                # TODO: use scikit learn stuff here, no need to reimplement these ourselves.
                true_positive = float(matrix[class_index, class_index])
                predicted_total = float(matrix[class_index, :].sum())
                actual_total = float(matrix[:, class_index].sum())

                precision = true_positive / predicted_total if predicted_total else None
                recall = true_positive / actual_total if actual_total else None

                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision is not None
                    and recall is not None
                    and precision + recall > 0
                    else None
                )
                classes[class_name] = {
                    "class_index": class_index,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": int(actual_total),
                    "predictions": int(predicted_total),
                }

        top1 = metrics.get(
            "metrics/accuracy_top1", getattr(validator.metrics, "top1", None)
        )
        top5 = metrics.get(
            "metrics/accuracy_top5", getattr(validator.metrics, "top5", None)
        )

        self.history[trainer.epoch + 1] = {
            "accuracy_top1": float(top1) if top1 is not None else None,
            "accuracy_top5": float(top5) if top5 is not None else None,
            "classes": classes,
        }

        if self.include_macro_f1:
            f1s = [c["f1"] for c in classes.values() if c["f1"] is not None]
            self.history[trainer.epoch + 1]["macro_f1"] = (
                sum(f1s) / len(f1s) if f1s else None
            )

    def dataframe(self) -> pd.DataFrame:
        """Return recorded classifier metrics as a long-form table.

        Returns:
            A DataFrame with one row per epoch/class. Global epoch metrics such as
            top-1 and top-5 accuracy are repeated on each class row.
        """
        return super().dataframe()
