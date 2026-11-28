"""Standalone Ultralytics training, tuning and export for detection and classification."""

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypedDict

import mlflow
import numpy as np
import pandas as pd
import torch
from pydantic import AfterValidator, ConfigDict, Field, model_validator, with_config
from pydantic.dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

from .config_loading import load_section
from .training_validation import (
    Count,
    Name,
    Nonblank,
    PathSetting,
    Probability,
    parse_settings,
)

# Arguments that the settings own: passing them through the kwargs would let two
# places decide the same fact.
_RESERVED_KEYS = ("data", "model", "task")
_PER_CLASS_CSV = "per_class_metrics.csv"


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


# ------------------------------------------------------------------------- settings


def _reject_reserved_keys(kwargs):
    """Reject keys that are configured through dedicated settings fields."""
    reserved = [key for key in _RESERVED_KEYS if key in kwargs]
    if reserved:
        raise ValueError(f"set {reserved} as top-level settings, not in the kwargs")
    return kwargs


def _reject_opset(kwargs):
    """Reject ``opset`` so Ultralytics keeps using its own best ONNX opset."""
    if "opset" in kwargs:
        raise ValueError("opset must not be set: Ultralytics picks its best opset")
    return kwargs


def _check_bounds(bounds):
    """Require ``low <= high`` in a search-space range."""
    if bounds[0] > bounds[1]:
        raise ValueError(f"low bound {bounds[0]} is above high bound {bounds[1]}")
    return bounds


# int | float (not float alone) keeps integer ranges such as close_mosaic as given.
type Bound = (
    Annotated[int, Field(strict=True)]
    | Annotated[float, Field(strict=True, allow_inf_nan=False)]
)
# (low, high) or (low, high, mutation gain). Lists from YAML become tuples, which is
# the range format Ultralytics' tuner expects.
type Bounds = Annotated[
    tuple[Bound, Bound] | tuple[Bound, Bound, Probability],
    AfterValidator(_check_bounds),
]


@with_config(ConfigDict(extra="allow"))
class _TrainKwargs(TypedDict):
    """Native ``YOLO.train`` arguments; only the ones we depend on are required."""

    project: PathSetting
    optimizer: Name


@with_config(ConfigDict(extra="allow"))
class _ExportKwargs(TypedDict):
    """Native ``YOLO.export`` arguments."""

    format: Name


@with_config(ConfigDict(extra="allow"))
class _TuneKwargs(TypedDict):
    """Tuner options; every other key overrides the matching ``train_kwargs`` value."""

    iterations: Count
    space: Annotated[dict[Name, Bounds], Field(min_length=1)]


# The keys that configure the tuner itself; every other tune key overrides a train one.
_TUNER_KEYS = tuple(_TuneKwargs.__annotations__)

type TrainKwargs = Annotated[_TrainKwargs, AfterValidator(_reject_reserved_keys)]
type ExportKwargs = Annotated[_ExportKwargs, AfterValidator(_reject_opset)]
type TuneKwargs = Annotated[_TuneKwargs, AfterValidator(_reject_reserved_keys)]


def _validate_with_ultralytics(label: str, arguments: Mapping[str, Any]) -> None:
    """Let Ultralytics' own config validation judge names and value types.

    Args:
        label: Which kwargs are validated, for the error message.
        arguments: Native Ultralytics arguments.

    Raises:
        ValueError: If Ultralytics rejects an argument name or type.
    """
    try:
        get_cfg(overrides=dict(arguments))
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: {error}") from error


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class UltralyticsSettings:
    """Validated configuration of one Ultralytics training / tuning / export setup.

    Args:
        task: ``detect`` or ``classify``.
        model: Weights file or Ultralytics model name.
        dataset: Dataset root as created by ``create_yolo_dataset``.
        train_kwargs: Arguments of ``YOLO.train``; ``project`` and ``optimizer``
            are required.
        export_kwargs: Arguments of ``YOLO.export``; ``format`` is required.
        tune_kwargs: ``iterations``, ``space`` and any real overrides of
            ``train_kwargs`` for ``YOLO.tune``; ``None`` disables tuning.
        mlflow: Optional MLflow tracking; ``None`` turns Ultralytics' MLflow off.
    """

    task: Literal["detect", "classify"]
    model: PathSetting
    dataset: PathSetting
    train_kwargs: TrainKwargs
    export_kwargs: ExportKwargs
    tune_kwargs: TuneKwargs | None = None
    mlflow: MlflowTrackingConfiguration | None = None

    @model_validator(mode="after")
    def _validate_against_ultralytics(self):
        """Check kwargs against Ultralytics and the tune/train overlap rules."""
        _validate_with_ultralytics("train_kwargs", self.train_kwargs)
        if self.tune_kwargs is None:
            return self

        unknown = self.tune_kwargs["space"].keys() - DEFAULT_CFG_DICT.keys()
        if unknown:
            raise ValueError(
                f"tune space: not Ultralytics arguments: {sorted(unknown)}"
            )
        repeated = [
            key
            for key, value in self._tune_overrides.items()
            if key in self.train_kwargs and self.train_kwargs[key] == value
        ]
        if repeated:
            raise ValueError(
                f"tune_kwargs repeat the train_kwargs values of {repeated}: "
                "list only real overrides"
            )
        _validate_with_ultralytics("tune_kwargs", self._merged_train_arguments)
        return self

    @property
    def _tune_overrides(self) -> dict[str, Any]:
        """The tune kwargs that are overrides of train arguments."""
        return {
            key: value
            for key, value in self.tune_kwargs.items()
            if key not in _TUNER_KEYS
        }

    @property
    def _merged_train_arguments(self) -> dict[str, Any]:
        """Train arguments with the tune overrides applied on top."""
        return {**self.train_kwargs, **self._tune_overrides}

    @property
    def train_project(self) -> Path:
        """Absolute training project; a relative one resolves against the cwd."""
        # Ultralytics would put a relative project under its own runs_dir. absolute()
        # rather than resolve() so symlinks (e.g. DVC-managed dirs) are not followed.
        return Path(self.train_kwargs["project"]).absolute()

    @property
    def tune_project(self) -> Path:
        """Absolute tuning project: the tune override, else the train project."""
        if self.tune_kwargs is not None and "project" in self.tune_kwargs:
            return Path(self.tune_kwargs["project"]).absolute()
        return self.train_project

    @property
    def tune_arguments(self) -> dict[str, Any]:
        """Arguments for ``YOLO.tune`` besides ``data``.

        ``project`` is the one given in the settings; ``UltralyticsTrainer`` overrides
        it with the absolute, working-directory-based ``tune_project``.

        Raises:
            ValueError: If no ``tune_kwargs`` are configured.
        """
        if self.tune_kwargs is None:
            raise ValueError("no tune_kwargs configured: tuning is not set up")
        return {
            **self._merged_train_arguments,
            **{key: self.tune_kwargs[key] for key in _TUNER_KEYS},
        }


# ----------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class EpochMetrics:
    """One validation epoch: global scores and per-class metrics.

    Args:
        summary: Global scores of the epoch, e.g. ``map50``.
        classes: Per-class metrics by class name, ordered by class index; values
            that are undefined for a class are NaN.
    """

    summary: Mapping[str, float]
    classes: Mapping[str, Mapping[str, float]]

    @property
    def macro_f1(self) -> float:
        """Mean F1 over the classes where it is defined; NaN if there is none."""
        f1s = [c["f1"] for c in self.classes.values() if not math.isnan(c["f1"])]
        return sum(f1s) / len(f1s) if f1s else math.nan

    def rows(self, epoch: int) -> list[dict[str, Any]]:
        """Long-form records, one per class, with the global scores repeated.

        Args:
            epoch: One-based epoch number.

        Returns:
            Records with ``epoch``, ``class_name``, the summary, ``macro_f1`` and
            the class metrics.
        """
        shared = {"epoch": epoch, **self.summary, "macro_f1": self.macro_f1}
        return [
            {**shared, "class_name": name, **metrics}
            for name, metrics in self.classes.items()
        ]


class UltralyticsTask(Protocol):
    """What differs between detection and classification."""

    def data_argument(self, dataset: Path) -> str:
        """Return the ``data`` argument Ultralytics expects for ``dataset``.

        Raises:
            FileNotFoundError: If the dataset lacks what the task needs.
        """
        ...

    def epoch_metrics(self, trainer) -> EpochMetrics:
        """Read the metrics of the epoch that ``trainer`` just validated."""
        ...


class DetectTask:
    """Detection: ``data.yaml`` in the dataset root, metrics from ``DetMetrics``."""

    def data_argument(self, dataset: Path) -> str:
        """Return the path of ``<dataset>/data.yaml``.

        Args:
            dataset: Dataset root.

        Raises:
            FileNotFoundError: If ``data.yaml`` is missing.
        """
        data_yaml = dataset / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"No data.yaml in {dataset}")
        return str(data_yaml)

    def epoch_metrics(self, trainer) -> EpochMetrics:
        """Collect mAP and per-class P/R/F1/AP from the validator's ``DetMetrics``.

        Args:
            trainer: Ultralytics trainer at ``on_fit_epoch_end``.

        Returns:
            The epoch's metrics.
        """
        det_metrics = trainer.validator.metrics
        names = det_metrics.names
        # Start with every known class: Ultralytics' summary only lists classes that
        # occur in the validation labels, the others stay explicitly undefined.
        undefined = dict.fromkeys(
            ("precision", "recall", "f1", "ap50", "ap50_95"), math.nan
        )
        classes = {
            name: {"class_index": index, **undefined}
            for index, name in sorted(names.items())
        }
        # summary() rounds to 5 decimals; ample for comparing runs.
        for row in det_metrics.summary():
            classes[row["Class"]].update(
                precision=float(row["Box-P"]),
                recall=float(row["Box-R"]),
                f1=float(row["Box-F1"]),
                ap50=float(row["mAP50"]),
                ap50_95=float(row["mAP50-95"]),
            )
        summary = {
            "map50": float(trainer.metrics["metrics/mAP50(B)"]),
            "map50_95": float(trainer.metrics["metrics/mAP50-95(B)"]),
        }
        return EpochMetrics(summary=summary, classes=classes)


class ClassifyTask:
    """Classification: ``train/`` in the dataset root, per-class metrics via sklearn."""

    def data_argument(self, dataset: Path) -> str:
        """Return the dataset root, which Ultralytics reads as an image folder.

        Args:
            dataset: Dataset root.

        Raises:
            FileNotFoundError: If ``<dataset>/train`` is missing.
        """
        if not (dataset / "train").is_dir():
            raise FileNotFoundError(f"No train directory in {dataset}")
        return str(dataset)

    def epoch_metrics(self, trainer) -> EpochMetrics:
        """Collect top-k accuracy and per-class P/R/F1 from the validator's outputs.

        Args:
            trainer: Ultralytics trainer at ``on_fit_epoch_end``.

        Returns:
            The epoch's metrics.
        """
        validator = trainer.validator
        names = validator.names
        y_true = torch.cat(validator.targets).numpy()
        y_pred = torch.cat(validator.pred)[:, 0].numpy()  # top-1 of the sorted top-N
        labels = list(range(validator.nc))
        # zero_division=nan marks undefined values as NaN instead of faking a 0.
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=np.nan
        )
        predictions = np.bincount(y_pred, minlength=validator.nc)
        classes = {
            names[index]: {
                "class_index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "predictions": int(predictions[index]),
            }
            for index in labels
        }
        summary = {
            "accuracy_top1": float(trainer.metrics["metrics/accuracy_top1"]),
            "accuracy_top5": float(trainer.metrics["metrics/accuracy_top5"]),
        }
        return EpochMetrics(summary=summary, classes=classes)


def ultralytics_task(name: str) -> UltralyticsTask:
    """Create the task strategy for a task name.

    Args:
        name: ``detect`` or ``classify``.

    Returns:
        The matching task.

    Raises:
        ValueError: If the name is not a supported task.
    """
    if name == "detect":
        return DetectTask()
    elif name == "classify":
        return ClassifyTask()
    else:
        raise ValueError(f"unknown task {name!r}: expected 'detect' or 'classify'")


# --------------------------------------------------------------------------- trainer


class PerClassMetricsRecorder:
    """Ultralytics ``on_fit_epoch_end`` callback that records per-class metrics.

    Rows accumulate over epochs and are rewritten to ``per_class_metrics.csv`` in
    the run directory after every epoch, so a crashed run keeps its history.

    Args:
        task: Task that reads the metrics of an epoch.
    """

    def __init__(self, task: UltralyticsTask):
        self._task = task
        self._rows: list[dict[str, Any]] = []

    def __call__(self, trainer) -> None:
        """Record one epoch; the final best-model evaluation is ignored.

        Args:
            trainer: Ultralytics trainer passed by the callback.
        """
        # Ultralytics fires this event again for its final best-model evaluation, run
        # by a validator outside of training. It repeats an epoch, so skip it. The
        # epoch counter cannot tell: after early stopping it is below ``epochs``.
        if trainer.validator.training is False:
            return
        self._rows.extend(self._task.epoch_metrics(trainer).rows(trainer.epoch + 1))
        self.frame().to_csv(Path(trainer.save_dir) / _PER_CLASS_CSV, index=False)

    def frame(self) -> pd.DataFrame:
        """Return the recorded metrics, one row per epoch and class."""
        return pd.DataFrame(self._rows)


class UltralyticsTrainer:
    """Train, tune or export one Ultralytics model of a configured task.

    Ultralytics keeps the training loop; this class wires the dataset, MLflow and
    the per-class metrics recorder around it. An instance fits at most once.

    Args:
        config: Validated settings.

    Raises:
        FileNotFoundError: If the dataset lacks what the task needs.
    """

    def __init__(self, config: UltralyticsSettings):
        self.settings = config
        self._task = ultralytics_task(config.task)
        # Check the dataset before building the model, which may download weights.
        self._data = self._task.data_argument(Path(config.dataset))
        self._model = YOLO(str(config.model), task=config.task)
        self._fitted = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> "UltralyticsTrainer":
        """Create a trainer from a YAML file holding the settings at its root.

        Args:
            config_path: Path to the YAML configuration.

        Returns:
            The configured trainer.
        """
        return cls(parse_settings(UltralyticsSettings, load_section(config_path)))

    @property
    def train_project(self) -> Path:
        """Absolute directory that training runs are written to."""
        return self.settings.train_project

    @property
    def tune_project(self) -> Path:
        """Absolute directory that tuning runs are written to."""
        return self.settings.tune_project

    def train(self) -> pd.DataFrame:
        """Run Ultralytics training and record per-class metrics every epoch.

        Returns:
            One row per epoch and class; also written to ``per_class_metrics.csv``.

        Raises:
            RuntimeError: If this trainer already trained or tuned.
        """
        self._claim_fit()
        self._track("train")
        recorder = PerClassMetricsRecorder(self._task)
        self._model.add_callback("on_fit_epoch_end", recorder)
        arguments = {**self.settings.train_kwargs, "project": str(self.train_project)}
        self._model.train(data=self._data, **arguments)
        return recorder.frame()

    def tune(self) -> Path:
        """Run Ultralytics' genetic hyperparameter tuner.

        Returns:
            The tuning project directory. Ultralytics writes the results (best
            weights, ``best_hyperparameters.yaml``) to a ``<name>`` (default
            ``tune``) subdirectory of it, numbered if it already exists.

        Raises:
            ValueError: If no ``tune_kwargs`` are configured.
            RuntimeError: If this trainer already trained or tuned.
        """
        arguments = {
            **self.settings.tune_arguments,
            "project": str(self.tune_project),
        }
        self._claim_fit()
        self._track("tune")
        self._model.tune(data=self._data, **arguments)
        return self.tune_project

    def export(self) -> str:
        """Export the model with ``export_kwargs``.

        Returns:
            The path of the exported model.
        """
        return self._model.export(**self.settings.export_kwargs)

    def _claim_fit(self) -> None:
        """Allow one fit per instance: the model carries state from the first one."""
        if self._fitted:
            raise RuntimeError("this trainer already trained or tuned")
        self._fitted = True

    def _track(self, operation: Literal["train", "tune"]) -> None:
        """Configure MLflow for the operation, or switch Ultralytics' MLflow off."""
        if self.settings.mlflow is None:
            settings.update({"mlflow": False})
        else:
            self.settings.mlflow.configure(operation)
