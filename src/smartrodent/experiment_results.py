"""Report completed training runs and rank their configured tracking metrics."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from smartrodent.training_validation import FiniteNumber, Mode, Nonblank, settings_dict


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Objective:
    """Identify the metric and direction used to compare trained results.

    Args:
        metric: Metric reported by the experiment.
        mode: Whether lower or higher values of the metric are better.
    """

    metric: Nonblank
    mode: Mode

    def check(self, other: "Objective") -> None:
        """Verify that two objectives compare the same metric in the same direction.

        Args:
            other: Objective of another experiment.

        Raises:
            ValueError: The objectives differ in metric or direction.
        """
        if self.metric != other.metric:
            raise ValueError("Experiments must use the same tracked metric")
        if self.mode != other.mode:
            raise ValueError("Experiments must use the same tracked mode")

    def select_best(self, results: Sequence["ExperimentResult"]) -> "ExperimentResult":
        """Choose a result using the objective and a shared validation split.

        Args:
            results: Nonempty completed runs ordered by the caller's tie preference.

        Returns:
            The original winning result. The first result wins an exact tie.

        Raises:
            ValueError: A result has a different objective or split, or lacks one.
        """
        split = results[0].validation_dataset
        if split is None:
            raise ValueError("Automatic experiment ranking requires validation data")
        split = split.resolve()
        for result in results:
            if result.validation_dataset is None:
                raise ValueError(
                    "Automatic experiment ranking requires validation data"
                )
            self.check(result.objective)
            if result.validation_dataset.resolve() != split:
                raise ValueError("Experiments must use the same validation dataset")
        rank = min if self.mode == "min" else max
        return rank(results, key=lambda result: result.best_value)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ExperimentResult:
    """Describe a completed run without inspecting checkpoint files.

    Args:
        output_directory: Directory containing the run's artifacts.
        last_checkpoint: Latest completed checkpoint.
        best_checkpoint: Checkpoint selected by the configured tracking metric.
        best_epoch: Epoch stored in the best checkpoint.
        tracked_metric: Name of the metric used to choose the best epoch.
        tracked_mode: ``min`` or ``max`` comparison direction.
        best_value: Original metric value, not an internally sign-inverted score.
        validation_dataset: Validation split, or ``None`` for training-only runs.
        mlflow_run_id: Tracking run ID when MLflow is enabled.
        registered_model_version: Published classifier version, when registered.
        onnx_model: ONNX file exported from the best checkpoint, when export is configured.
    """

    output_directory: Path
    last_checkpoint: Path
    best_checkpoint: Path
    best_epoch: int
    tracked_metric: Nonblank
    tracked_mode: Mode
    best_value: FiniteNumber
    validation_dataset: Path | None = None
    mlflow_run_id: str | None = None
    registered_model_version: str | None = None
    onnx_model: Path | None = None

    @property
    def objective(self) -> Objective:
        """Return the objective while preserving the serialized result fields."""
        return Objective(self.tracked_metric, self.tracked_mode)

    def to_dict(self) -> dict:
        """Serialize result paths without changing scores or optional registry metadata.

        Returns:
            YAML-compatible fields retaining the original, unnormalized paths.
        """
        return settings_dict(self)


def select_best_experiment(results: Sequence[ExperimentResult]) -> ExperimentResult:
    """Choose a validated run using its configured tracking metric and direction.

    Args:
        results: Completed results in run order, sharing a metric, direction,
            and validation split. The caller supplies the intended candidate set;
            this function does not discover runs or load checkpoint files.

    Returns:
        The original winning result. The first result wins an exact tie.

    Raises:
        ValueError: No results exist, validation is missing, or results report
            incompatible metrics, directions, or validation splits.
    """
    if not results:
        raise ValueError("Cannot select a winner from an empty set of results")
    return results[0].objective.select_best(results)
