"""Run Optuna studies of one search space and rank their trials.

``HyperparameterStudy`` is trainer-independent: a trial runner turns each resolved
document into an ``ExperimentResult``. ``training_studies`` binds it to the trainers.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import Any, ClassVar

import optuna
import yaml
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from smartrodent.components import ComponentSpec
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.search_space import SearchSpace
from smartrodent.training_validation import (
    Count,
    Flag,
    MappingInput,
    Mode,
    Nonblank,
    PositiveNumber,
    parse_settings,
    settings_dict,
)

type EpochCallback = Callable[[int, float], None]
type TrialRunner = Callable[[dict, Path, EpochCallback], ExperimentResult]


class TrialConfigurationError(ValueError):
    """A sampled combination of otherwise valid values is not a valid configuration."""


# --------------------------------------------------------------------------------------
# Study settings and objective
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StudyConfiguration:
    """Settings of a ``tuning:`` section. The direction follows the tracked mode.

    Args:
        study_name: Optuna study name, also the resume key in persistent storage.
        storage: Optuna storage URL, e.g. ``sqlite:///optuna.db``; null is in-memory.
        n_trials: Trials to run in this invocation (Optuna semantics).
        sampler: Sampler class (in ``optuna.samplers``) and its parameters.
        pruner: Pruner class (in ``optuna.pruners``) and its parameters, or null for
            no pruning.
        load_if_exists: Resume an existing study of the same name.
        timeout: Optional wall-clock budget in seconds for this invocation.
    """

    study_name: Nonblank
    storage: Nonblank | None
    n_trials: Count
    sampler: MappingInput[ComponentSpec]
    pruner: MappingInput[ComponentSpec] | None
    load_if_exists: Flag = False
    timeout: PositiveNumber | None = None

    def __post_init__(self):
        """Construct the sampler and pruner once, so bad settings fail before training."""
        self.build_sampler()
        self.build_pruner()

    @classmethod
    def from_mapping(cls, values: Mapping) -> "StudyConfiguration":
        """Parse a ``tuning:`` section.

        Args:
            values: Study settings mapping.

        Returns:
            Validated settings.
        """
        return parse_settings(cls, values)

    def to_dict(self) -> dict:
        """Return YAML-compatible settings including defaulted keys."""
        return settings_dict(self)

    def build_sampler(self) -> BaseSampler:
        """Return a new configured sampler."""
        return self.sampler.build(optuna.samplers, BaseSampler)

    def build_pruner(self) -> BasePruner:
        """Return a new configured pruner; no pruner configured means never prune."""
        if self.pruner is None:
            return NopPruner()
        else:
            return self.pruner.build(optuna.pruners, BasePruner)


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StudyObjective:
    """The ranking every trial result must report to be comparable.

    Args:
        tracked_metric: Metric selecting each trial's best epoch.
        tracked_mode: ``min`` or ``max``; determines the study direction.
        validation_dataset: Shared validation split; required for ranking.
    """

    tracked_metric: Nonblank
    tracked_mode: Mode
    validation_dataset: Path | None

    def __post_init__(self):
        """Studies rank trials, which needs validation data."""
        if self.validation_dataset is None:
            raise ValueError(
                "An Optuna study requires a validation_dataset to rank its trials"
            )

    @property
    def direction(self) -> str:
        """Return Optuna's direction name for the tracked mode."""
        return "maximize" if self.tracked_mode == "max" else "minimize"

    def to_dict(self) -> dict:
        """Return a JSON-compatible identity used to recognize the same objective on resume."""
        return {
            "tracked_metric": self.tracked_metric,
            "tracked_mode": self.tracked_mode,
            "validation_dataset": str(self.validation_dataset.resolve()),
        }

    def validate(self, result: ExperimentResult) -> None:
        """Reject a result reporting a different objective.

        Args:
            result: Completed trial result.

        Raises:
            ValueError: Metric, mode, or validation split differ.
        """
        if result.tracked_metric != self.tracked_metric:
            raise ValueError(
                f"Trial reported tracked metric '{result.tracked_metric}', "
                f"but the study ranks '{self.tracked_metric}'"
            )
        if result.tracked_mode != self.tracked_mode:
            raise ValueError(
                f"Trial reported tracked mode '{result.tracked_mode}', "
                f"but the study uses '{self.tracked_mode}'"
            )
        if result.validation_dataset is None or (
            result.validation_dataset.resolve() != self.validation_dataset.resolve()
        ):
            raise ValueError("Trial result validation dataset does not match the study")


# --------------------------------------------------------------------------------------
# The generic study loop
# --------------------------------------------------------------------------------------


@result_dataclass(frozen=True)
class StudyResult:
    """All trials of a study, its completed results, and the selected winner.

    Args:
        output_directory: Study root containing ``study.yaml`` and trial directories.
        study_name: Optuna study name.
        direction: ``maximize`` or ``minimize``.
        trials: Report rows for every trial, in trial-number order, with decoded
            parameter values.
        results: Completed trial results, in trial-number order.
        selected: Best completed result (earliest on exact ties), or None.
        selected_trial_number: Trial number of ``selected``, or None.
        best_params: Decoded parameter values of ``selected``, or None.
        registered_champion_version: Registry alias version after promotion, when
            a classifier study registers models; not part of ``study.yaml``.
    """

    output_directory: Path
    study_name: str
    direction: str
    trials: tuple[dict, ...]
    results: tuple[ExperimentResult, ...]
    selected: ExperimentResult | None
    selected_trial_number: int | None
    best_params: dict | None
    registered_champion_version: str | None = None

    def to_dict(self) -> dict:
        """Return the portable ``study.yaml`` report.

        Returns:
            YAML-compatible study metadata.
        """
        selected = (
            {
                "number": self.selected_trial_number,
                "params": dict(self.best_params),
                "result": self.selected.to_dict(),
            }
            if self.selected is not None
            else None
        )
        return {
            "output_directory": str(self.output_directory),
            "study_name": self.study_name,
            "direction": self.direction,
            "trials": [dict(trial) for trial in self.trials],
            "selected_trial": selected,
        }


class _PruningReporter:
    """Epoch callback that reports values to a trial and stops it when pruned."""

    def __init__(self, trial: optuna.Trial):
        """Bind the callback to the running trial."""
        self.trial = trial

    def __call__(self, epoch: int, value: float) -> None:
        """Report one epoch's tracked value; raise TrialPruned when told to prune."""
        self.trial.report(value, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned after epoch {epoch} with value {value}")


class HyperparameterStudy:
    """Run trials of one search space with Optuna and report and rank them.

    Optuna's own ``study.optimize`` loop runs the trials: it honors n_trials and
    timeout, records pruned and failed trials, and lets exhaustive samplers stop
    the study, which a hand-written ask-and-tell loop would have to replicate. A
    grid is a space of categorical parameters searched by ``BruteForceSampler``.

    Args:
        space: Parameters to sample.
        settings: Study settings.
        run_trial: Trains one resolved document in the given directory, reporting
            epochs to the callback, and returns its result.
        output_directory: Study root for ``study.yaml`` and ``trial_NNNN`` directories.
        objective: Ranking every result must report.
        identity: Further label -> JSON-compatible value pairs that a resumed study must
            match, e.g. the checksum of the encoder every trial trains on.
    """

    # Anticipated per-trial failures: a diverging configuration and an invalid
    # combination of individually valid values. Both fail one trial only.
    _anticipated_failures: ClassVar[tuple] = (
        FloatingPointError,
        TrialConfigurationError,
    )

    def __init__(
        self,
        space: SearchSpace,
        settings: StudyConfiguration,
        run_trial: TrialRunner,
        *,
        output_directory: str | Path,
        objective: StudyObjective,
        identity: Mapping[str, Any] | None = None,
    ):
        """Store the study's parts; nothing is created before ``run``."""
        self.space = space
        self.settings = settings
        self.run_trial = run_trial
        self.output_directory = Path(output_directory)
        self.objective = objective
        self.identity = dict(identity or {})

    def preflight(self) -> None:
        """Check the output directory and stored study without running or recording anything.

        Used before long earlier work, e.g. the SupCon stage of a pipeline. The extra
        identity is not checked: it may only be known once that work is done.

        Raises:
            ValueError: ``run`` would refuse the output directory or the stored study.
        """
        self._check_output_directory()
        if self.settings.storage is None:
            return  # an in-memory study is always new
        try:
            study = optuna.load_study(
                study_name=self.settings.study_name, storage=self.settings.storage
            )
        except KeyError:
            return  # a new study in persistent storage
        if not self.settings.load_if_exists:
            raise self._duplicate_error()
        self._verify_direction(study)
        self._verify_identities(
            study, self._identities(include_extra=False), record=False
        )

    def run(self) -> StudyResult:
        """Run this invocation's trials and select the best completed trial overall.

        Returns:
            The study's report, including trials of earlier invocations when resumed.

        Raises:
            ValueError: The study cannot be created or resumed as configured, a result
                disagrees with the objective, or no trial completed.
            Exception: Unanticipated trial errors propagate unchanged after the trial
                is recorded as failed and the report is written.
        """
        self._check_output_directory()
        study = self._open_study()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        try:
            study.optimize(
                self._objective,
                n_trials=self.settings.n_trials,
                timeout=self.settings.timeout,
                catch=self._anticipated_failures,
                callbacks=[
                    lambda study, trial: self._write_report(self._summarize(study))
                ],
            )
        finally:
            # Optuna skips callbacks for re-raised errors; keep the report accurate.
            report = self._summarize(study)
            self._write_report(report)
        if report.selected is None:
            raise ValueError(
                f"Study '{self.settings.study_name}' has no completed trial"
            )
        return report

    def _check_output_directory(self) -> None:
        """Refuse a directory holding another study's trials, which would be overwritten."""
        report = self.output_directory / "study.yaml"
        if not report.is_file():
            return
        values = yaml.safe_load(report.read_text(encoding="utf-8"))
        existing = values.get("study_name") if isinstance(values, Mapping) else None
        if existing != self.settings.study_name:
            raise ValueError(
                f"{self.output_directory} holds trials of study '{existing}'; use another "
                f"output_directory for study_name '{self.settings.study_name}'"
            )

    def _open_study(self) -> optuna.Study:
        """Create or resume the study, refusing a changed direction or search space."""
        settings = self.settings
        try:
            study = optuna.create_study(
                study_name=settings.study_name,
                storage=settings.storage,
                sampler=settings.build_sampler(),
                pruner=settings.build_pruner(),
                direction=self.objective.direction,
                load_if_exists=settings.load_if_exists,
            )
        except optuna.exceptions.DuplicatedStudyError as error:
            raise self._duplicate_error() from error
        self._verify_direction(study)
        self._verify_identities(
            study, self._identities(include_extra=True), record=True
        )
        return study

    def _duplicate_error(self) -> ValueError:
        """Describe an existing study that was not requested to be resumed."""
        return ValueError(
            f"Study '{self.settings.study_name}' already exists in {self.settings.storage}; "
            "set load_if_exists: true to resume it or choose another study_name"
        )

    def _verify_direction(self, study: optuna.Study) -> None:
        """Refuse a stored study optimizing in the other direction."""
        stored_direction = study.direction.name.lower()
        if stored_direction != self.objective.direction:
            raise ValueError(
                f"Stored study '{self.settings.study_name}' has direction {stored_direction}, "
                f"but tracked_mode '{self.objective.tracked_mode}' requires {self.objective.direction}"
            )

    def _identities(self, *, include_extra: bool) -> dict[str, Any]:
        """Return label -> value pairs that define which trials are comparable."""
        identities = {
            "search space": self.space.to_dict(),
            "objective": self.objective.to_dict(),
        }
        if include_extra:
            identities.update(json.loads(json.dumps(self.identity, default=str)))
        return identities

    def _verify_identities(
        self, study: optuna.Study, identities: Mapping, *, record: bool
    ) -> None:
        """Compare stored identities; record missing ones of a new study when requested."""
        name = self.settings.study_name
        for label, expected in identities.items():
            key = label.replace(" ", "_")
            stored = study.user_attrs.get(key)
            if stored is None and study.trials:
                raise ValueError(
                    f"Stored study '{name}' has trials but no recorded {label}"
                )
            elif stored is None and record:
                study.set_user_attr(key, expected)
            elif stored is None:
                pass  # a new study: recorded when it runs
            elif stored != expected:
                raise ValueError(
                    f"Stored study '{name}' uses a different {label}; "
                    "start a new study (new study_name and output_directory) instead"
                )
            else:
                pass  # resuming with the same identity

    def _objective(self, trial: optuna.Trial) -> float:
        """Train one sampled configuration and return its configured best value."""
        params = self.space.sample(trial)
        directory = self.output_directory / f"trial_{trial.number:04d}"
        try:
            result = self.run_trial(
                self.space.resolve(params), directory, _PruningReporter(trial)
            )
            self.objective.validate(result)
        except optuna.TrialPruned:
            raise
        except self._anticipated_failures as error:
            # Re-raised unchanged so Optuna's catch= marks the trial FAIL and continues.
            self._record_failure(trial, error)
            raise
        except Exception as error:
            # Deliberately unwrapped: unanticipated errors stop the study and reach the
            # caller as raised; Optuna marks the trial FAIL on the way out.
            self._record_failure(trial, error)
            raise
        trial.set_user_attr("result", result.to_dict())
        return result.best_value

    @staticmethod
    def _record_failure(trial: optuna.Trial, error: Exception) -> None:
        """Keep a failed trial's reason for the report."""
        trial.set_user_attr("error", f"{type(error).__name__}: {error}")

    def _summarize(self, study: optuna.Study) -> StudyResult:
        """Rebuild the report and winner from storage, covering resumed trials."""
        rows, completed = [], []
        for trial in sorted(
            study.get_trials(deepcopy=False), key=lambda item: item.number
        ):
            result = self._completed_result(trial)
            rows.append(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "params": self.space.decode(trial.params),
                    "value": trial.value,
                    "intermediate_values": dict(trial.intermediate_values),
                    "result": result.to_dict() if result is not None else None,
                    "error": trial.user_attrs.get("error"),
                }
            )
            if result is not None:
                completed.append((trial, result))
        results = [result for _, result in completed]
        # Configured direction; the earliest trial wins an exact tie.
        selected = select_best_experiment(results) if results else None
        winner = next(
            (trial for trial, result in completed if result is selected), None
        )
        return StudyResult(
            output_directory=self.output_directory,
            study_name=self.settings.study_name,
            direction=self.objective.direction,
            trials=tuple(rows),
            results=tuple(results),
            selected=selected,
            selected_trial_number=winner.number if winner is not None else None,
            best_params=self.space.decode(winner.params)
            if winner is not None
            else None,
        )

    @staticmethod
    def _completed_result(trial: optuna.trial.FrozenTrial) -> ExperimentResult | None:
        """Return a completed trial's stored result, or None for any other state."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return None
        elif "result" not in trial.user_attrs:
            raise ValueError(f"Completed trial {trial.number} has no recorded result")
        else:
            return ExperimentResult(**trial.user_attrs["result"])

    def _write_report(self, report: StudyResult) -> None:
        """Persist the current report, replacing the previous one."""
        (self.output_directory / "study.yaml").write_text(
            yaml.safe_dump(report.to_dict(), sort_keys=False), encoding="utf-8"
        )
