"""Run Optuna studies over configuration documents for the SupCon and classifier trainers.

A study document is an ordinary stage configuration in which some values are
``!suggest-float``, ``!suggest-int`` or ``!suggest-categorical`` tags, plus a
``tuning:`` section with the study settings. Each trial resolves the tags to one
concrete configuration, trains it with the existing trainer in its own
``trial_NNNN`` directory, and is ranked by the trainer's configured tracked metric.
"""

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass as result_dataclass
from dataclasses import field, replace
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, ClassVar

import mlflow
import optuna
import yaml
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from smartrodent import config_utils, utils
from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
    FreezeConfigurationError,
)
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.experiment_support import validate_stopping
from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration
from smartrodent.training_validation import (
    Count,
    FiniteNumber,
    Flag,
    MappingInput,
    Mode,
    Nonblank,
    PositiveNumber,
    parse_settings,
    settings_dict,
)

type StrictInt = Annotated[int, Field(strict=True)]
type Choice = None | bool | int | float | str
type EpochCallback = Callable[[int, float], None]
type TrialRunner = Callable[[dict, Path, EpochCallback], ExperimentResult]


class TrialConfigurationError(ValueError):
    """A sampled combination of otherwise valid values is not a valid configuration."""


# --------------------------------------------------------------------------------------
# Search parameters and the search space
# --------------------------------------------------------------------------------------


def _is_step_multiple(low: float, high: float, step: float) -> bool:
    """Check divisibility on decimal text, so 0.3 - 0.0 counts as 3 steps of 0.1."""
    return (Decimal(repr(high)) - Decimal(repr(low))) % Decimal(repr(step)) == 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class _FloatParameter:
    """Settings of a ``!suggest-float`` tag."""

    low: FiniteNumber
    high: FiniteNumber
    log: Flag = False
    step: PositiveNumber | None = None

    def __post_init__(self):
        """Reject ranges Optuna would adjust silently or reject only while sampling."""
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        if self.log and self.low <= 0:
            raise ValueError("log sampling requires low > 0")
        if self.log and self.step is not None:
            raise ValueError("log and step cannot be combined")
        if self.step is not None and not _is_step_multiple(
            self.low, self.high, self.step
        ):
            raise ValueError("high - low must be a multiple of step")

    def distribution(self) -> FloatDistribution:
        """Return the equivalent Optuna distribution."""
        return FloatDistribution(self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class _IntParameter:
    """Settings of a ``!suggest-int`` tag."""

    low: StrictInt
    high: StrictInt
    log: Flag = False
    step: Annotated[int, Field(strict=True, gt=0)] = 1

    def __post_init__(self):
        """Reject ranges Optuna would adjust silently or reject only while sampling."""
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        if self.log and self.low < 1:
            raise ValueError("log sampling requires low >= 1")
        if self.log and self.step != 1:
            raise ValueError("log and step cannot be combined")
        if (self.high - self.low) % self.step != 0:
            raise ValueError("high - low must be a multiple of step")

    def distribution(self) -> IntDistribution:
        """Return the equivalent Optuna distribution."""
        return IntDistribution(self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class _CategoricalParameter:
    """Settings of a ``!suggest-categorical`` tag."""

    # Optuna storage only round-trips primitive choices; structured values are
    # attached to a choice with !coupled-sweep instead.
    choices: Annotated[list[Choice], Field(min_length=1)]

    def __post_init__(self):
        """Reject duplicates, which would make coupled values ambiguous."""
        # Python equality is used, as by Optuna's storage: 1, 1.0 and True collide.
        if any(self.choices.count(choice) > 1 for choice in self.choices):
            raise ValueError("choices must be unique (1, 1.0 and True count as equal)")

    def distribution(self) -> CategoricalDistribution:
        """Return the equivalent Optuna distribution."""
        return CategoricalDistribution(self.choices)


@result_dataclass(frozen=True)
class _Coupling:
    """A ``!coupled-sweep`` value list that follows a categorical parameter's choice."""

    path: tuple
    name: str
    target: str
    values: tuple


class SearchSpace:
    """Named Optuna distributions declared by ``!suggest-*`` tags inside a document.

    Parameters are named by their full dotted path, with list indices written as
    ``a.b[0]`` (the ``!coupled-sweep`` target syntax). Use ``from_document``.

    Args:
        document: Source document still containing its tags.
        distributions: Parameter name -> Optuna distribution.
        locations: Parameter name -> key/index path within the document.
        couplings: Coupled values that follow categorical parameters.
    """

    _parameter_types: ClassVar[dict] = {
        "suggest-float": _FloatParameter,
        "suggest-int": _IntParameter,
        "suggest-categorical": _CategoricalParameter,
    }
    # A tuple, not a set: a user mapping may hold an unhashable "type" value.
    _tag_types: ClassVar[tuple] = ("sweep", "coupled-sweep", *config_utils.STUDY_TAGS)

    def __init__(
        self,
        document: Mapping,
        distributions: Mapping[str, BaseDistribution],
        locations: Mapping[str, tuple],
        couplings: Sequence[_Coupling],
    ):
        """Store an already validated space; ``from_document`` performs validation."""
        self._document = deepcopy(dict(document))
        self._distributions = dict(distributions)
        self._locations = dict(locations)
        self._couplings = tuple(couplings)

    @classmethod
    def from_document(
        cls, document: Mapping, *, protected: Sequence[str] = (), prefix: str = ""
    ) -> "SearchSpace":
        """Collect and validate every search parameter of a loaded document.

        Args:
            document: Mapping loaded with ``config_utils.get_loader()``.
            protected: Dotted paths, relative to ``document``, that must stay fixed;
                neither they nor their children may be sampled or coupled.
            prefix: Dotted path of ``document`` inside a larger configuration (a
                pipeline stage). Parameter names and coupling targets then use full
                paths, exactly as grid sweeps of that configuration do.

        Returns:
            The validated search space.

        Raises:
            ValueError: A tag is invalid, a grid sweep is present, a coupling does not
                fit its target, a protected path is searched, or no tag exists.
        """
        if not isinstance(document, Mapping):
            raise ValueError("A study configuration must be a mapping")
        prefix_path = utils.parse_path(prefix) if prefix else ()
        tagged = [
            (path, utils.format_path((*prefix_path, *path)), node)
            for path, node in cls._tagged_nodes((), document)
        ]
        # Mixing is reported first: a coupled-sweep may target a grid !sweep here.
        for _, name, node in tagged:
            if node["type"] == "sweep":
                raise ValueError(
                    f"!sweep at '{name}' cannot be combined with !suggest-* tags; "
                    "a configuration is either a grid sweep or an Optuna study"
                )
        protected_paths = [utils.parse_path(text) for text in protected]
        distributions, locations, couplings = {}, {}, []
        for path, name, node in tagged:
            cls._reject_protected(name, path, protected_paths, prefix_path)
            if node["type"] == "coupled-sweep":
                target = utils.format_path(tuple(node["target"]))
                couplings.append(_Coupling(path, name, target, tuple(node["values"])))
            else:
                distributions[name] = cls._distribution(name, node)
                locations[name] = path
        if not distributions:
            raise ValueError(
                "Search space is empty: add !suggest-* tags or remove the tuning section"
            )
        cls._validate_couplings(couplings, distributions)
        return cls(document, distributions, locations, couplings)

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        """Return parameter name -> Optuna distribution."""
        return dict(self._distributions)

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one value per parameter through Optuna's public suggest API.

        Args:
            trial: Running Optuna trial.

        Returns:
            Parameter name -> sampled value.
        """
        return {
            name: self._suggest(trial, name, distribution)
            for name, distribution in self._distributions.items()
        }

    def resolve(self, params: Mapping[str, Any]) -> dict:
        """Return an independent concrete document for one complete set of values.

        Args:
            params: Exactly one value per parameter, inside its distribution.

        Returns:
            The document with every tag replaced by its value.

        Raises:
            ValueError: A parameter is missing, unknown, or outside its distribution.
        """
        missing = sorted(set(self._distributions) - set(params))
        unknown = sorted(set(params) - set(self._distributions))
        if missing:
            raise ValueError(f"Missing search parameters: {missing}")
        if unknown:
            raise ValueError(f"Unknown search parameters: {unknown}")
        document = deepcopy(self._document)
        for name, value in params.items():
            if not self._contains(self._distributions[name], value):
                raise ValueError(
                    f"Value {value!r} is outside the search space of '{name}'"
                )
            utils.assign_at_path(document, self._locations[name], value)
        for coupling in self._couplings:
            choices = self._distributions[coupling.target].choices
            index = self._choice_index(choices, params[coupling.target])
            utils.assign_at_path(
                document, coupling.path, deepcopy(coupling.values[index])
            )
        return document

    def reference_document(self) -> dict:
        """Return the document with every parameter at its first choice or low end."""
        return self.resolve(self._reference_params())

    def candidates(self) -> list[tuple[str, Any, dict]]:
        """Return one document per other bound and choice, varying one parameter at a time.

        Every other parameter stays at its reference value (the low end or the first
        choice). Invalid combinations of individually valid values are found only
        when sampled.

        Returns:
            (parameter name, candidate value, resolved document) triples.
        """
        reference = self._reference_params()
        return [
            (name, value, self.resolve({**reference, name: value}))
            for name, distribution in self._distributions.items()
            for value in self._other_edges(distribution)
        ]

    def validate_candidates[T](self, build: Callable[[dict], T]) -> list[T]:
        """Build the reference document and every candidate before any training.

        Args:
            build: Turns a resolved document into a validated stage configuration.

        Returns:
            The reference configuration first, then the candidates in
            ``candidates()`` order.

        Raises:
            ValueError: The reference document is invalid, or a bound or choice cannot
                produce a valid configuration; the latter names the parameter and value.
        """
        # TypeError is anticipated, as in classifier sweeps: configuration
        # constructors report some invalid inputs that way.
        reference = self._reference_params()
        try:
            configurations = [build(self.resolve(reference))]
        except (ValueError, TypeError) as error:
            values = ", ".join(f"{name}={value!r}" for name, value in reference.items())
            raise ValueError(
                "The study configuration is invalid with every search parameter at its "
                f"first choice or low end ({values}): {error}"
            ) from error
        for name, value, document in self.candidates():
            try:
                configurations.append(build(document))
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Search parameter '{name}' value {value!r} is invalid: {error}"
                ) from error
        return configurations

    def to_dict(self) -> dict:
        """Return a JSON-compatible identity used to recognize the same space on resume.

        Returns:
            Serialized distributions and couplings.
        """
        values = {
            "parameters": {
                name: distribution_to_json(distribution)
                for name, distribution in self._distributions.items()
            },
            "couplings": [
                {"path": c.name, "target": c.target, "values": list(c.values)}
                for c in self._couplings
            ],
        }
        # Normalize through JSON exactly as Optuna storage would (tuples, keys, ...).
        return json.loads(json.dumps(values, default=str))

    @classmethod
    def _tagged_nodes(cls, path: tuple, node: Any) -> Iterator[tuple[tuple, dict]]:
        """Yield every sweep/study tag with its path, without descending into tags."""
        if isinstance(node, Mapping) and node.get("type") in cls._tag_types:
            yield path, node
        elif isinstance(node, Mapping):
            for key, value in node.items():
                yield from cls._tagged_nodes((*path, key), value)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from cls._tagged_nodes((*path, index), value)
        else:
            pass  # scalar leaves cannot hold tags

    @classmethod
    def _distribution(cls, name: str, node: Mapping) -> BaseDistribution:
        """Validate one tag's settings and convert them to a distribution."""
        settings = {key: value for key, value in node.items() if key != "type"}
        try:
            return parse_settings(
                cls._parameter_types[node["type"]], settings
            ).distribution()
        except ValueError as error:
            raise ValueError(f"Invalid !{node['type']} at '{name}': {error}") from error

    @classmethod
    def _reject_protected(
        cls,
        name: str,
        path: tuple,
        protected_paths: Sequence[tuple],
        prefix_path: tuple,
    ) -> None:
        """Keep settings that define comparability fixed for every trial."""
        for protected in protected_paths:
            if path[: len(protected)] == protected:
                raise ValueError(
                    f"'{name}' cannot be searched: '{utils.format_path((*prefix_path, *protected))}' "
                    "must stay fixed so that trials remain comparable"
                )

    @classmethod
    def _validate_couplings(
        cls, couplings: Sequence[_Coupling], distributions: Mapping
    ) -> None:
        """Require each coupling to pair one-to-one with a categorical's choices."""
        for coupling in couplings:
            location = coupling.name
            target = distributions.get(coupling.target)
            if target is None:
                raise ValueError(
                    f"!coupled-sweep at '{location}' has target '{coupling.target}', "
                    "which is not a search parameter of this document"
                )
            elif not isinstance(target, CategoricalDistribution):
                raise ValueError(
                    f"!coupled-sweep at '{location}' must target a !suggest-categorical, "
                    f"not '{coupling.target}'"
                )
            elif len(coupling.values) != len(target.choices):
                raise ValueError(
                    f"!coupled-sweep at '{location}' has {len(coupling.values)} values for "
                    f"{len(target.choices)} choices of '{coupling.target}'; their lengths must match"
                )
            else:
                pass  # valid coupling

    @staticmethod
    def _suggest(trial: optuna.Trial, name: str, distribution: BaseDistribution) -> Any:
        """Map one distribution to Optuna's matching public suggest method."""
        if isinstance(distribution, FloatDistribution):
            return trial.suggest_float(
                name,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, IntDistribution):
            return trial.suggest_int(
                name,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, CategoricalDistribution):
            return trial.suggest_categorical(name, distribution.choices)
        else:
            raise TypeError(f"Unsupported distribution {type(distribution).__name__}")

    @staticmethod
    def _choice_index(choices: Sequence, value: Any) -> int | None:
        """Find the choice equal to a value and of the same type (1, 1.0 and True differ)."""
        return next(
            (
                index
                for index, choice in enumerate(choices)
                if type(choice) is type(value) and choice == value
            ),
            None,
        )

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Accept ints and floats, but not booleans."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _contains(cls, distribution: BaseDistribution, value: Any) -> bool:
        """Check exact membership without Optuna's private helpers or any coercion."""
        if isinstance(distribution, CategoricalDistribution):
            return cls._choice_index(distribution.choices, value) is not None
        elif isinstance(distribution, IntDistribution):
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and distribution.low <= value <= distribution.high
                and (value - distribution.low) % distribution.step == 0
            )
        elif isinstance(distribution, FloatDistribution):
            return (
                cls._is_number(value)
                and distribution.low <= value <= distribution.high
                and (
                    distribution.step is None
                    or cls._on_step_grid(distribution.low, value, distribution.step)
                )
            )
        else:
            raise TypeError(f"Unsupported distribution {type(distribution).__name__}")

    @staticmethod
    def _on_step_grid(low: float, value: float, step: float) -> bool:
        """Allow floating-point error: Optuna computes grid values as low + k * step."""
        steps = (value - low) / step
        return abs(steps - round(steps)) <= 1e-8

    def _reference_params(self) -> dict[str, Any]:
        """Return every parameter at its first choice or low end."""
        return {
            name: self._first(distribution)
            for name, distribution in self._distributions.items()
        }

    @staticmethod
    def _first(distribution: BaseDistribution) -> Any:
        """Return the reference value: the first choice or the low end."""
        if isinstance(distribution, CategoricalDistribution):
            return distribution.choices[0]
        else:
            return distribution.low

    @staticmethod
    def _other_edges(distribution: BaseDistribution) -> list:
        """Return the prevalidated values besides the reference: other choices or high."""
        if isinstance(distribution, CategoricalDistribution):
            return list(distribution.choices[1:])
        elif distribution.high != distribution.low:
            return [distribution.high]
        else:
            return []


# --------------------------------------------------------------------------------------
# Study settings and objective
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class StudyComponent:
    """An Optuna sampler or pruner chosen by class name.

    Args:
        name: Class name in ``optuna.samplers`` or ``optuna.pruners``.
        parameters: Keyword arguments for its constructor; empty means its defaults.
    """

    name: Nonblank
    parameters: dict[str, Any] = field(default_factory=dict)

    def build(self, module: ModuleType, base: type):
        """Construct the named class from the given Optuna module.

        Args:
            module: ``optuna.samplers`` or ``optuna.pruners``.
            base: Required base class.

        Returns:
            A new sampler or pruner instance.

        Raises:
            ValueError: The name is not a concrete subclass of base, or the
                parameters do not fit its constructor.
        """
        factory = getattr(module, self.name, None)
        if not (
            isinstance(factory, type) and issubclass(factory, base)
        ) or inspect.isabstract(factory):
            raise ValueError(
                f"'{self.name}' is not a concrete class in {module.__name__}"
            )
        try:
            return factory(**self.parameters)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid parameters for {self.name}: {error}") from error


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StudyConfiguration:
    """Settings of a ``tuning:`` section. The direction follows the tracked mode.

    Args:
        study_name: Optuna study name, also the resume key in persistent storage.
        storage: Optuna storage URL, e.g. ``sqlite:///optuna.db``; null is in-memory.
        n_trials: Trials to run in this invocation (Optuna semantics).
        sampler: Sampler class name and parameters.
        pruner: Pruner class name and parameters, or null for no pruning.
        load_if_exists: Resume an existing study of the same name.
        timeout: Optional wall-clock budget in seconds for this invocation.
    """

    study_name: Nonblank
    storage: Nonblank | None
    n_trials: Count
    sampler: MappingInput[StudyComponent]
    pruner: MappingInput[StudyComponent] | None
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
        """Studies rank trials, which, as for sweep selection, needs validation data."""
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
        trials: Report rows for every trial, in trial-number order.
        results: Completed trial results, in trial-number order.
        selected: Best completed result (earliest on exact ties), or None.
        selected_trial_number: Trial number of ``selected``, or None.
        best_params: Sampled parameters of ``selected``, or None.
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
    the study, which a hand-written ask-and-tell loop would have to replicate.

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
                    "params": dict(trial.params),
                    "value": trial.value,
                    "intermediate_values": dict(trial.intermediate_values),
                    "result": result.to_dict() if result is not None else None,
                    "error": trial.user_attrs.get("error"),
                }
            )
            if result is not None:
                completed.append((trial, result))
        results = [result for _, result in completed]
        # The established sweep rule: configured direction, earliest exact tie.
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
            best_params=dict(winner.params) if winner is not None else None,
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


# --------------------------------------------------------------------------------------
# Studies of the two trainers
# --------------------------------------------------------------------------------------


def load_document(path: str | Path) -> Any:
    """Load a YAML configuration with the project's custom tags.

    Args:
        path: Configuration file.

    Returns:
        The loaded document.
    """
    return yaml.load(
        Path(path).read_text(encoding="utf-8"), Loader=config_utils.get_loader()
    )


def is_study_configuration(path: str | Path) -> bool:
    """Return whether a configuration file defines an Optuna study (a tuning section).

    Args:
        path: Configuration file.

    Returns:
        True when the document is a mapping with a ``tuning`` key.
    """
    document = load_document(path)
    return isinstance(document, Mapping) and "tuning" in document


class _StageStudy(ABC):
    """Shared flow of a study over one trainer's standalone configuration.

    The document is prevalidated on construction; ``run`` trains the trials.

    Args:
        document: Stage configuration with ``!suggest-*`` tags and a tuning section.
        prefix: Dotted path of the stage inside a pipeline document, for naming.
    """

    # Settings that define what trials are compared on, or where they read and write.
    protected: ClassVar[tuple[str, ...]] = (
        "tracked_metric",
        "tracked_mode",
        "validation_metrics",
        "train_dataset",
        "validation_dataset",
        "class_names",
        "output_directory",
        "mlflow",
    )

    # Errors raised while preparing training (not by the configuration classes) that
    # mean an invalid combination of sampled values rather than a broken study.
    _combination_errors: ClassVar[tuple[type[Exception], ...]] = ()

    def __init__(self, document: Mapping, *, prefix: str = ""):
        """Parse the study settings and build every candidate configuration."""
        if not isinstance(document, Mapping):
            raise ValueError("A study configuration must be a mapping")
        values = deepcopy(dict(document))
        if "tuning" not in values:
            raise ValueError("An Optuna study requires a tuning section")
        self.settings = StudyConfiguration.from_mapping(values.pop("tuning"))
        # The space first: it rejects a tag on output_directory with a clear message.
        self.space = SearchSpace.from_document(
            values, protected=self.protected, prefix=prefix
        )
        if values.get("output_directory") is None:
            raise ValueError("An Optuna study requires an output_directory")
        self.output_directory = Path(values["output_directory"])
        reference_directory = self.output_directory / "trial_0000"
        self.configurations = tuple(
            self.space.validate_candidates(
                partial(self._configuration, directory=reference_directory)
            )
        )
        self.objective = self._objective(self.configurations[0])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "_StageStudy":
        """Load and prevalidate a study configuration file.

        Args:
            path: YAML file; paths inside are relative to the working directory.

        Returns:
            The prepared study.
        """
        return cls(load_document(path))

    @property
    def reference(self):
        """Return the reference configuration (all parameters at their first value)."""
        return self.configurations[0]

    def run(self, *, parent_run_id: str | None = None) -> StudyResult:
        """Run the study, tracking trials under one parent run when MLflow is configured.

        Args:
            parent_run_id: Active pipeline run to parent trials to; when None and
                tracking is enabled, the study opens its own parent run.

        Returns:
            The study report and winner.
        """
        tracking = self.reference.mlflow
        if tracking is None or parent_run_id is not None:
            return self._finish(self._study(parent_run_id, self._identity()).run())
        with tracking.start_run(self.output_directory.name) as run:
            mlflow.set_tags(tracking.run_tags())
            try:
                result = self._study(run.info.run_id, self._identity()).run()
            finally:
                report = self.output_directory / "study.yaml"
                if report.is_file():
                    mlflow.log_artifact(str(report))
            mlflow.log_metric(
                f"best_validation_{self.objective.tracked_metric}",
                result.selected.best_value,
            )
            return self._finish(result)

    def preflight(self) -> None:
        """Check output and storage conflicts without training; see HyperparameterStudy.preflight."""
        self._study(None, {}).preflight()

    def _study(
        self, parent_run_id: str | None, identity: Mapping
    ) -> HyperparameterStudy:
        """Bind trial execution to the run that parents this invocation's trials."""
        return HyperparameterStudy(
            self.space,
            self.settings,
            partial(self._run_trial, parent_run_id=parent_run_id),
            output_directory=self.output_directory,
            objective=self.objective,
            identity=identity,
        )

    def _identity(self) -> dict[str, Any]:
        """Return stage inputs a resumed study must share besides space and objective."""
        return {}

    def _run_trial(
        self,
        document: dict,
        directory: Path,
        epoch_callback: EpochCallback,
        *,
        parent_run_id: str | None,
    ) -> ExperimentResult:
        """Validate one sampled document and train it."""
        # TypeError is anticipated, as in classifier sweeps: configuration
        # constructors report some invalid inputs that way.
        try:
            configuration = self._configuration(document, directory=directory)
        except (ValueError, TypeError) as error:
            raise TrialConfigurationError(
                f"Sampled configuration is invalid: {error}"
            ) from error
        try:
            return self._train(
                configuration,
                parent_run_id=parent_run_id,
                epoch_callback=epoch_callback,
            )
        except self._combination_errors as error:
            raise TrialConfigurationError(
                f"Sampled combination does not fit: {error}"
            ) from error

    def _finish(self, result: StudyResult) -> StudyResult:
        """Complete stage-specific work after all trials; nothing by default."""
        return result

    @abstractmethod
    def _configuration(self, document: dict, *, directory: Path):
        """Build one validated stage configuration writing to the given directory."""

    @abstractmethod
    def _objective(self, configuration) -> StudyObjective:
        """Return the ranking defined by a stage configuration."""

    @abstractmethod
    def _train(
        self, configuration, *, parent_run_id: str | None, epoch_callback: EpochCallback
    ) -> ExperimentResult:
        """Train one configuration with the stage's trainer."""


class SupConStudy(_StageStudy):
    """Optuna study over a standalone SupCon configuration.

    Args:
        document: SupCon configuration with ``!suggest-*`` tags and a tuning section.
    """

    # Loss values at different temperatures are not comparable (the sweep rule).
    protected: ClassVar[tuple[str, ...]] = (*_StageStudy.protected, "temperature")

    def _configuration(
        self, document: dict, *, directory: Path
    ) -> TrainingConfiguration:
        """Build a SupCon configuration whose stopping settings agree with the ranking."""
        configuration = TrainingConfiguration.from_mapping(
            {**document, "output_directory": directory}
        )
        validate_stopping(configuration, configuration.effective_tracked_mode)
        return configuration

    def _objective(self, configuration: TrainingConfiguration) -> StudyObjective:
        """Rank by the effective tracked metric, as SupCon sweeps do."""
        return StudyObjective(
            tracked_metric=configuration.effective_tracked_metric,
            tracked_mode=configuration.effective_tracked_mode,
            validation_dataset=configuration.validation_dataset,
        )

    def _train(
        self,
        configuration: TrainingConfiguration,
        *,
        parent_run_id: str | None,
        epoch_callback: EpochCallback,
    ) -> ExperimentResult:
        """Train one SupCon trial."""
        return SupConExperiment(configuration).run_with_result(
            parent_run_id=parent_run_id, epoch_callback=epoch_callback
        )


class ClassifierStudy(_StageStudy):
    """Optuna study over a standalone classifier configuration.

    As in classifier sweeps, every completed trial registers its model when a
    registry is configured; only the study winner is compared for promotion.

    Args:
        document: Classifier configuration with ``!suggest-*`` tags and a tuning section.
        encoder_provenance: Optional selected-SupCon metadata from the pipeline.
        prefix: Dotted path of the stage inside a pipeline document, for naming.
    """

    # Freeze validity depends on the encoder network, which a trial only loads when
    # training starts; e.g. a high freeze combined with a full head replacement.
    _combination_errors: ClassVar[tuple[type[Exception], ...]] = (
        FreezeConfigurationError,
    )

    def __init__(
        self,
        document: Mapping,
        *,
        encoder_provenance: Mapping | None = None,
        prefix: str = "",
    ):
        """Prevalidate the study and keep resources shared by all trials."""
        super().__init__(document, prefix=prefix)
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        self._head_cache = {}
        tracking = self.reference.mlflow
        self._registered = (
            tracking is not None and tracking.registered_model_name is not None
        )

    def _identity(self) -> dict[str, Any]:
        """Trials trained on different encoder bytes are not comparable."""
        checkpoint = Path(self.reference.initialization["encoder"]["checkpoint"])
        with checkpoint.open("rb") as stream:
            return {
                "encoder checkpoint": hashlib.file_digest(stream, "sha256").hexdigest()
            }

    def _configuration(
        self, document: dict, *, directory: Path
    ) -> ClassifierConfiguration:
        """Build one classifier configuration."""
        return ClassifierConfiguration.from_mapping(
            {**document, "output_directory": directory}
        )

    def _objective(self, configuration: ClassifierConfiguration) -> StudyObjective:
        """Rank by the configured classifier metric."""
        return StudyObjective(
            tracked_metric=configuration.tracked_metric,
            tracked_mode=configuration.effective_tracked_mode,
            validation_dataset=configuration.validation_dataset,
        )

    def _train(
        self,
        configuration: ClassifierConfiguration,
        *,
        parent_run_id: str | None,
        epoch_callback: EpochCallback,
    ) -> ExperimentResult:
        """Train one classifier trial, deferring promotion to the end of the study."""
        experiment = ClassifierExperiment(
            configuration,
            encoder_provenance=self.encoder_provenance,
            head_cache=self._head_cache,
        )
        return experiment.run_with_result(
            parent_run_id=parent_run_id,
            defer_promotion=self._registered,
            epoch_callback=epoch_callback,
        )

    def _finish(self, result: StudyResult) -> StudyResult:
        """Compare only the study winner against the registered champion."""
        if not self._registered:
            return result
        else:
            reference = self.reference
            registry = ClassifierChampionRegistry(
                reference.mlflow,
                reference.tracked_metric_definition,
                reference.class_names,
            )
            return replace(
                result, registered_champion_version=registry.promote(result.selected)
            )
