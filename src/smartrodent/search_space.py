"""Search spaces declared by ``!suggest-*`` tags inside configuration documents.

A tag marks a value to be sampled by Optuna; its parameter is named by the value's
dotted path (``a.b[0]``). ``!coupled-sweep`` values follow a categorical's choice.
"""

import json
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, NotRequired, TypedDict

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)
from pydantic import AfterValidator, ConfigDict, Field, with_config

from smartrodent import config_utils, utils
from smartrodent.training_validation import FiniteNumber, Flag, parse_settings


def _distinct_choices(choices):
    """Reject duplicates by Python equality, as Optuna storage does (1 == 1.0 == True)."""
    if any(choices.count(choice) > 1 for choice in choices):
        raise ValueError("choices must be unique (1, 1.0 and True count as equal)")
    return choices


type _Integer = Annotated[int, Field(strict=True)]


# Strict schemas reject what Optuna would silently coerce (0.5 -> 0, booleans, NaN
# or infinite bounds); Optuna itself still validates ranges, steps and log scales.
@with_config(ConfigDict(extra="forbid"))
class _FloatTag(TypedDict):
    low: FiniteNumber
    high: FiniteNumber
    step: NotRequired[FiniteNumber | None]
    log: NotRequired[Flag]


@with_config(ConfigDict(extra="forbid"))
class _IntTag(TypedDict):
    low: _Integer
    high: _Integer
    step: NotRequired[_Integer]
    log: NotRequired[Flag]


@with_config(ConfigDict(extra="forbid"))
class _CategoricalTag(TypedDict):
    # strict: a tuple or YAML set would be converted, and a set's order varies by run
    choices: Annotated[list[Any], Field(strict=True), AfterValidator(_distinct_choices)]


@dataclass(frozen=True)
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

    _distribution_types: ClassVar[dict] = {
        "suggest-float": (_FloatTag, FloatDistribution),
        "suggest-int": (_IntTag, IntDistribution),
        "suggest-categorical": (_CategoricalTag, CategoricalDistribution),
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

        Values are trusted: they come from ``sample`` or a stored trial, and the stage
        configuration classes validate the resolved document anyway.

        Args:
            params: Exactly one value per parameter.

        Returns:
            The document with every tag replaced by its value.

        Raises:
            ValueError: A parameter is missing or unknown, or a coupled categorical
                value is not one of its choices.
        """
        missing = sorted(set(self._distributions) - set(params))
        unknown = sorted(set(params) - set(self._distributions))
        if missing:
            raise ValueError(f"Missing search parameters: {missing}")
        if unknown:
            raise ValueError(f"Unknown search parameters: {unknown}")
        document = deepcopy(self._document)
        for name, value in params.items():
            utils.assign_at_path(document, self._locations[name], value)
        for coupling in self._couplings:
            # Choices are unique (checked at construction), so the index is unambiguous.
            index = self._distributions[coupling.target].choices.index(
                params[coupling.target]
            )
            utils.assign_at_path(
                document, coupling.path, deepcopy(coupling.values[index])
            )
        return document

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
        """Build Optuna's distribution from one tag's settings.

        Optuna's constructors validate ranges, steps, log scales and keys. Two of its
        checks are only warnings (a range that is no multiple of step is shortened;
        structured choices break storage later), so warnings are raised as errors.
        """
        schema, distribution_type = cls._distribution_types[node["type"]]
        settings = {key: value for key, value in node.items() if key != "type"}
        try:
            parsed = parse_settings(schema, settings)
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                return distribution_type(**parsed)
        except (TypeError, ValueError, UserWarning) as error:
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
