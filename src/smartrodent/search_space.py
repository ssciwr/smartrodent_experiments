"""Search spaces declared by ``!suggest-*`` tags inside configuration documents.

A tag marks a value to be sampled by Optuna; its parameter is named by the value's
dotted path (``a.b[0]``). ``!coupled`` values follow a categorical's choice.
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

from smartrodent.training_validation import (
    FiniteNumber,
    Flag,
    Nonblank,
    parse_settings,
)


def _canonical(value: Any) -> str:
    """Return a value's canonical JSON text, the identity of choices and spaces."""
    return json.dumps(value, sort_keys=True, default=str)


def _distinct_choices(choices):
    """Reject repeated choices: they would be ambiguous in reports and coupled values."""
    if len({_canonical(choice) for choice in choices}) != len(choices):
        raise ValueError("choices must be unique")
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
    choices: Annotated[
        list[Any],
        Field(strict=True, min_length=1),
        AfterValidator(_distinct_choices),
    ]


@with_config(ConfigDict(extra="forbid"))
class _CoupledTag(TypedDict):
    target: Nonblank
    values: Annotated[list[Any], Field(strict=True)]


@dataclass(frozen=True)
class _Parameter:
    """One sampled setting: where it goes, how Optuna samples it, and its choices.

    Categorical values may be any YAML value. Optuna only stores primitives, so it
    samples the choice index and ``value`` maps the index back to the choice.
    """

    path: tuple
    distribution: BaseDistribution
    choices: tuple | None = None

    def value(self, sampled: Any) -> Any:
        """Return the setting for an Optuna-sampled value (a choice index for categoricals)."""
        if self.choices is not None:
            return deepcopy(self.choices[sampled])
        else:
            return sampled

    @property
    def reference(self) -> Any:
        """Return the Optuna value of the reference configuration: first choice or low end."""
        if self.choices is not None:
            return 0
        else:
            return self.distribution.low

    @property
    def other_edges(self) -> list:
        """Return the Optuna values prevalidated besides the reference: other choices or high."""
        if self.choices is not None:
            return list(range(1, len(self.choices)))
        elif self.distribution.high != self.distribution.low:
            return [self.distribution.high]
        else:
            return []

    def to_dict(self) -> dict:
        """Return the JSON-compatible identity, including the choice values themselves."""
        identity = {"distribution": distribution_to_json(self.distribution)}
        if self.choices is not None:
            identity["choices"] = list(self.choices)
        return identity


@dataclass(frozen=True)
class _Coupling:
    """A ``!coupled`` value list that follows a categorical parameter's choice."""

    path: tuple
    name: str
    target: str
    values: tuple


class SearchSpace:
    """Named Optuna parameters declared by ``!suggest-*`` tags inside a document.

    Parameters are named by their full dotted path, with list indices written as
    ``a.b[0]`` (the ``!coupled`` target syntax). Use ``from_document``.

    Args:
        document: Source document still containing its tags.
        parameters: Parameter name -> placement, distribution and choices.
        couplings: Coupled values that follow categorical parameters.
    """

    _distribution_types: ClassVar[dict] = {
        "suggest-float": (_FloatTag, FloatDistribution),
        "suggest-int": (_IntTag, IntDistribution),
        "suggest-categorical": (_CategoricalTag, CategoricalDistribution),
    }
    # A tuple, not a set: a user mapping may hold an unhashable "type" value.
    _tag_types: ClassVar[tuple] = ("coupled", *_distribution_types)

    def __init__(
        self,
        document: Mapping,
        parameters: Mapping[str, _Parameter],
        couplings: Sequence[_Coupling],
    ):
        """Store an already validated space; ``from_document`` performs validation."""
        self._document = deepcopy(dict(document))
        self._parameters = dict(parameters)
        self._couplings = tuple(couplings)

    @classmethod
    def from_document(
        cls, document: Mapping, *, protected: Sequence[str] = (), prefix: str = ""
    ) -> "SearchSpace":
        """Collect and validate every search parameter of a loaded document.

        Args:
            document: Mapping loaded with ``config_loading``.
            protected: Dotted paths, relative to ``document``, that must stay fixed;
                neither they nor their children may be sampled or coupled.
            prefix: Dotted path of ``document`` inside a larger configuration (a
                pipeline stage). Parameter names and coupling targets then use full
                paths.

        Returns:
            The validated search space.

        Raises:
            ValueError: A tag is invalid, replaces the whole document, contains another
                tag, a coupling does not fit its target, a protected path is searched,
                or no tag exists.
        """
        if not isinstance(document, Mapping):
            raise ValueError("A study configuration must be a mapping")
        prefix_path = _parse_path(prefix) if prefix else ()
        protected_paths = [_parse_path(text) for text in protected]
        parameters, couplings = {}, []
        for path, node in _tagged_nodes((), document):
            name = _format_path((*prefix_path, *path)) or "the document root"
            cls._reject_misplaced(name, path, node)
            cls._reject_protected(name, path, protected_paths, prefix_path)
            if node["type"] == "coupled":
                couplings.append(cls._coupling(name, path, node))
            else:
                parameters[name] = cls._parameter(name, path, node)
        if not parameters:
            raise ValueError(
                "Search space is empty: add !suggest-* tags or remove the tuning section"
            )
        cls._validate_couplings(couplings, parameters)
        return cls(document, parameters, couplings)

    @property
    def distributions(self) -> dict[str, BaseDistribution]:
        """Return parameter name -> Optuna distribution (choice indices for categoricals)."""
        return {name: item.distribution for name, item in self._parameters.items()}

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one value per parameter through Optuna's public suggest API.

        Args:
            trial: Running Optuna trial.

        Returns:
            Parameter name -> Optuna value (a choice index for categoricals).
        """
        return {
            name: _suggest(trial, name, item.distribution)
            for name, item in self._parameters.items()
        }

    def decode(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return the settings that Optuna values stand for, e.g. for reports.

        Args:
            params: Optuna values of some or all parameters.

        Returns:
            Parameter name -> setting (the chosen value for categoricals).

        Raises:
            ValueError: A parameter is unknown.
        """
        unknown = sorted(set(params) - set(self._parameters))
        if unknown:
            raise ValueError(f"Unknown search parameters: {unknown}")
        return {
            name: self._parameters[name].value(value) for name, value in params.items()
        }

    def resolve(self, params: Mapping[str, Any]) -> dict:
        """Return an independent concrete document for one complete set of values.

        Values are trusted: they come from ``sample`` or a stored trial, and the stage
        configuration classes validate the resolved document anyway.

        Args:
            params: Exactly one Optuna value per parameter.

        Returns:
            The document with every tag replaced by its setting.

        Raises:
            ValueError: A parameter is missing or unknown.
        """
        missing = sorted(set(self._parameters) - set(params))
        if missing:
            raise ValueError(f"Missing search parameters: {missing}")
        document = deepcopy(self._document)
        for name, value in self.decode(params).items():
            _assign(document, self._parameters[name].path, value)
        for coupling in self._couplings:
            _assign(
                document,
                coupling.path,
                deepcopy(coupling.values[params[coupling.target]]),
            )
        return document

    def _candidates(self) -> list[tuple[str, Any, dict]]:
        """Return one document per other bound and choice, varying one parameter at a time.

        Every other parameter stays at its reference value (the low end or the first
        choice). Invalid combinations of individually valid values are found only
        when sampled.

        Returns:
            (parameter name, candidate setting, resolved document) triples.
        """
        reference = self._reference_params()
        return [
            (name, item.value(value), self.resolve({**reference, name: value}))
            for name, item in self._parameters.items()
            for value in item.other_edges
        ]

    def validate_candidates[T](self, build: Callable[[dict], T]) -> list[T]:
        """Build the reference document and every candidate before any training.

        Args:
            build: Turns a resolved document into a validated stage configuration.

        Returns:
            The reference configuration first, then one candidate per other choice or
            bound, varying one parameter at a time.

        Raises:
            ValueError: The reference document is invalid, or a bound or choice cannot
                produce a valid configuration; the latter names the parameter and value.
        """
        # TypeError is anticipated: Ultralytics' get_cfg, called while validating a
        # classifier configuration, reports wrongly typed train_kwargs that way.
        reference = self._reference_params()
        try:
            configurations = [build(self.resolve(reference))]
        except (ValueError, TypeError) as error:
            values = ", ".join(
                f"{name}={value!r}" for name, value in self.decode(reference).items()
            )
            raise ValueError(
                "The study configuration is invalid with every search parameter at its "
                f"first choice or low end ({values}): {error}"
            ) from error
        for name, value, document in self._candidates():
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
            Serialized parameters, including categorical choice values, and couplings.
        """
        values = {
            "parameters": {
                name: item.to_dict() for name, item in self._parameters.items()
            },
            "couplings": [
                {"path": c.name, "target": c.target, "values": list(c.values)}
                for c in self._couplings
            ],
        }
        # Normalize through JSON exactly as Optuna storage would (tuples, keys, ...).
        return json.loads(json.dumps(values, default=str))

    @classmethod
    def _parameter(cls, name: str, path: tuple, node: Mapping) -> _Parameter:
        """Build one parameter from its tag settings.

        Optuna's constructors validate ranges, steps, log scales and keys. A range that
        is no multiple of step is only a warning (Optuna shortens it), so warnings are
        raised as errors.
        """
        schema, distribution_type = cls._distribution_types[node["type"]]
        settings = {key: value for key, value in node.items() if key != "type"}
        try:
            parsed = parse_settings(schema, settings)
            if distribution_type is CategoricalDistribution:
                choices = tuple(parsed["choices"])
                return _Parameter(
                    path, CategoricalDistribution(tuple(range(len(choices)))), choices
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    return _Parameter(path, distribution_type(**parsed))
        except (TypeError, ValueError, UserWarning) as error:
            raise ValueError(f"Invalid !{node['type']} at '{name}': {error}") from error

    @staticmethod
    def _coupling(name: str, path: tuple, node: Mapping) -> _Coupling:
        """Build one coupling from its tag settings; its target is a full dotted path."""
        settings = {key: value for key, value in node.items() if key != "type"}
        try:
            parsed = parse_settings(_CoupledTag, settings)
        except ValueError as error:
            raise ValueError(f"Invalid !coupled at '{name}': {error}") from error
        target = _format_path(_parse_path(parsed["target"]))
        return _Coupling(path, name, target, tuple(parsed["values"]))

    @staticmethod
    def _reject_misplaced(name: str, path: tuple, node: Mapping) -> None:
        """A tag stands for one setting; elsewhere it would bypass that setting's validation."""
        if not path:
            raise ValueError(
                f"!{node['type']} at '{name}' cannot replace the whole section; "
                "tag the individual settings instead"
            )
        nested = [
            inner
            for _, inner in _tagged_nodes((), [node.get("choices"), node.get("values")])
        ]
        if nested:
            raise ValueError(
                f"!{node['type']} at '{name}' cannot contain the search tag "
                f"!{nested[0]['type']}"
            )

    @staticmethod
    def _reject_protected(
        name: str,
        path: tuple,
        protected_paths: Sequence[tuple],
        prefix_path: tuple,
    ) -> None:
        """Keep settings that define comparability fixed for every trial."""
        for protected in protected_paths:
            if path[: len(protected)] == protected:
                raise ValueError(
                    f"'{name}' cannot be searched: '{_format_path((*prefix_path, *protected))}' "
                    "must stay fixed so that trials remain comparable"
                )

    @staticmethod
    def _validate_couplings(
        couplings: Sequence[_Coupling], parameters: Mapping[str, _Parameter]
    ) -> None:
        """Require each coupling to pair one-to-one with a categorical's choices."""
        for coupling in couplings:
            location = coupling.name
            target = parameters.get(coupling.target)
            if target is None:
                raise ValueError(
                    f"!coupled at '{location}' has target '{coupling.target}', "
                    "which is not a search parameter of this document"
                )
            elif target.choices is None:
                raise ValueError(
                    f"!coupled at '{location}' must target a !suggest-categorical, "
                    f"not '{coupling.target}'"
                )
            elif len(coupling.values) != len(target.choices):
                raise ValueError(
                    f"!coupled at '{location}' has {len(coupling.values)} values for "
                    f"{len(target.choices)} choices of '{coupling.target}'; their lengths must match"
                )
            else:
                pass  # valid coupling

    def _reference_params(self) -> dict[str, Any]:
        """Return every parameter's Optuna value at its first choice or low end."""
        return {name: item.reference for name, item in self._parameters.items()}


def reject_search_tags(document: Any, *, prefix: str = "") -> None:
    """Require a concrete single-run document: search tags need a study.

    Args:
        document: Loaded configuration of one run.
        prefix: Dotted path of ``document`` inside a larger configuration, for messages.

    Raises:
        ValueError: The document contains a search tag.
    """
    tagged = next(_tagged_nodes((), document), None)
    if tagged is not None:
        path, node = tagged
        prefix_path = _parse_path(prefix) if prefix else ()
        name = _format_path((*prefix_path, *path)) or "the document root"
        raise ValueError(
            f"!{node['type']} at '{name}' defines a search parameter; add a tuning "
            "section for this configuration (in a two-stage pipeline: tuning.<stage>) "
            "to run an Optuna study"
        )


def _tagged_nodes(path: tuple, node: Any) -> Iterator[tuple[tuple, dict]]:
    """Yield every study tag with its path, without descending into tags."""
    if isinstance(node, Mapping) and node.get("type") in SearchSpace._tag_types:
        yield path, node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _tagged_nodes((*path, key), value)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _tagged_nodes((*path, index), value)
    else:
        pass  # scalar leaves cannot hold tags


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


def _parse_path(text: str) -> tuple:
    """Parse a dotted path such as ``a.b[0]`` into ``("a", "b", 0)``."""
    components = []
    for token in text.split("."):
        key, *indices = token.split("[")
        components.append(key)
        components.extend(int(index.rstrip("]")) for index in indices)
    return tuple(components)


def _format_path(path: Sequence[Any]) -> str:
    """Format keys and list indices as a dotted path such as ``a.b[0]``; inverse of ``_parse_path``."""
    text = ""
    for component in path:
        if isinstance(component, int) and not isinstance(component, bool):
            text += f"[{component}]"
        elif text:
            text += f".{component}"
        else:
            text = str(component)
    return text


def _assign(document: dict, path: Sequence[Any], value: Any) -> None:
    """Assign a value at a key/index path within a nested mapping."""
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
