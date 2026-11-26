"""YAML search-space tags resolved into named Optuna distributions."""

from copy import deepcopy
from decimal import Decimal

import optuna
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from study_test_support import load_tagged_yaml

from smartrodent.search_space import SearchSpace


def _space(text: str, **kwargs):
    """Build a search space from YAML text."""
    return SearchSpace.from_document(load_tagged_yaml(text), **kwargs)


def test_suggest_tags_define_optuna_distributions_named_by_dotted_path():
    """Each tag becomes one Optuna distribution keyed by its full document path."""
    space = _space(
        """
optimizer:
  lr: !suggest-float {low: 1.0e-5, high: 1.0e-3, log: true}
  weight_decay: !suggest-float {low: 0.0, high: 0.1, step: 0.01}
classes_per_batch: !suggest-int {low: 4, high: 12, step: 2}
epochs: !suggest-int {low: 1, high: 64, log: true}
head: !suggest-categorical {choices: [linear, full, null]}
fixed: 3
""",
    )
    assert space.distributions == {
        "optimizer.lr": FloatDistribution(1.0e-5, 1.0e-3, log=True),
        "optimizer.weight_decay": FloatDistribution(0.0, 0.1, step=0.01),
        "classes_per_batch": IntDistribution(4, 12, step=2),
        "epochs": IntDistribution(1, 64, log=True),
        "head": CategoricalDistribution(["linear", "full", None]),
    }


def test_block_style_tags_are_equivalent_to_flow_style():
    """Tags accept ordinary YAML block mappings, as the existing sweep tags do."""
    flow = _space("lr: !suggest-float {low: 0.1, high: 0.9}\n")
    block = _space("lr: !suggest-float\n  low: 0.1\n  high: 0.9\n")
    assert flow.distributions == block.distributions


def test_tags_inside_lists_are_named_with_index_paths_and_resolved_in_place():
    """List elements use the coupled-sweep path syntax and keep their siblings."""
    space = _space(
        """
transforms:
  random_resized_crop:
    scale: [!suggest-float {low: 0.5, high: 0.9}, 1.0]
""",
    )
    assert set(space.distributions) == {"transforms.random_resized_crop.scale[0]"}
    resolved = space.resolve({"transforms.random_resized_crop.scale[0]": 0.6})
    assert resolved["transforms"]["random_resized_crop"]["scale"] == [0.6, 1.0]


def test_coupled_sweep_follows_the_chosen_categorical_index():
    """Existing !coupled-sweep semantics pair structured values with a categorical choice."""
    space = _space(
        """
initialization:
  head:
    replace: !suggest-categorical {choices: [linear, full]}
freeze: !coupled-sweep
  target: initialization.head.replace
  values: [11, 10]
augment: !coupled-sweep
  target: initialization.head.replace
  values: [{p: 0.1}, {p: 0.5}]
""",
    )
    # Coupled values are not independent Optuna parameters.
    assert set(space.distributions) == {"initialization.head.replace"}
    linear = space.resolve({"initialization.head.replace": "linear"})
    full = space.resolve({"initialization.head.replace": "full"})
    assert (linear["freeze"], linear["augment"]) == (11, {"p": 0.1})
    assert (full["freeze"], full["augment"]) == (10, {"p": 0.5})
    assert full["initialization"]["head"]["replace"] == "full"


def test_resolution_leaves_the_source_document_and_previous_results_untouched():
    """Every resolution is an independent concrete copy."""
    document = load_tagged_yaml(
        """
lr: !suggest-float {low: 0.1, high: 0.9}
nested: {values: [1, 2]}
"""
    )
    snapshot = deepcopy(document)
    space = SearchSpace.from_document(document)
    first = space.resolve({"lr": 0.2})
    first["nested"]["values"].append(3)
    second = space.resolve({"lr": 0.3})
    assert second == {"lr": 0.3, "nested": {"values": [1, 2]}}
    assert document == snapshot


@pytest.mark.parametrize(
    "params,match",
    [
        ({"head": "linear"}, "lr"),  # missing parameter
        ({"lr": 0.5, "head": "linear", "other": 1}, "other"),  # unknown parameter
    ],
)
def test_resolution_rejects_parameters_that_do_not_match_the_space(params, match):
    """Resolution never fills in, drops, or clamps a parameter silently."""
    space = _space(
        """
lr: !suggest-float {low: 0.0, high: 1.0}
head: !suggest-categorical {choices: [linear, full]}
""",
    )
    with pytest.raises(ValueError, match=match):
        space.resolve(params)


# One schema case per tag type, plus our own rules: unique choices, and Optuna's
# step and structured-choice warnings escalated to errors. Range, step and log
# consistency is Optuna's to check.
@pytest.mark.parametrize(
    "text",
    [
        "lr: !suggest-float {low: 0.0, high: 1.0, step: 0.3}\n",  # range not a step multiple
        "lr: !suggest-float {low: .nan, high: 1.0}\n",  # nonfinite bound
        "lr: !suggest-float 0.1\n",  # scalar instead of mapping
        "n: !suggest-int {low: 0.5, high: 3}\n",  # float bound for an integer
        "c: !suggest-categorical {choices: [a, a]}\n",  # ambiguous duplicate choice
        "c: !suggest-categorical {choices: [[1, 2], [3]]}\n",  # non-primitive choices
        "c: !suggest-categorical {values: [a, b]}\n",  # wrong key
    ],
)
def test_invalid_tag_definitions_are_rejected_with_value_errors(text):
    """Optuna would warn, adjust, or crash later; the project rejects these up front."""
    with pytest.raises(ValueError):
        SearchSpace.from_document(load_tagged_yaml(text))


def test_a_document_without_suggest_tags_is_not_a_search_space():
    """An empty search space would run identical trials; reject it explicitly."""
    with pytest.raises(ValueError, match="empty"):
        _space("lr: 0.1\nnested: {a: 1}\n")


@pytest.mark.parametrize(
    "text",
    [
        "lr: !suggest-float {low: 0.1, high: 0.9}\nseed: !sweep\n  values: [1, 2]\n",
        "lr: !suggest-float {low: 0.1, high: 0.9}\nnested:\n  seed: !sweep\n    values: [1, 2]\n",
        # Mixing is reported before coupling problems: this coupled-sweep targets a !sweep.
        (
            "lr: !suggest-float {low: 0.1, high: 0.9}\n"
            "model: !sweep\n  values: [a, b]\n"
            "batch: !coupled-sweep\n  target: model\n  values: [1, 2]\n"
        ),
    ],
)
def test_grid_sweeps_cannot_be_mixed_into_a_search_space(text):
    """Decision: a configuration is either a grid sweep or an Optuna study, never both."""
    with pytest.raises(ValueError, match="!sweep"):
        SearchSpace.from_document(load_tagged_yaml(text))


@pytest.mark.parametrize(
    "text,match",
    [
        (
            (
                "lr: !suggest-float {low: 0.1, high: 0.9}\n"
                "batch: !coupled-sweep\n  target: lr\n  values: [1, 2]\n"
            ),
            "categorical",
        ),
        (
            (
                "head: !suggest-categorical {choices: [a, b]}\n"
                "batch: !coupled-sweep\n  target: absent.path\n  values: [1, 2]\n"
            ),
            r"target.*absent\.path|absent\.path.*target",
        ),
        (
            (
                "head: !suggest-categorical {choices: [a, b]}\n"
                "batch: !coupled-sweep\n  target: head\n  values: [1, 2, 3]\n"
            ),
            "length",
        ),
    ],
)
def test_invalid_couplings_are_rejected(text, match):
    """A coupled value must pair one-to-one with an existing categorical's choices."""
    with pytest.raises(ValueError, match=match):
        SearchSpace.from_document(load_tagged_yaml(text))


@pytest.mark.parametrize(
    "text,path",
    [
        ("tracked_metric: !suggest-categorical {choices: [a, b]}\n", "tracked_metric"),
        (
            "validation_metrics:\n  f1:\n    average: !suggest-categorical {choices: [macro, micro]}\n",
            "validation_metrics.f1.average",
        ),
        (
            (
                "head: !suggest-categorical {choices: [a, b]}\n"
                "tracked_metric: !coupled-sweep\n  target: head\n  values: [x, y]\n"
            ),
            "tracked_metric",
        ),
    ],
)
def test_protected_paths_and_their_children_cannot_be_searched(text, path):
    """Settings that define comparability must stay fixed, whether sampled or coupled."""
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        SearchSpace.from_document(
            load_tagged_yaml(text), protected=("tracked_metric", "validation_metrics")
        )


def _flow(mapping) -> str:
    """Render a mapping as round-trippable YAML flow text."""
    return yaml.safe_dump(mapping, default_flow_style=True).strip()


@st.composite
def _parameter(draw):
    """Draw one tag definition and its exact expected bounds/choices."""
    kind = draw(
        st.sampled_from(["float", "float_step", "float_log", "int", "categorical"])
    )
    if kind == "float":
        low = draw(st.floats(-1e3, 1e3))
        high = low + draw(st.floats(1e-3, 1e3))
        return kind, {"low": low, "high": high}
    elif kind == "float_step":
        step = draw(st.sampled_from(["0.5", "0.25", "0.1", "0.01"]))
        low = Decimal(draw(st.integers(-50, 50))) * Decimal(step)
        high = low + Decimal(draw(st.integers(1, 40))) * Decimal(step)
        return kind, {"low": float(low), "high": float(high), "step": float(step)}
    elif kind == "float_log":
        low = draw(st.floats(1e-6, 1e2))
        return kind, {"low": low, "high": low * draw(st.floats(1.5, 1e4)), "log": True}
    elif kind == "int":
        low = draw(st.integers(-100, 100))
        step = draw(st.integers(1, 5))
        return kind, {
            "low": low,
            "high": low + step * draw(st.integers(1, 20)),
            "step": step,
        }
    elif kind == "categorical":
        choices = draw(
            st.lists(
                st.one_of(
                    st.text("abcxyz", min_size=1, max_size=4),
                    st.integers(-9, 9),
                    st.none(),
                ),
                min_size=1,
                max_size=5,
                unique=True,
            )
        )
        return kind, {"choices": choices}
    else:
        raise AssertionError(kind)


@st.composite
def _search_documents(draw):
    """Draw YAML text with top-level and nested tags, couplings, and fixed values."""
    # The prefix keeps names clear of structural keys ("fixed", "group") and of YAML 1.1
    # keywords such as "no" or "on", which would load as booleans instead of strings.
    name = st.from_regex(r"[a-z]{1,5}", fullmatch=True).map(lambda text: f"p_{text}")
    names = draw(st.lists(name, min_size=1, max_size=5, unique=True))
    tags = {
        "float": "!suggest-float",
        "float_step": "!suggest-float",
        "float_log": "!suggest-float",
        "int": "!suggest-int",
        "categorical": "!suggest-categorical",
    }
    root, nested, specs, couplings = ["fixed: {keep: 7}"], [], {}, {}
    for name in names:
        kind, spec = draw(_parameter())
        in_group = draw(st.booleans())
        path = f"group.{name}" if in_group else name
        (nested if in_group else root).append(f"{name}: {tags[kind]} {_flow(spec)}")
        specs[path] = (kind, spec)
        if kind == "categorical" and draw(st.booleans()):
            values = list(range(100, 100 + len(spec["choices"])))
            root.append(
                f"{name}_coupled: !coupled-sweep {{target: {path}, values: {values}}}"
            )
            couplings[f"{name}_coupled"] = (path, values)
    lines = root + (["group:"] + [f"  {line}" for line in nested] if nested else [])
    return "\n".join(lines) + "\n", specs, couplings


def _value_at(document, path):
    """Read a dotted path from a resolved document."""
    for part in path.split("."):
        document = document[part]
    return document


def _leaves(node):
    """Yield every scalar leaf of a nested document."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value)
    else:
        yield node


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=_search_documents(), seed=st.integers(0, 2**16))
def test_every_sampled_resolution_is_concrete_and_inside_its_declared_space(case, seed):
    """Any Optuna sample resolves to a plain document honoring bounds, steps and couplings."""
    text, specs, couplings = case
    source = load_tagged_yaml(text)
    space = SearchSpace.from_document(source)
    assert set(space.distributions) == set(specs)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    for _ in range(3):
        trial = study.ask(space.distributions)
        resolved = space.resolve(trial.params)
        study.tell(trial, 0.0)
        assert all(
            leaf is None or isinstance(leaf, (bool, int, float, str))
            for leaf in _leaves(resolved)
        )
        assert set(resolved) == set(source)
        assert resolved["fixed"] == {"keep": 7}
        for path, (kind, spec) in specs.items():
            value = _value_at(resolved, path)
            assert value == trial.params[path]
            if kind == "categorical":
                assert value in spec["choices"]
            else:
                assert spec["low"] <= value <= spec["high"]
            if kind == "int":
                assert (
                    isinstance(value, int) and (value - spec["low"]) % spec["step"] == 0
                )
            elif kind == "float_step":
                steps = (value - spec["low"]) / spec["step"]
                assert steps == pytest.approx(round(steps), abs=1e-6)
        for name, (target, values) in couplings.items():
            choices = specs[target][1]["choices"]
            assert resolved[name] == values[choices.index(_value_at(resolved, target))]


def test_prefixed_spaces_name_parameters_and_couplings_by_full_document_path():
    """A stage embedded in a larger document keeps the document's full parameter paths."""
    space = SearchSpace.from_document(
        load_tagged_yaml(
            """
initialization:
  head:
    replace: !suggest-categorical {choices: [linear, full]}
freeze: !coupled-sweep
  target: classifier.initialization.head.replace
  values: [11, 10]
tracked_metric: accuracy
"""
        ),
        protected=("tracked_metric",),
        prefix="classifier",
    )
    assert set(space.distributions) == {"classifier.initialization.head.replace"}
    resolved = space.resolve({"classifier.initialization.head.replace": "full"})
    assert (
        resolved["freeze"] == 10
        and resolved["initialization"]["head"]["replace"] == "full"
    )


def test_protected_paths_are_named_with_the_stage_prefix():
    """Messages use one naming style: the full pipeline path."""
    with pytest.raises(
        ValueError,
        match=r"'supcon\.temperature' cannot be searched: 'supcon\.temperature'",
    ):
        SearchSpace.from_document(
            load_tagged_yaml("temperature: !suggest-float {low: 0.05, high: 0.2}\n"),
            protected=("temperature",),
            prefix="supcon",
        )
