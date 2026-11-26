"""Optuna search-space tags, study settings, and the generic study loop.

Trial execution is replaced by a scripted runner here; YAML loading, search-space
resolution, Optuna sampling/pruning/storage, and report writing stay real.
Trainer-specific integration lives in ``test_training_studies.py``.
"""

import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

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
from study_test_support import (
    STUDY_NAME,
    study_report,
    tuning,
)

from smartrodent import config_utils
from smartrodent.experiment_results import ExperimentResult
from smartrodent.hyperparameter_search import (
    HyperparameterStudy,
    StudyConfiguration,
    StudyObjective,
    TrialConfigurationError,
)
from smartrodent.search_space import SearchSpace


def _load(text: str):
    """Load YAML with the project's custom tags, exactly like every entry point."""
    return yaml.load(text, Loader=config_utils.get_loader())


def _space(text: str, **kwargs):
    """Build a search space from YAML text."""
    return SearchSpace.from_document(_load(text), **kwargs)


# --------------------------------------------------------------------------------------
# YAML tags -> named Optuna distributions
# --------------------------------------------------------------------------------------


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
    document = _load(
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


@pytest.mark.parametrize(
    "text",
    [
        "lr: !suggest-float {low: 0.1}\n",  # missing high
        "lr: !suggest-float {high: 0.1}\n",  # missing low
        "lr: !suggest-float {low: 0.1, high: 0.2, hihg: 3}\n",  # unknown key
        "lr: !suggest-float {low: 0.5, high: 0.1}\n",  # empty range
        "lr: !suggest-float {low: 0.0, high: 1.0, log: true}\n",  # log needs low > 0
        "lr: !suggest-float {low: 0.1, high: 1.0, log: true, step: 0.1}\n",  # log + step
        "lr: !suggest-float {low: 0.0, high: 1.0, step: 0.3}\n",  # range not a step multiple
        "lr: !suggest-float {low: 0.0, high: 1.0, step: 0.0}\n",  # nonpositive step
        "lr: !suggest-float {low: .nan, high: 1.0}\n",  # nonfinite bound
        "lr: !suggest-float {low: 0.0, high: .inf}\n",  # nonfinite bound
        "lr: !suggest-float {low: a, high: 1.0}\n",  # non-numeric bound
        "lr: !suggest-float {low: 0.0, high: 1.0, log: maybe}\n",  # non-boolean flag
        "lr: !suggest-float 0.1\n",  # scalar instead of mapping
        "n: !suggest-int {low: 0.5, high: 3}\n",  # float bound for an integer
        "n: !suggest-int {low: true, high: 3}\n",  # boolean is not an integer
        "n: !suggest-int {low: 5, high: 1}\n",  # empty range
        "n: !suggest-int {low: 0, high: 10, step: 3}\n",  # range not a step multiple
        "n: !suggest-int {low: 0, high: 10, step: 0}\n",  # nonpositive step
        "n: !suggest-int {low: 0, high: 10, log: true}\n",  # log needs low > 0
        "n: !suggest-int {low: 1, high: 10, log: true, step: 2}\n",  # log + step
        "c: !suggest-categorical {choices: []}\n",  # no choices
        "c: !suggest-categorical {choices: [a, a]}\n",  # ambiguous duplicate choice
        "c: !suggest-categorical {choices: [[1, 2], [3]]}\n",  # non-primitive choices
        "c: !suggest-categorical {choices: [{a: 1}]}\n",  # non-primitive choices
        "c: !suggest-categorical {values: [a, b]}\n",  # wrong key
        "c: !suggest-categorical [a, b]\n",  # sequence instead of mapping
    ],
)
def test_invalid_tag_definitions_are_rejected_with_value_errors(text):
    """Optuna would warn, adjust, or crash later; the project rejects these up front."""
    with pytest.raises(ValueError):
        SearchSpace.from_document(_load(text))


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
        SearchSpace.from_document(_load(text))


@pytest.mark.parametrize(
    "text,match",
    [
        (
            "lr: !suggest-float {low: 0.1, high: 0.9}\n"
            "batch: !coupled-sweep\n  target: lr\n  values: [1, 2]\n",
            "categorical",
        ),
        (
            "head: !suggest-categorical {choices: [a, b]}\n"
            "batch: !coupled-sweep\n  target: absent.path\n  values: [1, 2]\n",
            r"target.*absent\.path|absent\.path.*target",
        ),
        (
            "head: !suggest-categorical {choices: [a, b]}\n"
            "batch: !coupled-sweep\n  target: head\n  values: [1, 2, 3]\n",
            "length",
        ),
    ],
)
def test_invalid_couplings_are_rejected(text, match):
    """A coupled value must pair one-to-one with an existing categorical's choices."""
    with pytest.raises(ValueError, match=match):
        SearchSpace.from_document(_load(text))


@pytest.mark.parametrize(
    "text,path",
    [
        ("tracked_metric: !suggest-categorical {choices: [a, b]}\n", "tracked_metric"),
        (
            "validation_metrics:\n  f1:\n    average: !suggest-categorical {choices: [macro, micro]}\n",
            "validation_metrics.f1.average",
        ),
        (
            "head: !suggest-categorical {choices: [a, b]}\n"
            "tracked_metric: !coupled-sweep\n  target: head\n  values: [x, y]\n",
            "tracked_metric",
        ),
    ],
)
def test_protected_paths_and_their_children_cannot_be_searched(text, path):
    """Settings that define comparability must stay fixed, whether sampled or coupled."""
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        SearchSpace.from_document(
            _load(text), protected=("tracked_metric", "validation_metrics")
        )


@pytest.mark.parametrize(
    "text",
    [
        "lr: !suggest-float {low: 0.1, high: 0.9}\n",
        "optimizer:\n  lr: !suggest-float {low: 0.1, high: 0.9}\n",
        "scale: [!suggest-float {low: 0.5, high: 0.9}, 1.0]\n",
        "seed: !sweep {values: [1, 2]}\nhead: !suggest-categorical {choices: [a, b]}\n",
    ],
)
def test_grid_sweep_expansion_rejects_suggest_tags_without_a_study(text):
    """Decision: tags outside a study fail clearly instead of leaking into a run config."""
    with pytest.raises(ValueError, match="tuning"):
        config_utils.ConfigHandler(_load(text))


# --------------------------------------------------------------------------------------
# Property-based resolution
# --------------------------------------------------------------------------------------


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
    source = _load(text)
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


# --------------------------------------------------------------------------------------
# Study settings (the `tuning:` section)
# --------------------------------------------------------------------------------------


def test_study_settings_round_trip_with_explicit_optional_defaults():
    """Optional keys have visible defaults; omitted sampler parameters mean the constructor default."""
    parsed = StudyConfiguration.from_mapping(
        tuning(
            sampler={"name": "TPESampler"},
            pruner={"name": "MedianPruner", "parameters": {"n_startup_trials": 2}},
        )
    )
    assert parsed.to_dict() == {
        **tuning(),
        "sampler": {"name": "TPESampler", "parameters": {}},
        "pruner": {"name": "MedianPruner", "parameters": {"n_startup_trials": 2}},
        "load_if_exists": False,
        "timeout": None,
    }


@pytest.mark.parametrize(
    "values",
    [
        {k: v for k, v in tuning().items() if k != "study_name"},
        {
            k: v for k, v in tuning().items() if k != "storage"
        },  # in-memory must be explicit
        {
            k: v for k, v in tuning().items() if k != "pruner"
        },  # no pruning must be explicit
        {k: v for k, v in tuning().items() if k != "sampler"},
        {k: v for k, v in tuning().items() if k != "n_trials"},
        tuning(study_name=""),
        tuning(storage=""),
        tuning(n_trials=0),
        tuning(n_trials="3"),
        tuning(n_trials=2.0),
        tuning(timeout=0),
        tuning(timeout=-1.0),
        tuning(load_if_exists="yes"),
        tuning(sampler=None),
        tuning(sampler={"name": "FancySampler"}),
        tuning(sampler={"name": "BaseSampler"}),
        tuning(sampler={"name": "RandomSampler", "parameters": {"bogus": 1}}),
        tuning(sampler={"name": "RandomSampler", "options": {}}),
        tuning(pruner={"name": "FancyPruner"}),
        tuning(pruner={"name": "MedianPruner", "parameters": {"bogus": 1}}),
        tuning(
            direction="maximize"
        ),  # derived from tracked_mode, never configured twice
        tuning(unknown=True),
    ],
)
def test_invalid_study_settings_fail_before_any_trial(values):
    """Unknown, missing, or unconstructible settings are configuration errors."""
    with pytest.raises(ValueError):
        StudyConfiguration.from_mapping(values)


# --------------------------------------------------------------------------------------
# Generic study loop with a scripted trial runner
# --------------------------------------------------------------------------------------

SPACE = """
lr: !suggest-float {low: 0.0, high: 1.0}
head: !suggest-categorical {choices: [linear, full]}
freeze: !coupled-sweep
  target: head
  values: [11, 10]
"""


class ScriptedRunner:
    """Write distinct trial artifacts and report a score derived from the sampled document.

    Args:
        validation: Validation split reported by every result.
        score: Maps a resolved document to the reported best value.
        epoch_values: Maps a resolved document to per-epoch values reported to the hook.
        failures: Call index -> exception to raise instead of training.
        mode: Reported tracked mode.
        overrides: Call index -> ExperimentResult fields replacing the defaults.
        on_call: Called with (call index, trial directory) before anything else.
    """

    def __init__(
        self,
        validation,
        *,
        score=lambda document: document["lr"],
        epoch_values=lambda document: [],
        failures=None,
        mode="max",
        overrides=None,
        on_call=None,
    ):
        self.validation = validation
        self.score, self.epoch_values = score, epoch_values
        self.failures, self.mode = failures or {}, mode
        self.overrides, self.on_call = overrides or {}, on_call
        self.calls, self.results = [], []

    def __call__(self, document, output_directory, epoch_callback):
        index = len(self.calls)
        self.calls.append(
            SimpleNamespace(document=document, output_directory=output_directory)
        )
        if self.on_call is not None:
            self.on_call(index, output_directory)
        if index in self.failures:
            raise self.failures[index]
        for epoch, value in enumerate(self.epoch_values(document), start=1):
            epoch_callback(epoch, value)
        output_directory.mkdir(parents=True, exist_ok=False)
        best, last = output_directory / "best.pt", output_directory / "last.pt"
        best.write_text(f"best {index}")
        last.write_text(f"last {index}")
        fields = {
            "output_directory": output_directory,
            "last_checkpoint": last,
            "best_checkpoint": best,
            "best_epoch": 1,
            "best_value": self.score(document),
            "tracked_metric": "balanced_accuracy",
            "tracked_mode": self.mode,
            "validation_dataset": self.validation,
        }
        result = ExperimentResult(**{**fields, **self.overrides.get(index, {})})
        self.results.append(result)
        return result


@pytest.fixture
def study_case(tmp_path):
    """Build real studies around a scripted runner, varying settings per test."""

    def build(runner=None, *, mode="max", space=SPACE, root="study", **study_settings):
        validation = tmp_path / "val"
        runner = runner if runner is not None else ScriptedRunner(validation, mode=mode)
        study = HyperparameterStudy(
            SearchSpace.from_document(_load(space)),
            StudyConfiguration.from_mapping(tuning(**study_settings)),
            runner,
            output_directory=tmp_path / root,
            objective=StudyObjective(
                tracked_metric="balanced_accuracy",
                tracked_mode=mode,
                validation_dataset=validation,
            ),
        )
        return SimpleNamespace(
            study=study, runner=runner, root=tmp_path / root, validation=validation
        )

    return build


@pytest.mark.parametrize("mode", ["max", "min"])
def test_study_runs_isolated_trials_and_selects_the_configured_direction(
    study_case, mode
):
    """Each trial gets a concrete document and its own directory; the winner follows tracked_mode."""
    case = study_case(mode=mode, n_trials=5)
    result = case.study.run()
    calls = case.runner.calls
    assert [call.output_directory for call in calls] == [
        case.root / f"trial_{n:04d}" for n in range(5)
    ]
    for call in calls:
        assert set(call.document) == {"lr", "head", "freeze"}
        assert (
            call.document["freeze"] == {"linear": 11, "full": 10}[call.document["head"]]
        )
    assert list(result.results) == case.runner.results
    rank = max if mode == "max" else min
    assert result.selected == rank(
        case.runner.results, key=lambda item: item.best_value
    )
    winner = next(
        call
        for call in calls
        if call.output_directory == result.selected.output_directory
    )
    assert result.best_params == {
        "lr": winner.document["lr"],
        "head": winner.document["head"],
    }


def test_exact_ties_keep_the_earliest_trial_like_grid_sweeps(study_case, tmp_path):
    """The established first-tie rule also applies to studies."""
    runner = ScriptedRunner(tmp_path / "val", score=lambda document: 0.5)
    result = study_case(runner, n_trials=3).study.run()
    assert result.selected.output_directory.name == "trial_0000"


def test_seeded_samplers_make_independent_studies_reproducible(study_case, tmp_path):
    """The sampler seed alone determines the sampled trial sequence."""
    sequences = []
    for name in ("first", "second"):
        runner = ScriptedRunner(tmp_path / "val")
        sampler = {"name": "TPESampler", "parameters": {"seed": 5}}
        study_case(runner, n_trials=4, sampler=sampler, root=name).study.run()
        sequences.append([call.document for call in runner.calls])
    assert sequences[0] == sequences[1]


def test_pruner_stops_weak_trials_from_reported_epoch_values(study_case, tmp_path):
    """The study's epoch hook reports values to Optuna and stops trials the pruner rejects."""
    runner = ScriptedRunner(
        tmp_path / "val",
        score=lambda d: d["lr"],
        epoch_values=lambda d: [d["lr"], d["lr"]],
    )
    case = study_case(
        runner,
        space="lr: !suggest-categorical {choices: [0.1, 0.9]}\n",
        n_trials=5,  # BruteForceSampler ends the study after exhausting both choices
        sampler={"name": "BruteForceSampler", "parameters": {"seed": 0}},
        pruner={"name": "ThresholdPruner", "parameters": {"lower": 0.5}},
    )
    result = case.study.run()
    trials = {
        trial["params"]["lr"]: trial for trial in study_report(case.root)["trials"]
    }
    assert set(trials) == {0.1, 0.9}
    assert trials[0.1]["state"] == "PRUNED" and trials[0.1]["intermediate_values"] == {
        1: 0.1
    }
    assert trials[0.1]["result"] is None
    assert trials[0.9]["state"] == "COMPLETE" and trials[0.9][
        "intermediate_values"
    ] == {1: 0.9, 2: 0.9}
    assert [item.best_value for item in result.results] == [0.9]
    assert result.best_params == {"lr": 0.9}


@pytest.mark.parametrize(
    "error",
    ["nonfinite", "invalid_combination"],
)
def test_anticipated_trial_failures_are_recorded_and_the_study_continues(
    study_case, tmp_path, error
):
    """Diverging runs and invalid sampled combinations fail one trial, not the study."""
    exception = (
        FloatingPointError("Nonfinite tracked validation metric")
        if error == "nonfinite"
        else TrialConfigurationError("classes_per_batch * samples_per_class too large")
    )
    runner = ScriptedRunner(tmp_path / "val", failures={1: exception})
    case = study_case(runner, n_trials=3)
    result = case.study.run()
    assert len(runner.calls) == 3 and len(result.results) == 2
    trials = study_report(case.root)["trials"]
    assert [trial["state"] for trial in trials] == ["COMPLETE", "FAIL", "COMPLETE"]
    assert str(exception) in trials[1]["error"]
    assert trials[0]["error"] is None and trials[1]["result"] is None


def test_a_study_objective_requires_a_validation_split():
    """Ranking training-only runs is rejected for studies, as for sweep selection."""
    with pytest.raises(ValueError, match="validation"):
        StudyObjective(
            tracked_metric="supcon_loss_mean",
            tracked_mode="min",
            validation_dataset=None,
        )


def test_invalid_trial_configuration_errors_are_value_errors():
    """Callers that only know configuration errors can still handle them uniformly."""
    assert issubclass(TrialConfigurationError, ValueError)


def test_unexpected_trial_errors_stop_the_study_and_are_recorded(study_case, tmp_path):
    """Anything unanticipated propagates unchanged, after the trial is marked failed in storage."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    runner = ScriptedRunner(
        tmp_path / "val", failures={1: RuntimeError("deliberate crash")}
    )
    case = study_case(runner, n_trials=4, storage=storage)
    with pytest.raises(RuntimeError, match="deliberate crash"):
        case.study.run()
    assert len(runner.calls) == 2
    stored = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    assert [trial.state.name for trial in stored.trials] == ["COMPLETE", "FAIL"]
    report = study_report(case.root)
    assert [trial["state"] for trial in report["trials"]] == ["COMPLETE", "FAIL"]
    assert "deliberate crash" in report["trials"][1]["error"]


def test_a_study_without_any_completed_trial_reports_no_winner_and_fails(
    study_case, tmp_path
):
    """No partial or failed trial can be selected; the report still lists every attempt."""
    failures = {index: FloatingPointError("diverged") for index in range(3)}
    runner = ScriptedRunner(tmp_path / "val", failures=failures)
    case = study_case(runner, n_trials=3)
    with pytest.raises(ValueError, match="no completed trial"):
        case.study.run()
    report = study_report(case.root)
    assert report["selected_trial"] is None
    assert [trial["state"] for trial in report["trials"]] == ["FAIL"] * 3


@pytest.mark.parametrize(
    "override",
    [
        {"tracked_metric": "accuracy"},
        {"tracked_mode": "min"},
        {"validation_dataset": Path("/elsewhere/val")},
    ],
)
def test_results_that_disagree_with_the_study_objective_are_rejected(
    study_case, tmp_path, override
):
    """A trial cannot be ranked against others when it reports a different objective."""
    runner = ScriptedRunner(tmp_path / "val", overrides={0: override})
    with pytest.raises(ValueError):
        study_case(runner, n_trials=2).study.run()


def test_validation_path_aliases_are_the_same_objective(study_case, tmp_path):
    """Equivalent paths to the shared split remain comparable, as in grid sweeps."""
    (tmp_path / "val").mkdir()
    alias = tmp_path / "val" / ".." / "val"
    runner = ScriptedRunner(
        tmp_path / "val", overrides={0: {"validation_dataset": alias}}
    )
    result = study_case(runner, n_trials=2).study.run()
    assert len(result.results) == 2


def test_report_is_rewritten_after_every_trial(study_case, tmp_path):
    """A crash or interruption leaves an accurate partial report."""
    observed = {}

    def on_call(index, output_directory):
        if index == 1:
            observed["trials"] = study_report(output_directory.parent)["trials"]

    runner = ScriptedRunner(tmp_path / "val", on_call=on_call)
    study_case(runner, n_trials=2).study.run()
    assert [trial["state"] for trial in observed["trials"]] == ["COMPLETE"]


def test_report_describes_every_trial_and_the_selected_one(study_case):
    """The portable report matches the returned result and holds everything to reproduce a trial."""
    case = study_case(n_trials=3)
    result = case.study.run()
    report = study_report(case.root)
    assert report == result.to_dict()
    assert report["study_name"] == STUDY_NAME and report["direction"] == "maximize"
    assert [trial["number"] for trial in report["trials"]] == [0, 1, 2]
    for trial, call in zip(report["trials"], case.runner.calls, strict=True):
        assert trial["state"] == "COMPLETE"
        assert trial["params"] == {
            "lr": call.document["lr"],
            "head": call.document["head"],
        }
        assert trial["result"]["output_directory"] == str(call.output_directory)
        assert trial["value"] == trial["result"]["best_value"]
    selected = report["selected_trial"]
    assert selected["params"] == result.best_params
    assert selected["result"]["best_checkpoint"] == str(result.selected.best_checkpoint)


def test_time_budget_ends_the_study_before_n_trials(study_case, tmp_path):
    """No new trial starts once the wall-clock budget is used up."""
    runner = ScriptedRunner(
        tmp_path / "val", on_call=lambda index, directory: time.sleep(0.05)
    )
    result = study_case(runner, n_trials=10, timeout=0.01).study.run()
    assert len(runner.calls) == 1 and len(result.results) == 1


def test_resumed_studies_continue_numbering_and_rank_all_completed_trials(
    study_case, tmp_path
):
    """Persistent storage resumes a study; earlier winners and artifacts survive later runs."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    first = ScriptedRunner(
        tmp_path / "val", score=lambda document: 1.0 + document["lr"]
    )
    first_result = study_case(
        first, n_trials=2, storage=storage, load_if_exists=True
    ).study.run()
    second = ScriptedRunner(
        tmp_path / "val", score=lambda document: document["lr"] - 1.0
    )
    case = study_case(second, n_trials=2, storage=storage, load_if_exists=True)
    result = case.study.run()
    assert [call.output_directory.name for call in second.calls] == [
        "trial_0002",
        "trial_0003",
    ]
    assert [path.name for path in sorted(case.root.glob("trial_*"))] == [
        f"trial_{n:04d}" for n in range(4)
    ]
    assert len(result.results) == 4
    assert (
        result.selected == first_result.selected
    )  # reconstructed from storage, not rerun
    assert [trial["number"] for trial in study_report(case.root)["trials"]] == [
        0,
        1,
        2,
        3,
    ]


def test_existing_studies_are_not_resumed_unless_requested(study_case, tmp_path):
    """Reusing a study name requires load_if_exists; nothing runs otherwise."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study_case(n_trials=1, storage=storage).study.run()
    runner = ScriptedRunner(tmp_path / "val")
    with pytest.raises(ValueError, match="load_if_exists"):
        study_case(runner, n_trials=1, storage=storage).study.run()
    assert runner.calls == []
    assert len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials) == 1


@pytest.mark.parametrize(
    "changed",
    [
        SPACE.replace("high: 1.0", "high: 2.0"),
        SPACE.replace("[linear, full]", "[linear, full, none]").replace(
            "[11, 10]", "[11, 10, 0]"
        ),
        "lr: !suggest-float {low: 0.0, high: 1.0}\n",
    ],
)
def test_resuming_with_a_different_search_space_fails_before_any_trial(
    study_case, tmp_path, changed
):
    """Trials of one study must come from one search space to be comparable."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study_case(n_trials=1, storage=storage, load_if_exists=True).study.run()
    runner = ScriptedRunner(tmp_path / "val")
    with pytest.raises(ValueError, match="search space"):
        study_case(
            runner, space=changed, n_trials=1, storage=storage, load_if_exists=True
        ).study.run()
    assert runner.calls == []
    assert len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials) == 1


def test_resuming_with_the_opposite_direction_fails_before_any_trial(
    study_case, tmp_path
):
    """A stored maximize study cannot silently continue as a minimize study."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study_case(n_trials=1, storage=storage, load_if_exists=True, mode="max").study.run()
    runner = ScriptedRunner(tmp_path / "val", mode="min")
    with pytest.raises(ValueError, match="direction"):
        study_case(
            runner, mode="min", n_trials=1, storage=storage, load_if_exists=True
        ).study.run()
    assert runner.calls == []
    assert len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials) == 1


# --- Review round 1 additions ---


@pytest.mark.parametrize(
    "objective",
    [
        {"tracked_metric": "accuracy"},
        {"validation_dataset": "other-val"},
    ],
)
def test_resuming_with_a_different_objective_fails_before_any_trial(
    study_case, tmp_path, objective
):
    """Stored trials ranked on another metric or split cannot be compared with new ones."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    study_case(n_trials=1, storage=storage, load_if_exists=True).study.run()
    values = {
        "tracked_metric": "balanced_accuracy",
        "tracked_mode": "max",
        "validation_dataset": tmp_path / "val",
    }
    values.update(
        {
            key: tmp_path / value if key == "validation_dataset" else value
            for key, value in objective.items()
        }
    )
    runner = ScriptedRunner(values["validation_dataset"])
    study = HyperparameterStudy(
        SearchSpace.from_document(_load(SPACE)),
        StudyConfiguration.from_mapping(
            tuning(n_trials=1, storage=storage, load_if_exists=True)
        ),
        runner,
        output_directory=tmp_path / "study",
        objective=StudyObjective(**values),
    )
    with pytest.raises(ValueError, match="objective"):
        study.run()
    assert runner.calls == []
    assert len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials) == 1


def test_another_study_cannot_reuse_an_output_directory(study_case, tmp_path):
    """Trial directories of one study must never be overwritten by a differently named study."""
    study_case(n_trials=1).study.run()
    runner = ScriptedRunner(tmp_path / "val")
    with pytest.raises(ValueError, match="study_name"):
        study_case(runner, n_trials=1, study_name="another-study").study.run()
    assert runner.calls == []
    assert study_report(tmp_path / "study")["study_name"] == STUDY_NAME


def test_prefixed_spaces_name_parameters_and_couplings_by_full_document_path():
    """A stage embedded in a larger document keeps the document's full parameter paths."""
    space = SearchSpace.from_document(
        _load(
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


# --- Review round 2 additions ---


def _identity_study(tmp_path, runner, identity, **settings):
    """A real study over SPACE with an extra stored identity."""
    return HyperparameterStudy(
        SearchSpace.from_document(_load(SPACE)),
        StudyConfiguration.from_mapping(tuning(**settings)),
        runner,
        output_directory=tmp_path / "study",
        objective=StudyObjective(
            tracked_metric="balanced_accuracy",
            tracked_mode="max",
            validation_dataset=tmp_path / "val",
        ),
        identity=identity,
    )


def test_resuming_requires_the_same_extra_identity(tmp_path):
    """Callers can pin more than space and objective, e.g. the encoder all trials share."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    settings = {"n_trials": 1, "storage": storage, "load_if_exists": True}
    _identity_study(
        tmp_path,
        ScriptedRunner(tmp_path / "val"),
        {"encoder checkpoint": "sha-a"},
        **settings,
    ).run()
    same = ScriptedRunner(tmp_path / "val")
    _identity_study(tmp_path, same, {"encoder checkpoint": "sha-a"}, **settings).run()
    assert len(same.calls) == 1
    changed = ScriptedRunner(tmp_path / "val")
    with pytest.raises(ValueError, match="encoder checkpoint"):
        _identity_study(
            tmp_path, changed, {"encoder checkpoint": "sha-b"}, **settings
        ).run()
    assert changed.calls == []


def test_preflight_reports_conflicts_without_running_or_recording_anything(
    study_case, tmp_path
):
    """The check used before long earlier stages: read-only, and silent when all is well."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    fresh = study_case(n_trials=1, storage=storage)
    fresh.study.preflight()
    assert optuna.get_all_study_names(storage=storage) == []
    assert not fresh.root.exists()
    fresh.study.run()
    with pytest.raises(ValueError, match="load_if_exists"):
        study_case(n_trials=1, storage=storage).study.preflight()
    with pytest.raises(ValueError, match="study_name"):
        study_case(n_trials=1, study_name="another-study").study.preflight()
    with pytest.raises(ValueError, match="search space"):
        study_case(
            space="lr: !suggest-float {low: 0.0, high: 1.0}\n",
            n_trials=1,
            storage=storage,
            load_if_exists=True,
        ).study.preflight()
    study_case(n_trials=1, storage=storage, load_if_exists=True).study.preflight()
    assert len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials) == 1


def test_protected_paths_are_named_with_the_stage_prefix():
    """Messages use one naming style: the full pipeline path."""
    with pytest.raises(
        ValueError,
        match=r"'supcon\.temperature' cannot be searched: 'supcon\.temperature'",
    ):
        SearchSpace.from_document(
            _load("temperature: !suggest-float {low: 0.05, high: 0.2}\n"),
            protected=("temperature",),
            prefix="supcon",
        )
