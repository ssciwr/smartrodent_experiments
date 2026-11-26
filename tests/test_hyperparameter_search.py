"""Optuna study settings and the generic study loop.

Trial execution is replaced by a scripted runner here; YAML loading, search-space
resolution, Optuna sampling/pruning/storage, and report writing stay real.
Search-space tags are tested in ``test_search_space.py``; trainer-specific
integration lives in ``test_training_studies.py``.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from study_test_support import STUDY_NAME, load_tagged_yaml, study_report, tuning

from smartrodent.experiment_results import ExperimentResult
from smartrodent.hyperparameter_search import (
    HyperparameterStudy,
    StudyConfiguration,
    StudyObjective,
    TrialConfigurationError,
)
from smartrodent.search_space import SearchSpace

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


# Schema rules once each (explicit optional sections, blank, strict and positive
# numbers, derived and unknown keys), plus every sampler/pruner resolution branch.
@pytest.mark.parametrize(
    "values",
    [
        tuning(sampler={"name": "FancySampler"}),
        tuning(sampler={"name": "BaseSampler"}),
        tuning(sampler={"name": "RandomSampler", "parameters": {"bogus": 1}}),
        tuning(pruner={"name": "FancyPruner"}),
        tuning(pruner={"name": "MedianPruner", "parameters": {"bogus": 1}}),
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
            SearchSpace.from_document(load_tagged_yaml(space)),
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
        SearchSpace.from_document(load_tagged_yaml(SPACE)),
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


def _identity_study(tmp_path, runner, identity, **settings):
    """A real study over SPACE with an extra stored identity."""
    return HyperparameterStudy(
        SearchSpace.from_document(load_tagged_yaml(SPACE)),
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
