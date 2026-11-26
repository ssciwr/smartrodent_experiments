"""Classifier sweep contracts shared by the standalone and two-stage entry points.

Only stage execution is replaced in most tests: YAML loading, expansion,
configuration validation, orchestration, ranking, and summary writing stay real.
"""

import shutil
import warnings
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from training_test_support import (
    invoke_entrypoint,
    selection_summary,
    serialized_result,
)
from ultralytics import YOLO

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.experiment_results import ExperimentResult


def _assert_saved_results(summary, results):
    """Require an ordered, complete report of every run, allowing extra provenance fields."""
    saved_results = summary["classifier_results"]
    assert len(saved_results) == len(results)
    for saved, result in zip(saved_results, results, strict=True):
        assert serialized_result(result).items() <= saved.items()


@pytest.fixture
def standalone_document(classifier_configuration):
    """Supply real temporary inputs so no project checkpoint or dataset is touched."""
    return classifier_configuration.to_dict()


@pytest.fixture
def recorded_classifiers(recorded_stages, monkeypatch):
    """Return controlled best-epoch results while keeping configuration handling real."""
    observed = SimpleNamespace(
        configurations=[], results=[], values=[], fail_at=None, overrides={}
    )

    def run_with_result(self, **kwargs):
        """Write distinct artifacts and report scores independent of execution order."""
        configuration = self.configuration
        index = len(observed.configurations)
        observed.configurations.append(configuration)
        recorded_stages.events.append("classifier")
        if index == observed.fail_at:
            raise RuntimeError("deliberate classifier sweep failure")
        directory = configuration.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights").mkdir(exist_ok=True)
        best, last = directory / "weights/best.pt", directory / "weights/last.pt"
        torch.save({"run": index, "epoch": 2}, best)
        torch.save({"run": index, "epoch": 3}, last)
        (directory / "config.yaml").write_text(yaml.safe_dump(configuration.to_dict()))
        (directory / "results.csv").write_text(f"epoch,score\n2,{index}\n")
        result = ExperimentResult(
            output_directory=directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=2,
            best_value=observed.values[index] if observed.values else 0.4,
            tracked_metric=configuration.tracked_metric,
            tracked_mode=configuration.effective_tracked_mode,
            validation_dataset=configuration.validation_dataset,
        )
        result = replace(result, **observed.overrides.get(index, {}))
        observed.results.append(result)
        return result

    monkeypatch.setattr(ClassifierExperiment, "run_with_result", run_with_result)
    return observed


# Ranking and validation rules live in the shared sweep code; tests of those rules
# run through the standalone entry point only. Both entry points are kept where the
# entry point changes paths, coupled targets or summaries.
SHARED_RULE = pytest.mark.parametrize("sweep_case", ["standalone"], indirect=True)


@pytest.fixture(params=["standalone", "two_stage"])
def sweep_case(
    request,
    standalone_document,
    sweep_pipeline_document,
    recorded_classifiers,
    tmp_path,
    monkeypatch,
):
    """Apply identical classifier behavior tests to both supported orchestration paths."""
    entrypoint = request.param
    document = (
        standalone_document if entrypoint == "standalone" else sweep_pipeline_document
    )
    stage = document if entrypoint == "standalone" else document["classifier"]
    root = Path(document["output_directory"])
    classifier_root = root if entrypoint == "standalone" else root / "classifier"
    path = tmp_path / f"{entrypoint}.yaml"

    def run(*, text=None):
        """Load and execute this case through its production entry point."""
        return invoke_entrypoint(entrypoint, document, path, monkeypatch, text=text)

    return SimpleNamespace(
        entrypoint=entrypoint,
        document=document,
        stage=stage,
        root=root,
        classifier_root=classifier_root,
        run=run,
        path=path,
    )


def test_classifier_cartesian_and_coupled_sweeps_reuse_one_encoder(
    sweep_case, recorded_classifiers, recorded_stages, classifier_assets
):
    """Head-source/checkpoint coupling must coexist with independent replacement sweeps."""
    case = sweep_case
    head = case.stage["initialization"]["head"]
    head.update(
        replace={"type": "sweep", "values": ["linear", "full"]},
        source={"type": "sweep", "values": ["random", "yolo_checkpoint"]},
        checkpoint={
            "type": "coupled-sweep",
            "target": (["classifier"] if case.entrypoint == "two_stage" else [])
            + ["initialization", "head", "source"],
            "values": [None, str(classifier_assets.ultralytics)],
        },
    )
    before = deepcopy(case.document)
    result = case.run()
    configurations = recorded_classifiers.configurations
    assert [
        (c.initialization["head"]["replace"], c.initialization["head"]["source"])
        for c in configurations
    ] == list(product(["linear", "full"], ["random", "yolo_checkpoint"]))
    for configuration in configurations:
        selected_head = configuration.initialization["head"]
        expected = (
            None
            if selected_head["source"] == "random"
            else str(classifier_assets.ultralytics)
        )
        assert selected_head.get("checkpoint") == expected
    assert [c.output_directory for c in configurations] == [
        case.classifier_root / f"run_{i:03d}" for i in range(1, 5)
    ]
    sources = {
        Path(c.initialization["encoder"]["checkpoint"]).resolve()
        for c in configurations
    }
    assert len(sources) == 1
    if case.entrypoint == "two_stage":
        assert len(recorded_stages.supcon) == 3
        assert recorded_stages.events == ["supcon"] * 3 + ["classifier"] * 4
        assert sources == {result.selected_supcon.best_checkpoint.resolve()}
        assert tuple(result.classifier_results) == tuple(recorded_classifiers.results)
        assert result.selected_classifier == result.classifier_results[0]
    elif case.entrypoint == "standalone":
        assert recorded_stages.supcon == []
        assert sources == {
            Path(before["initialization"]["encoder"]["checkpoint"]).resolve()
        }
    assert case.document == before
    summary = selection_summary(case.document)
    _assert_saved_results(summary, recorded_classifiers.results)
    assert summary["selected_classifier"]["best_checkpoint"] == str(
        recorded_classifiers.results[0].best_checkpoint
    )
    assert (
        len({c.best_checkpoint.read_bytes() for c in recorded_classifiers.results}) == 4
    )


@pytest.fixture
def thousand_class_head(classifier_assets, tmp_path):
    """Use a real local native checkpoint with the stock checkpoint's output count."""
    yolo = YOLO(str(classifier_assets.ultralytics), task="classify")
    yolo.model.model[-1].linear = torch.nn.Linear(
        yolo.model.model[-1].linear.in_features, 1000
    )
    yolo.model.names = {index: f"imagenet_{index}" for index in range(1000)}
    path = tmp_path / "thousand-class.pt"
    yolo.save(path)
    return path


@SHARED_RULE
def test_redundant_linear_initialization_is_flagged_before_training_without_deduplication(
    sweep_case, recorded_classifiers, recorded_stages, thousand_class_head, monkeypatch
):
    """Warn before SupCon/CE starts, but preserve all explicitly configured variants."""
    case = sweep_case
    case.stage["initialization"]["head"] = {
        "replace": {"type": "sweep", "values": ["linear", "full"]},
        "source": {"type": "sweep", "values": ["random", "yolo_checkpoint"]},
        "checkpoint": {
            "type": "coupled-sweep",
            "target": (["classifier"] if case.entrypoint == "two_stage" else [])
            + ["initialization", "head", "source"],
            "values": [None, str(thousand_class_head)],
        },
    }
    original = ClassifierExperiment.run_with_result

    # The wrappers read `caught` when called, i.e. inside the warns block below.
    def run(self, **kwargs):
        assert any(
            "1000" in str(warning.message) and "2" in str(warning.message)
            for warning in caught
        )
        return original(self, **kwargs)

    monkeypatch.setattr(ClassifierExperiment, "run_with_result", run)
    if case.entrypoint == "two_stage":
        from smartrodent.supcon_training import SupConExperiment

        original_supcon = SupConExperiment.run_with_result

        def run_supcon(self, **kwargs):
            assert caught, (
                "The redundant-head warning arrived only after SupCon training"
            )
            return original_supcon(self, **kwargs)

        monkeypatch.setattr(SupConExperiment, "run_with_result", run_supcon)
    with pytest.warns(UserWarning, match="(?i)linear.*random|random.*linear") as caught:
        case.run()
    assert len(recorded_classifiers.results) == 4
    assert len(selection_summary(case.document)["classifier_results"]) == 4


@SHARED_RULE
@pytest.mark.parametrize("boundary,source_count", [("linear", 2), ("full", 1000)])
def test_distinct_head_initializations_are_not_flagged_as_redundant(
    sweep_case, classifier_assets, thousand_class_head, boundary, source_count
):
    """A same-size linear donor or a transferred full head retains meaningful pretrained weights."""
    checkpoint = (
        classifier_assets.ultralytics if source_count == 2 else thousand_class_head
    )
    sweep_case.stage["initialization"]["head"] = {
        "replace": boundary,
        "source": "yolo_checkpoint",
        "checkpoint": str(checkpoint),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep_case.run()
    assert not any(
        "linear" in str(w.message).lower() and "random" in str(w.message).lower()
        for w in caught
    )


@SHARED_RULE
@pytest.mark.parametrize(
    "field,values",
    [
        ("encoder_batchnorm", ["eval", "adapt"]),
        ("freeze", [1, [0, 1]]),
        ("batch_size", [2, 4]),
    ],
)
def test_classifier_sweeps_are_not_limited_to_head_parameters(
    sweep_case, recorded_classifiers, field, values
):
    """Ordinary valid classifier settings retain the existing general sweep semantics."""
    sweep_case.stage[field] = {"type": "sweep", "values": values}
    sweep_case.run()
    assert [getattr(c, field) for c in recorded_classifiers.configurations] == values


def test_custom_yaml_tags_and_full_document_coupled_targets(
    sweep_case, recorded_classifiers
):
    """Use actual YAML tags, not only the equivalent preconstructed mapping nodes."""
    case = sweep_case
    case.stage["train_kwargs"]["lr0"] = "SWEEP_LEARNING_RATE"
    case.stage["batch_size"] = "COUPLED_BATCH_SIZE"
    target = (
        "classifier.train_kwargs.lr0"
        if case.entrypoint == "two_stage"
        else "train_kwargs.lr0"
    )
    indent = "  " if case.entrypoint == "two_stage" else ""
    text = (
        yaml.safe_dump(case.document)
        .replace(
            "lr0: SWEEP_LEARNING_RATE",
            f"lr0: !sweep\n{indent}    values: [0.001, 0.002]",
        )
        .replace(
            "batch_size: COUPLED_BATCH_SIZE",
            f"batch_size: !coupled-sweep\n{indent}  target: {target}\n{indent}  values: [2, 4]",
        )
    )
    case.run(text=text)
    assert [
        (c.train_kwargs["lr0"], c.batch_size)
        for c in recorded_classifiers.configurations
    ] == [(0.001, 2), (0.002, 4)]
    summary = selection_summary(case.document)
    assert len(summary["classifier_results"]) == 2


@pytest.mark.parametrize(
    "sweep_case,mode,values,winner",
    [
        ("standalone", "max", [0.2, 0.9, 0.4], 1),
        ("standalone", "min", [0.8, 0.1, 0.6], 1),
        ("standalone", "max", [0.9, 0.9, 0.2], 0),
        ("standalone", "min", [0.1, 0.1, 0.8], 0),
        # the pipeline result's classifier_result raises for a multi-run sweep
        ("two_stage", "max", [0.2, 0.9, 0.4], 1),
    ],
    indirect=["sweep_case"],
)
def test_sweep_champion_uses_best_validation_result_and_stable_ties(
    sweep_case, recorded_classifiers, mode, values, winner
):
    """Choose a best checkpoint by the configured objective, never the last run/epoch."""
    case = sweep_case
    case.stage["train_kwargs"]["lr0"] = {
        "type": "sweep",
        "values": [0.001, 0.002, 0.003],
    }
    case.stage["tracked_mode"] = mode
    recorded_classifiers.values = values
    result = case.run()
    selected = recorded_classifiers.results[winner]
    summary = selection_summary(case.document)
    _assert_saved_results(summary, recorded_classifiers.results)
    assert summary["selected_classifier"]["best_checkpoint"] == str(
        selected.best_checkpoint
    )
    assert summary["selected_classifier"]["best_epoch"] == 2
    assert summary["selected_classifier"]["best_value"] == values[winner]
    assert (
        summary["selected_classifier"]["tracked_metric"] == case.stage["tracked_metric"]
    )
    assert summary["selected_classifier"]["tracked_mode"] == mode
    if case.entrypoint == "two_stage":
        assert result.selected_classifier == selected
        with pytest.raises(ValueError, match="multiple|ambiguous|sweep"):
            _ = result.classifier_result


@pytest.mark.parametrize("tagged_singleton", [False, True])
def test_single_classifier_keeps_existing_output_path_and_singular_result(
    sweep_case, recorded_classifiers, tagged_singleton
):
    """A non-sweep or singleton sweep must not introduce a gratuitous run subdirectory."""
    case = sweep_case
    if tagged_singleton:
        case.stage["train_kwargs"]["lr0"] = {"type": "sweep", "values": [0.004]}
    result = case.run()
    assert len(recorded_classifiers.results) == 1
    selected = recorded_classifiers.results[0]
    assert selected.output_directory == case.classifier_root
    assert selection_summary(case.document)["selected_classifier"][
        "best_checkpoint"
    ] == str(selected.best_checkpoint)
    if case.entrypoint == "two_stage":
        assert result.classifier_result == result.selected_classifier == selected
        assert tuple(result.classifier_results) == (selected,)


@SHARED_RULE
@pytest.mark.parametrize("values", [[], [0.001, 0]])
def test_all_classifier_variants_are_validated_before_any_training(
    sweep_case, recorded_classifiers, recorded_stages, values
):
    """An empty sweep or invalid later variant must not waste a SupCon/CE run."""
    sweep_case.stage["train_kwargs"]["lr0"] = {"type": "sweep", "values": values}
    with pytest.raises(ValueError, match="empty|sweep|candidate|optimizer|learning|lr"):
        sweep_case.run()
    assert recorded_classifiers.configurations == []
    assert recorded_stages.supcon == []
    assert not sweep_case.root.exists()


@SHARED_RULE
@pytest.mark.parametrize("field", ["tracked_metric", "tracked_mode"])
def test_incompatible_classifier_ranking_objectives_are_rejected_before_training(
    sweep_case, recorded_classifiers, recorded_stages, field
):
    """Do not compare scores with different meanings or opposite directions."""
    case = sweep_case
    case.stage["validation_metrics"] = {
        "balanced_accuracy": {"function": "balanced_accuracy_score"}
    }
    values = (
        ["ce_loss_mean", "balanced_accuracy"]
        if field == "tracked_metric"
        else ["min", "max"]
    )
    case.stage[field] = {"type": "sweep", "values": values}
    with pytest.raises(ValueError, match="metric|mode|direction|compar"):
        case.run()
    assert recorded_classifiers.configurations == []
    assert recorded_stages.supcon == []


@SHARED_RULE
@pytest.mark.parametrize(
    "definitions",
    [
        [{"function": "accuracy_score"}, {"function": "balanced_accuracy_score"}],
        [
            {"function": "accuracy_score", "normalize": True},
            {"function": "accuracy_score", "normalize": False},
        ],
    ],
)
def test_same_metric_name_with_different_definitions_is_rejected_before_training(
    sweep_case, recorded_classifiers, recorded_stages, definitions
):
    """A shared display name is insufficient when scorers measure different objectives."""
    sweep_case.stage["tracked_metric"] = "score"
    sweep_case.stage["tracked_mode"] = "max"
    sweep_case.stage["validation_metrics"] = {
        "score": {"type": "sweep", "values": definitions}
    }
    with pytest.raises(ValueError, match="metric|definition|compar"):
        sweep_case.run()
    assert recorded_classifiers.configurations == []
    assert recorded_stages.supcon == []


def test_equivalent_validation_path_aliases_rank_without_altering_reported_paths(
    standalone_document, recorded_classifiers, tmp_path, monkeypatch
):
    """Canonical input comparison must stay consistent through preparation and ranking."""
    original = standalone_document["validation_dataset"]
    alias = tmp_path / "validation-alias"
    alias.symlink_to(original, target_is_directory=True)
    standalone_document["validation_dataset"] = {
        "type": "sweep",
        "values": [original, str(alias)],
    }
    recorded_classifiers.values = [0.8, 0.1]
    invoke_entrypoint(
        "standalone", standalone_document, tmp_path / "aliases.yaml", monkeypatch
    )
    report = selection_summary(standalone_document)
    _assert_saved_results(report, recorded_classifiers.results)
    assert [
        result["validation_dataset"] for result in report["classifier_results"]
    ] == [original, str(alias)]
    assert report["selected_classifier"]["best_checkpoint"] == str(
        recorded_classifiers.results[1].best_checkpoint
    )


@pytest.mark.parametrize("validation", [True, False])
def test_different_class_subsets_cannot_share_a_validation_champion(
    standalone_document, recorded_classifiers, tmp_path, monkeypatch, validation
):
    """The evaluated class population is part of the objective, not just the directory path."""
    standalone_document["class_names"] = {
        "type": "sweep",
        "values": [["rat", "mouse"], ["mouse"]],
    }
    if validation:
        with pytest.raises(ValueError, match="class|population|compar"):
            invoke_entrypoint(
                "standalone",
                standalone_document,
                tmp_path / "classes.yaml",
                monkeypatch,
            )
        assert recorded_classifiers.configurations == []
    else:
        standalone_document.update(validation_dataset=None, validation_metrics=None)
        invoke_entrypoint(
            "standalone", standalone_document, tmp_path / "classes.yaml", monkeypatch
        )
        assert len(recorded_classifiers.results) == 2
        assert selection_summary(standalone_document)["selected_classifier"] is None


def test_reordered_class_declarations_remain_comparable(
    standalone_document, recorded_classifiers, tmp_path, monkeypatch
):
    """Class IDs are assigned alphabetically, so declaration order must not change identity."""
    standalone_document["class_names"] = {
        "type": "sweep",
        "values": [["rat", "mouse"], ["mouse", "rat"]],
    }
    recorded_classifiers.values = [0.8, 0.1]
    invoke_entrypoint(
        "standalone", standalone_document, tmp_path / "class-order.yaml", monkeypatch
    )
    report = selection_summary(standalone_document)
    _assert_saved_results(report, recorded_classifiers.results)
    assert report["selected_classifier"]["best_checkpoint"] == str(
        recorded_classifiers.results[1].best_checkpoint
    )


def test_standalone_classifier_champion_requires_the_same_validation_dataset(
    standalone_document, recorded_classifiers, tmp_path, monkeypatch
):
    """Validation-path sweeps cannot produce a meaningful shared champion."""
    other = tmp_path / "other-validation"
    shutil.copytree(standalone_document["validation_dataset"], other)
    standalone_document["validation_dataset"] = {
        "type": "sweep",
        "values": [standalone_document["validation_dataset"], str(other)],
    }
    with pytest.raises(ValueError, match="validation|compar"):
        invoke_entrypoint(
            "standalone", standalone_document, tmp_path / "config.yaml", monkeypatch
        )
    assert recorded_classifiers.configurations == []


@SHARED_RULE
@pytest.mark.parametrize(
    "override",
    [
        {"tracked_metric": "unconfigured_metric"},
        {"tracked_mode": "max"},
        {"validation_dataset": None},
    ],
)
@pytest.mark.parametrize("reported_pattern", ["second", "all", "singleton"])
def test_incompatible_reported_classifier_results_are_not_ranked(
    sweep_case, recorded_classifiers, override, reported_pattern
):
    """Check results against configuration, not merely against other candidates' metadata."""
    if reported_pattern == "singleton":
        sweep_case.stage["train_kwargs"]["lr0"] = 0.001
        recorded_classifiers.overrides[0] = override
    elif reported_pattern == "all":
        sweep_case.stage["train_kwargs"]["lr0"] = {
            "type": "sweep",
            "values": [0.001, 0.002],
        }
        recorded_classifiers.overrides.update({0: override, 1: override})
    elif reported_pattern == "second":
        sweep_case.stage["train_kwargs"]["lr0"] = {
            "type": "sweep",
            "values": [0.001, 0.002],
        }
        recorded_classifiers.overrides[1] = override
    else:
        raise AssertionError(f"Unknown result pattern: {reported_pattern}")
    with pytest.raises(ValueError, match="metric|mode|direction|validation|compar"):
        sweep_case.run()
    path = sweep_case.root / "selection.yaml"
    if path.exists():
        assert yaml.safe_load(path.read_text()).get("selected_classifier") is None


@SHARED_RULE
@pytest.mark.parametrize("fail_at", [0, 2])  # first and last run
def test_classifier_failure_stops_the_sweep_without_selecting_a_partial_champion(
    sweep_case, recorded_classifiers, fail_at
):
    """Retain completed artifacts, propagate failure, and never rank a partial sweep."""
    case = sweep_case
    case.stage["train_kwargs"]["lr0"] = {
        "type": "sweep",
        "values": [0.001, 0.002, 0.003],
    }
    recorded_classifiers.fail_at = fail_at
    with pytest.raises(RuntimeError, match="classifier sweep failure"):
        case.run()
    assert len(recorded_classifiers.configurations) == fail_at + 1
    assert all(
        result.best_checkpoint.is_file() for result in recorded_classifiers.results
    )
    path = case.root / "selection.yaml"
    if path.exists():
        assert yaml.safe_load(path.read_text()).get("selected_classifier") is None


@pytest.mark.parametrize("location", ["classifier_run", "summary", "root_config"])
def test_sweep_artifacts_cannot_overwrite_any_input_checkpoint(
    sweep_case, recorded_classifiers, recorded_stages, classifier_assets, location
):
    """Protect inputs against every generated artifact, not just the current variant's files."""
    case = sweep_case
    if location == "classifier_run":
        protected = case.classifier_root / "run_002" / "weights" / "best.pt"
    elif location == "summary":
        protected = case.root / "selection.yaml"
    elif location == "root_config":
        protected = case.root / "config.yaml"
    else:
        raise AssertionError(f"Unknown collision location: {location}")
    protected.parent.mkdir(parents=True, exist_ok=True)
    original = classifier_assets.ultralytics.read_bytes()
    protected.write_bytes(original)
    head = case.stage["initialization"]["head"]
    head.update(source="yolo_checkpoint", checkpoint=str(protected))
    case.stage["train_kwargs"]["lr0"] = {"type": "sweep", "values": [0.001, 0.002]}
    if location == "classifier_run":
        # Neither variant collides with its own outputs: run one reads run two's
        # future best.pt, while run two reads an external donor. Per-run guards
        # alone cannot detect the required cross-variant protection.
        head["checkpoint"] = {
            "type": "coupled-sweep",
            "target": (["classifier"] if case.entrypoint == "two_stage" else [])
            + ["train_kwargs", "lr0"],
            "values": [str(protected), str(classifier_assets.ultralytics)],
        }
    with pytest.raises(ValueError, match="collid|checkpoint|source|artifact"):
        case.run()
    assert recorded_classifiers.configurations == []
    assert recorded_stages.supcon == []
    assert protected.read_bytes() == original


def test_mismatched_coupled_sweeps_fail_before_training(
    sweep_case, recorded_classifiers, recorded_stages
):
    """A malformed coupling must not truncate runs through zip-like behavior."""
    case = sweep_case
    case.stage["train_kwargs"]["lr0"] = {"type": "sweep", "values": [0.001, 0.002]}
    case.stage["batch_size"] = {
        "type": "coupled-sweep",
        "target": (["classifier"] if case.entrypoint == "two_stage" else [])
        + ["train_kwargs", "lr0"],
        "values": [2],
    }
    with pytest.raises(ValueError, match="length|coupl"):
        case.run()
    assert recorded_classifiers.configurations == []
    assert recorded_stages.supcon == []


def test_failed_rerun_invalidates_an_old_classifier_champion_summary(
    sweep_case, recorded_classifiers, recorded_stages
):
    """An earlier successful invocation must not masquerade as the failed invocation's winner."""
    case = sweep_case
    case.stage["train_kwargs"]["lr0"] = {"type": "sweep", "values": [0.001, 0.002]}
    case.run()
    assert selection_summary(case.document)["selected_classifier"] is not None
    first = recorded_classifiers.results[0]
    completed_bytes = first.best_checkpoint.read_bytes()
    recorded_classifiers.configurations.clear()
    recorded_classifiers.results.clear()
    recorded_classifiers.fail_at = 0
    # Replay the same SupCon sweep rather than extending the fixture's sequence.
    recorded_stages.supcon.clear()
    recorded_stages.results.clear()
    recorded_stages.events.clear()
    with pytest.raises(RuntimeError, match="classifier sweep failure"):
        case.run()
    summary_path = case.root / "selection.yaml"
    if summary_path.exists():
        assert (
            yaml.safe_load(summary_path.read_text()).get("selected_classifier") is None
        )
    assert first.best_checkpoint.read_bytes() == completed_bytes


def test_standalone_training_only_sweeps_do_not_declare_a_validation_champion(
    standalone_document, recorded_classifiers, tmp_path, monkeypatch
):
    """Training without validation remains supported without promoting training-loss winners."""
    standalone_document.update(validation_dataset=None, validation_metrics=None)
    standalone_document["train_kwargs"]["lr0"] = {
        "type": "sweep",
        "values": [0.001, 0.002],
    }
    invoke_entrypoint(
        "standalone", standalone_document, tmp_path / "config.yaml", monkeypatch
    )
    summary = selection_summary(standalone_document)
    assert len(recorded_classifiers.results) == 2
    _assert_saved_results(summary, recorded_classifiers.results)
    assert summary["selected_classifier"] is None


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    learning_rates=st.lists(
        st.floats(
            min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=3,
        unique=True,
    ),
    batch_sizes=st.lists(
        st.integers(min_value=1, max_value=8), min_size=1, max_size=3, unique=True
    ),
)
def test_standalone_cartesian_expansion_preserves_every_configuration(
    standalone_document,
    recorded_classifiers,
    tmp_path,
    monkeypatch,
    learning_rates,
    batch_sizes,
):
    """Arbitrary finite valid sweep values must not be dropped, merged, or reordered."""
    document = deepcopy(standalone_document)
    document["output_directory"] = str(tmp_path / "property-sweep")
    document["train_kwargs"]["lr0"] = {"type": "sweep", "values": learning_rates}
    document["batch_size"] = {"type": "sweep", "values": batch_sizes}
    recorded_classifiers.configurations.clear()
    recorded_classifiers.results.clear()
    invoke_entrypoint("standalone", document, tmp_path / "property.yaml", monkeypatch)
    # safe_dump sorts keys: batch_size is encountered before train_kwargs.lr0.
    expected = list(product(batch_sizes, learning_rates))
    assert [
        (c.batch_size, c.train_kwargs["lr0"])
        for c in recorded_classifiers.configurations
    ] == expected
    assert len(
        {c.output_directory for c in recorded_classifiers.configurations}
    ) == len(expected)
