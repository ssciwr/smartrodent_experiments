"""Optuna studies driving the SupCon and classifier trainers, standalone and in two-stage.

Most tests replace only ``run_with_result`` with a recorder; YAML loading, search-space
resolution, stage configuration validation, Optuna, and reports stay real. A few
tests run the real CPU trainers to prove the per-epoch pruning hook end to end.
"""

import hashlib
import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
import yaml
from study_test_support import STUDY_NAME, study_report, tuning

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.experiment_results import ExperimentResult
from smartrodent.supcon_training import SupConExperiment
from smartrodent.training_studies import ClassifierStudy, SupConStudy
from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment

LR_TAG = "!suggest-float {low: 1.0e-4, high: 1.0e-2, log: true}"


def _tagged_yaml(document, tags):
    """Dump a plain document, then splice tags in place of unique string placeholders."""
    text = yaml.safe_dump(document, sort_keys=False)
    for placeholder, tag in tags.items():
        assert text.count(placeholder) == 1, placeholder
        text = text.replace(placeholder, tag)
    return text


def _write(path: Path, document, tags) -> Path:
    """Write a tagged configuration file next to the test's other inputs."""
    path.write_text(_tagged_yaml(document, tags), encoding="utf-8")
    return path


def _encoder_study_root(root: Path, checkpoint: Path) -> Path:
    """Return the pipeline classifier study directory of one selected encoder."""
    return Path(root) / "classifier" / _encoder_key(checkpoint)


def _encoder_key(checkpoint: Path) -> str:
    """Return the short checksum that names a selected encoder's classifier study."""
    return hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12]


def _peak_score(rate: float) -> float:
    """A smooth, deterministic objective peaking at a learning rate of 1e-3."""
    return -abs(math.log10(rate) + 3.0)


@pytest.fixture
def recorded_trials(monkeypatch):
    """Replace only trainer execution; report a score derived from each sampled configuration."""
    observed = SimpleNamespace(supcon=[], classifier=[], events=[])

    def _artifacts(directory: Path, weights: bool):
        directory = directory / "weights" if weights else directory
        directory.mkdir(parents=True, exist_ok=True)
        best, last = directory / "best.pt", directory / "last.pt"
        # Bytes depend on the run directory: different winners are different encoders.
        torch.save({"epoch": 1, "run": str(directory)}, best)
        torch.save({"epoch": 2, "run": str(directory)}, last)
        return best, last

    def run_supcon(self, *, parent_run_id=None, epoch_callback=None):
        config = self.configuration
        observed.supcon.append(
            SimpleNamespace(
                configuration=config,
                parent_run_id=parent_run_id,
                epoch_callback=epoch_callback,
            )
        )
        observed.events.append("supcon")
        value = _peak_score(config.optimizer["lr"])
        for epoch in (1, 2):
            if epoch_callback is not None:
                epoch_callback(epoch, value)
        best, last = _artifacts(Path(config.output_directory), weights=False)
        return ExperimentResult(
            output_directory=config.output_directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=1,
            best_value=value,
            tracked_metric=config.effective_tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
            mlflow_run_id=None,
        )

    def run_classifier(
        self, *, parent_run_id=None, defer_promotion=False, epoch_callback=None
    ):
        config = self.configuration
        observed.classifier.append(
            SimpleNamespace(
                configuration=config,
                parent_run_id=parent_run_id,
                defer_promotion=defer_promotion,
                epoch_callback=epoch_callback,
            )
        )
        observed.events.append("classifier")
        value = _peak_score(config.train_kwargs["lr0"])
        for epoch in (1, 2):
            if epoch_callback is not None:
                epoch_callback(epoch, value)
        best, last = _artifacts(Path(config.output_directory), weights=True)
        return ExperimentResult(
            output_directory=config.output_directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=1,
            best_value=value,
            tracked_metric=config.tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
        )

    monkeypatch.setattr(SupConExperiment, "run_with_result", run_supcon)
    monkeypatch.setattr(ClassifierExperiment, "run_with_result", run_classifier)
    return observed


# --------------------------------------------------------------------------------------
# Standalone SupCon studies
# --------------------------------------------------------------------------------------


@pytest.fixture
def supcon_document(validation_configuration):
    """A standalone SupCon document ranked by a validation score, with an lr placeholder."""
    document = validation_configuration.to_dict()
    document.update(
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        optimizer={"lr": "__LR__"},
        tuning=tuning(),
    )
    return document


def test_supcon_study_trains_each_sampled_configuration_in_its_own_trial(
    supcon_document, recorded_trials, tmp_path
):
    """Only searched values vary; every trial is a fully validated standalone configuration."""
    path = _write(tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG})
    result = SupConStudy.from_yaml(path).run()
    root = Path(supcon_document["output_directory"])
    calls = recorded_trials.supcon
    assert [call.configuration.output_directory for call in calls] == [
        root / f"trial_{number:04d}" for number in range(3)
    ]
    for call in calls:
        assert 1.0e-4 <= call.configuration.optimizer["lr"] <= 1.0e-2
        assert callable(call.epoch_callback)
        expected = {**supcon_document, "optimizer": call.configuration.optimizer}
        expected.pop("tuning")
        expected["output_directory"] = str(call.configuration.output_directory)
        assert call.configuration.to_dict() == expected
    assert result.selected.best_value == max(
        _peak_score(c.configuration.optimizer["lr"]) for c in calls
    )
    assert study_report(root)["selected_trial"]["result"]["best_checkpoint"] == str(
        result.selected.best_checkpoint
    )


# The two SupCon script-dispatch tests below and
# test_classifier_script_reports_the_study_winner exercise scripts/train_supcon.py and
# scripts/train_classifier.py but stay here: they need this module's study fixtures,
# which would otherwise have to become suite-wide plugins.
def test_supcon_script_runs_a_study_when_the_configuration_has_a_tuning_section(
    supcon_document, recorded_trials, tmp_path, monkeypatch, capsys
):
    """The existing entry point dispatches on the configuration; no separate tuning script."""
    from scripts import train_supcon

    path = _write(tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG})
    monkeypatch.setattr(sys, "argv", ["train_supcon.py", str(path)])
    train_supcon.main()
    report = study_report(supcon_document["output_directory"])
    assert len(recorded_trials.supcon) == 3
    assert (
        f"Selected checkpoint: {report['selected_trial']['result']['best_checkpoint']}"
        in capsys.readouterr().out
    )


def test_supcon_script_keeps_grid_sweeps_for_configurations_without_tuning(
    supcon_document, recorded_trials, tmp_path, monkeypatch
):
    """Configurations without a tuning section keep their existing sweep behavior."""
    from scripts import train_supcon

    supcon_document.pop("tuning")
    path = _write(
        tmp_path / "supcon.yaml",
        supcon_document,
        {"__LR__": "!sweep {values: [0.001, 0.002]}"},
    )
    monkeypatch.setattr(sys, "argv", ["train_supcon.py", str(path)])
    train_supcon.main()
    assert [call.configuration.optimizer["lr"] for call in recorded_trials.supcon] == [
        0.001,
        0.002,
    ]
    assert not (Path(supcon_document["output_directory"]) / "study.yaml").exists()


@pytest.mark.parametrize(
    "key,placeholder_at,tag",
    [
        ("temperature", ("temperature",), "!suggest-float {low: 0.05, high: 0.2}"),
        (
            "class_names",
            ("class_names", 0),
            "!suggest-categorical {choices: [rat, vole]}",
        ),
        (
            "train_dataset",
            ("train_dataset",),
            "!suggest-categorical {choices: [@TMP@/a, @TMP@/b]}",
        ),
    ],
)
def test_supcon_study_rejects_searching_comparability_settings(
    supcon_document, recorded_trials, tmp_path, key, placeholder_at, tag
):
    """Fixed objective, data, outputs and temperature keep trials comparable (as in sweeps)."""
    node = supcon_document
    for part in placeholder_at[:-1]:
        node = node[part]
    node[placeholder_at[-1]] = "__P__"
    tag = tag.replace("@TMP@", str(tmp_path))
    path = _write(
        tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG, "__P__": tag}
    )
    with pytest.raises(ValueError, match=key):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []


@pytest.mark.parametrize(
    "placeholder_at,tag",
    [
        (("samples_per_class",), "!suggest-int {low: 0, high: 2}"),  # invalid low end
        (("dropout",), "!suggest-float {low: 0.0, high: 1.5}"),  # invalid high end
        (
            ("classes_per_batch",),
            "!suggest-categorical {choices: [2, 1]}",
        ),  # invalid later choice
    ],
    ids=["samples_per_class", "dropout", "classes_per_batch"],
)
def test_supcon_study_validates_every_bound_and_choice_before_any_trial(
    supcon_document, recorded_trials, tmp_path, placeholder_at, tag
):
    """A value no trial could ever use is a configuration error, found before training."""
    supcon_document[placeholder_at[0]] = "__P__"
    path = _write(
        tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG, "__P__": tag}
    )
    with pytest.raises(ValueError, match=placeholder_at[0]):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []
    assert not Path(supcon_document["output_directory"]).exists()


def test_supcon_invalid_sampled_combinations_fail_only_their_trial(
    supcon_document, recorded_trials, tmp_path
):
    """Each value is valid alone (checked around the first choice); one combination is not.

    batch = classes_per_batch(2) * samples_per_class, which must fit samples_per_epoch.
    """
    supcon_document.update(
        samples_per_class="__M__",
        samples_per_epoch="__E__",
        optimizer={"lr": 0.001},
        tuning=tuning(
            n_trials=10,
            sampler={"name": "BruteForceSampler", "parameters": {"seed": 0}},
        ),
    )
    path = _write(
        tmp_path / "supcon.yaml",
        supcon_document,
        {
            "__M__": "!suggest-categorical {choices: [1, 3]}",
            "__E__": "!suggest-categorical {choices: [8, 4]}",
        },
    )
    result = SupConStudy.from_yaml(path).run()
    trials = study_report(supcon_document["output_directory"])["trials"]
    states = {
        (t["params"]["samples_per_class"], t["params"]["samples_per_epoch"]): t["state"]
        for t in trials
    }
    assert states == {
        (1, 8): "COMPLETE",
        (3, 8): "COMPLETE",
        (1, 4): "COMPLETE",
        (3, 4): "FAIL",
    }
    assert len(recorded_trials.supcon) == len(result.results) == 3


def test_supcon_study_requires_validation_data_for_ranking(
    supcon_document, recorded_trials, tmp_path
):
    """Like sweep selection, a study cannot rank training-only runs."""
    # A valid training-only configuration, so only the study-specific rule can reject it.
    supcon_document.update(
        validation_dataset=None,
        validation_metrics=None,
        tracked_metric=None,
        tracked_mode=None,
    )
    path = _write(tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG})
    with pytest.raises(ValueError, match=r"(?i)study.*validation|validation.*study"):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_a_tuning_section_without_tags_is_rejected(
    supcon_document, recorded_trials, tmp_path
):
    """A study over an empty space would train identical trials."""
    supcon_document["optimizer"] = {"lr": 0.001}
    path = _write(tmp_path / "supcon.yaml", supcon_document, {})
    with pytest.raises(ValueError, match="empty"):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_tracked_supcon_study_parents_every_trial_to_one_study_run(
    supcon_document, recorded_trials, tracking_store, tmp_path
):
    """A standalone study owns one MLflow parent run holding the report and best score."""
    store = tracking_store
    supcon_document["mlflow"] = deepcopy(store.configuration)
    path = _write(tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG})
    result = SupConStudy.from_yaml(path).run()
    (parent,) = store.client.search_runs(
        [store.experiment_id]
    )  # trials are recorded fakes
    assert "mlflow.parentRunId" not in parent.data.tags
    assert [call.parent_run_id for call in recorded_trials.supcon] == [
        parent.info.run_id
    ] * 3
    assert "study.yaml" in {
        item.path for item in store.client.list_artifacts(parent.info.run_id)
    }
    assert (
        parent.data.metrics["best_validation_balanced_accuracy"]
        == result.selected.best_value
    )


def test_supcon_study_rejects_searching_tracking_settings(
    supcon_document, recorded_trials, tracking_store, tmp_path
):
    """Tracking identity is shared by all trials and cannot be searched."""
    supcon_document["mlflow"] = {
        **tracking_store.configuration,
        "experiment_name": "__P__",
    }
    path = _write(
        tmp_path / "supcon.yaml",
        supcon_document,
        {"__LR__": LR_TAG, "__P__": "!suggest-categorical {choices: [one, two]}"},
    )
    with pytest.raises(ValueError, match="mlflow"):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_real_supcon_trials_write_complete_run_artifacts(
    validation_configuration, tmp_path
):
    """Real CPU training: every completed trial is a normal SupCon run with its concrete config."""
    torch.set_num_threads(2)
    document = validation_configuration.to_dict()
    document.update(
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        optimizer={"lr": "__LR__"},
        tuning=tuning(n_trials=2),
    )
    path = _write(tmp_path / "supcon.yaml", document, {"__LR__": LR_TAG})
    result = SupConStudy.from_yaml(path).run()
    report = study_report(document["output_directory"])
    assert [trial["state"] for trial in report["trials"]] == ["COMPLETE", "COMPLETE"]
    for trial in report["trials"]:
        directory = Path(trial["result"]["output_directory"])
        assert {item.name for item in directory.glob("*.pt")} == {"best.pt", "last.pt"}
        saved = yaml.safe_load((directory / "config.yaml").read_text())
        assert saved["optimizer"]["lr"] == trial["params"]["optimizer.lr"]
    assert result.selected.best_checkpoint.is_file()


# --------------------------------------------------------------------------------------
# Standalone classifier studies
# --------------------------------------------------------------------------------------


@pytest.fixture
def classifier_document(classifier_configuration):
    """A standalone classifier document ranked by a validation score, with tag placeholders."""
    document = classifier_configuration.to_dict()
    document.update(
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        tuning=tuning(),
    )
    document["train_kwargs"]["lr0"] = "__LR0__"
    document["initialization"]["head"]["replace"] = "__HEAD__"
    document["encoder_batchnorm"] = "__BN__"
    return document


CLASSIFIER_TAGS = {
    "__LR0__": LR_TAG,
    "__HEAD__": "!suggest-categorical {choices: [linear, full]}",
    "__BN__": "!coupled-sweep {target: initialization.head.replace, values: [eval, adapt]}",
}


def test_classifier_study_trains_each_sampled_configuration_in_its_own_trial(
    classifier_document, recorded_trials, tmp_path
):
    """Sampled native options and coupled settings reach each standalone classifier run."""
    path = _write(tmp_path / "classifier.yaml", classifier_document, CLASSIFIER_TAGS)
    result = ClassifierStudy.from_yaml(path).run()
    root = Path(classifier_document["output_directory"])
    calls = recorded_trials.classifier
    assert [call.configuration.output_directory for call in calls] == [
        root / f"trial_{number:04d}" for number in range(3)
    ]
    for call in calls:
        config = call.configuration
        assert 1.0e-4 <= config.train_kwargs["lr0"] <= 1.0e-2
        replace_mode = config.initialization["head"]["replace"]
        assert (
            config.encoder_batchnorm
            == {"linear": "eval", "full": "adapt"}[replace_mode]
        )
        assert callable(call.epoch_callback)
        assert call.defer_promotion is False  # no registry configured
    assert result.selected.best_value == max(
        _peak_score(c.configuration.train_kwargs["lr0"]) for c in calls
    )


# Script-dispatch test kept here for its study fixtures (see the SupCon section).
def test_classifier_script_reports_the_study_winner(
    classifier_document, recorded_trials, tmp_path, monkeypatch, capsys
):
    """The existing classifier entry point dispatches to a study on a tuning section."""
    from scripts import train_classifier

    path = _write(tmp_path / "classifier.yaml", classifier_document, CLASSIFIER_TAGS)
    monkeypatch.setattr(sys, "argv", ["train_classifier.py", str(path)])
    train_classifier.main()
    report = study_report(classifier_document["output_directory"])
    expected = report["selected_trial"]["result"]["best_checkpoint"]
    assert f"Selected classifier checkpoint: {expected}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "placeholder_at,tag,match",
    [
        (
            ("tracked_metric",),
            "!suggest-categorical {choices: [balanced_accuracy, ce_loss_mean]}",
            "tracked_metric",
        ),
        (
            ("train_dataset",),
            "!suggest-categorical {choices: [@TMP@/a, @TMP@/b]}",
            "train_dataset",
        ),
        (
            ("validation_dataset",),
            "!suggest-categorical {choices: [@TMP@/a, @TMP@/b]}",
            "validation_dataset",
        ),
        (
            ("class_names", 0),
            "!suggest-categorical {choices: [rat, vole]}",
            "class_names",
        ),
    ],
)
def test_classifier_study_rejects_invalid_spaces_before_any_trial(
    classifier_document,
    recorded_trials,
    tmp_path,
    placeholder_at,
    tag,
    match,
):
    """Comparability settings stay fixed, and a choice no trial could use is rejected."""
    node = classifier_document
    for part in placeholder_at[:-1]:
        node = node[part]
    node[placeholder_at[-1]] = "__P__"
    tag = tag.replace("@TMP@", str(tmp_path))
    path = _write(
        tmp_path / "classifier.yaml",
        classifier_document,
        {**CLASSIFIER_TAGS, "__P__": tag},
    )
    with pytest.raises(ValueError, match=match):
        ClassifierStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_classifier_study_prevalidates_coupled_values_with_their_choice(
    classifier_document, recorded_trials, tmp_path
):
    """A coupled value is checked together with the categorical choice it belongs to."""
    tags = {
        **CLASSIFIER_TAGS,
        "__BN__": "!coupled-sweep {target: initialization.head.replace, values: [eval, bogus]}",
    }
    path = _write(tmp_path / "classifier.yaml", classifier_document, tags)
    with pytest.raises(ValueError, match="encoder_batchnorm"):
        ClassifierStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_real_classifier_pruning_stops_native_training_of_weak_trials(
    classifier_configuration, controlled_scores, tmp_path
):
    """The pruner, fed by the native epoch hook, stops a weak trial after its first epoch."""
    document = classifier_configuration.to_dict()
    document.update(
        epochs=3,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        tuning=tuning(
            n_trials=5,
            sampler={"name": "BruteForceSampler", "parameters": {"seed": 0}},
            pruner={"name": "ThresholdPruner", "parameters": {"lower": 0.5}},
        ),
    )
    document["train_kwargs"]["lr0"] = "__LR0__"
    controlled_scores.score_sets = {0.002: [0.2, 0.3, 0.4], 0.004: [0.9, 0.8, 0.7]}
    path = _write(
        tmp_path / "classifier.yaml",
        document,
        {"__LR0__": "!suggest-categorical {choices: [0.002, 0.004]}"},
    )
    result = ClassifierStudy.from_yaml(path).run()
    trials = {
        t["params"]["train_kwargs.lr0"]: t
        for t in study_report(document["output_directory"])["trials"]
    }
    weak, strong = trials[0.002], trials[0.004]
    assert weak["state"] == "PRUNED" and strong["state"] == "COMPLETE"
    weak_directory = Path(document["output_directory"]) / f"trial_{weak['number']:04d}"
    assert controlled_scores.epochs[str(weak_directory)] == 1
    assert not (weak_directory / "weights/best.model_card.yaml").exists()
    assert result.selected.best_value == 0.9 and result.best_params == {
        "train_kwargs.lr0": 0.004
    }


def test_tracked_classifier_study_nests_trials_and_promotes_only_the_winner(
    classifier_configuration, controlled_scores, tracking_store, tmp_path
):
    """One study parent run, one child per trial; every completed trial registers, the best is champion."""
    store = tracking_store
    document = classifier_configuration.to_dict()
    document.update(
        epochs=2,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        mlflow={**store.configuration, "registered_model_name": store.registry_name},
        tuning=tuning(
            n_trials=5, sampler={"name": "BruteForceSampler", "parameters": {"seed": 0}}
        ),
    )
    document["train_kwargs"]["lr0"] = "__LR0__"
    controlled_scores.score_sets = {0.002: [0.2, 0.3], 0.004: [0.9, 0.8]}
    path = _write(
        tmp_path / "classifier.yaml",
        document,
        {"__LR0__": "!suggest-categorical {choices: [0.002, 0.004]}"},
    )
    result = ClassifierStudy.from_yaml(path).run()
    runs = store.client.search_runs([store.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
    assert len(parents) == 1 and len(children) == 2
    parent = parents[0]
    assert all(
        run.data.tags["mlflow.parentRunId"] == parent.info.run_id for run in children
    )
    assert sorted(run.info.run_name for run in children) == ["trial_0000", "trial_0001"]
    assert all(run.info.status == "FINISHED" for run in runs)
    assert "study.yaml" in {
        item.path for item in store.client.list_artifacts(parent.info.run_id)
    }
    assert parent.data.metrics["best_validation_balanced_accuracy"] == 0.9
    versions = store.client.search_model_versions(f"name='{store.registry_name}'")
    assert len(versions) == 2  # decision 3: every completed trial registers for now
    champion = store.client.get_model_version_by_alias(
        store.registry_name, store.configuration["champion_alias"]
    )
    assert champion.run_id == result.selected.mlflow_run_id
    assert result.selected.registered_model_version == str(champion.version)


# --------------------------------------------------------------------------------------
# Two-stage pipeline: each stage is a grid sweep or a study
# --------------------------------------------------------------------------------------


def _pipeline(document, tags, tmp_path, *, studies):
    """Write and load a tagged two-stage configuration with per-stage study settings."""
    document = deepcopy(document)
    document["tuning"] = studies
    path = _write(tmp_path / "pipeline.yaml", document, tags)
    return TwoStageConfiguration.from_yaml(path)


@pytest.fixture
def study_pipeline(pipeline_document):
    """Placeholders for one tag per stage; sweeps stay unless a test removes them."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"] = "__LR__"
    document["classifier"]["train_kwargs"]["lr0"] = "__LR0__"
    # Rank classifiers by a validation score, so "best" means the highest reported value.
    document["classifier"].update(
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    return document


def test_supcon_study_selects_the_encoder_for_a_classifier_grid_sweep(
    study_pipeline, recorded_trials, tmp_path
):
    """The SupCon stage runs as a study; its winner feeds every classifier sweep run."""
    config = _pipeline(
        study_pipeline,
        {"__LR__": LR_TAG, "__LR0__": "!sweep {values: [0.001, 0.004]}"},
        tmp_path,
        studies={"supcon": tuning()},
    )
    result = TwoStageExperiment(config).run()
    root = Path(study_pipeline["output_directory"])
    assert recorded_trials.events == ["supcon"] * 3 + ["classifier"] * 2
    assert [call.configuration.output_directory for call in recorded_trials.supcon] == [
        root / "supcon" / f"trial_{number:04d}" for number in range(3)
    ]
    assert len(result.supcon_results) == 3
    assert result.selected_supcon.best_value == max(
        item.best_value for item in result.supcon_results
    )
    winner = result.selected_supcon.best_checkpoint.resolve()
    for call in recorded_trials.classifier:
        assert call.configuration.initialization["encoder"] == {
            "source": "supcon",
            "checkpoint": str(winner),
        }
    assert study_report(root / "supcon")["selected_trial"]["result"][
        "best_checkpoint"
    ] == str(result.selected_supcon.best_checkpoint)
    selection = yaml.safe_load((root / "selection.yaml").read_text())
    assert selection["selected_supcon"]["best_checkpoint"] == str(
        result.selected_supcon.best_checkpoint
    )


def test_classifier_study_runs_on_the_encoder_selected_by_a_supcon_sweep(
    study_pipeline, recorded_trials, tmp_path
):
    """The classifier stage runs as a study, every trial on the one selected encoder."""
    config = _pipeline(
        study_pipeline,
        {"__LR__": "!sweep {values: [0.001, 0.002]}", "__LR0__": LR_TAG},
        tmp_path,
        studies={"classifier": tuning(n_trials=4)},
    )
    result = TwoStageExperiment(config).run()
    root = Path(study_pipeline["output_directory"])
    assert recorded_trials.events == ["supcon"] * 2 + ["classifier"] * 4
    winner = result.selected_supcon.best_checkpoint.resolve()
    study_root = _encoder_study_root(root, winner)
    assert [
        call.configuration.output_directory for call in recorded_trials.classifier
    ] == [study_root / f"trial_{number:04d}" for number in range(4)]
    for call in recorded_trials.classifier:
        assert call.configuration.initialization["encoder"] == {
            "source": "supcon",
            "checkpoint": str(winner),
        }
        assert callable(call.epoch_callback)
    assert len(result.classifier_results) == 4
    assert result.selected_classifier.best_value == max(
        item.best_value for item in result.classifier_results
    )
    assert study_report(study_root)["study_name"] == (
        f"{STUDY_NAME}-{_encoder_key(winner)}"
    )


def test_both_stages_can_run_as_studies(study_pipeline, recorded_trials, tmp_path):
    """Stages are tuned sequentially and independently: SupCon first, then the classifier."""
    config = _pipeline(
        study_pipeline,
        {"__LR__": LR_TAG, "__LR0__": LR_TAG},
        tmp_path,
        studies={
            "supcon": tuning(n_trials=2),
            "classifier": tuning(n_trials=3, study_name="ce"),
        },
    )
    result = TwoStageExperiment(config).run()
    assert recorded_trials.events == ["supcon"] * 2 + ["classifier"] * 3
    assert len(result.supcon_results) == 2 and len(result.classifier_results) == 3


def test_tracked_pipeline_studies_parent_every_trial_to_the_pipeline_run(
    study_pipeline, recorded_trials, tracking_store, tmp_path
):
    """Trials of both stages are children of the one pipeline run, like sweep candidates."""
    study_pipeline["mlflow"] = deepcopy(tracking_store.configuration)
    config = _pipeline(
        study_pipeline,
        {"__LR__": LR_TAG, "__LR0__": LR_TAG},
        tmp_path,
        studies={
            "supcon": tuning(n_trials=2),
            "classifier": tuning(n_trials=2, study_name="ce"),
        },
    )
    result = TwoStageExperiment(config).run()
    calls = [*recorded_trials.supcon, *recorded_trials.classifier]
    assert result.mlflow_run_id is not None
    assert [call.parent_run_id for call in calls] == [result.mlflow_run_id] * 4


def test_classifier_freeze_is_checked_against_every_supcon_model_choice(
    study_pipeline, recorded_trials, tmp_path
):
    """The winning encoder is unknown before the SupCon study, so freeze must fit every choice."""
    deeper = tmp_path / "deeper-cls.yaml"
    deeper.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 1]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    shallow = study_pipeline["supcon"]["model_name"]
    study_pipeline["supcon"]["model_name"] = "__MODEL__"
    study_pipeline["classifier"]["freeze"] = 3  # fits only the deeper network
    with pytest.raises(ValueError, match="freeze"):
        config = _pipeline(
            study_pipeline,
            {
                "__LR__": LR_TAG,
                "__LR0__": "0.004",
                "__MODEL__": f"!suggest-categorical {{choices: [{deeper}, {shallow}]}}",
            },
            tmp_path,
            studies={"supcon": tuning()},
        )
        TwoStageExperiment(config).run()
    assert recorded_trials.events == []
    assert not Path(study_pipeline["output_directory"]).exists()


@pytest.mark.parametrize(
    "tags,studies,match",
    [
        # search tags in a stage without study settings
        (
            {"__LR__": LR_TAG, "__LR0__": LR_TAG},
            {"supcon": tuning()},
            "tuning",
        ),
        # study settings for a stage without search tags
        (
            {"__LR__": LR_TAG, "__LR0__": "0.004"},
            {"supcon": tuning(), "classifier": tuning()},
            "empty",
        ),
        # a search bound the classifier stage can never use
        (
            {"__LR__": "0.001", "__LR0__": "!suggest-float {low: -1.0, high: 0.01}"},
            {"classifier": tuning()},
            "lr0",
        ),
    ],
)
def test_invalid_pipeline_studies_fail_before_any_stage_trains(
    study_pipeline, recorded_trials, tmp_path, tags, studies, match
):
    """The complete pipeline, both stages and their studies, is validated before training."""
    with pytest.raises(ValueError, match=match):
        TwoStageExperiment(
            _pipeline(study_pipeline, tags, tmp_path, studies=studies)
        ).run()
    assert recorded_trials.events == []
    assert not Path(study_pipeline["output_directory"]).exists()


@pytest.mark.parametrize(
    "stage,placeholder_at,tag,match",
    [
        (
            "supcon",
            ("temperature",),
            "!suggest-float {low: 0.05, high: 0.2}",
            "temperature",
        ),
        (
            "classifier",
            ("tracked_mode",),
            "!suggest-categorical {choices: [min, max]}",
            "tracked_mode",
        ),
        (
            "classifier",
            ("initialization", "encoder", "source"),
            "!suggest-categorical {choices: [selected_supcon, supcon]}",
            "initialization.encoder|selected_supcon",
        ),
    ],
)
def test_pipeline_studies_reject_searching_comparability_or_handoff_settings(
    study_pipeline, recorded_trials, tmp_path, stage, placeholder_at, tag, match
):
    """Per-stage protections match the standalone studies and the existing handoff contract."""
    node = study_pipeline[stage]
    for part in placeholder_at[:-1]:
        node = node[part]
    node[placeholder_at[-1]] = "__P__"
    with pytest.raises(ValueError, match=match):
        config = _pipeline(
            study_pipeline,
            {"__LR__": LR_TAG, "__LR0__": LR_TAG, "__P__": tag},
            tmp_path,
            studies={"supcon": tuning(), "classifier": tuning()},
        )
        TwoStageExperiment(config).run()
    assert recorded_trials.events == []


def test_invalid_base_settings_are_not_blamed_on_a_search_parameter(
    supcon_document, recorded_trials, tmp_path
):
    """A mistake in a fixed setting is reported as such, not as a bad sampled value."""
    supcon_document["epochz"] = 3
    path = _write(tmp_path / "supcon.yaml", supcon_document, {"__LR__": LR_TAG})
    with pytest.raises(ValueError, match="epochz") as error:
        SupConStudy.from_yaml(path).run()
    assert "Search parameter" not in str(error.value)
    assert recorded_trials.events == []


def test_real_classifier_freeze_combinations_that_do_not_fit_fail_only_their_trial(
    classifier_configuration, tmp_path
):
    """Freeze validity depends on the head mode and network depth, known only at trial time.

    Fixture network: 0=Conv, 1=Dropout, 2=Classify. freeze=3 also freezes the head's
    conv, which only a linear head replacement allows.
    """
    document = classifier_configuration.to_dict()
    document.update(
        epochs=1,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        freeze="__F__",
        tuning=tuning(
            n_trials=10,
            sampler={"name": "BruteForceSampler", "parameters": {"seed": 0}},
        ),
    )
    document["initialization"]["head"]["replace"] = "__HEAD__"
    path = _write(
        tmp_path / "classifier.yaml",
        document,
        {
            "__F__": "!suggest-int {low: 2, high: 3}",
            "__HEAD__": "!suggest-categorical {choices: [linear, full]}",
        },
    )
    result = ClassifierStudy.from_yaml(path).run()
    trials = study_report(document["output_directory"])["trials"]
    states = {
        (t["params"]["freeze"], t["params"]["initialization.head.replace"]): t
        for t in trials
    }
    assert {key: trial["state"] for key, trial in states.items()} == {
        (2, "linear"): "COMPLETE",
        (3, "linear"): "COMPLETE",
        (2, "full"): "COMPLETE",
        (3, "full"): "FAIL",
    }
    assert "freeze" in states[(3, "full")]["error"]
    assert len(result.results) == 3


def test_pipeline_study_parameters_use_full_document_paths_like_sweeps(
    study_pipeline, recorded_trials, tmp_path
):
    """Coupled targets and parameter names keep the pipeline's full paths in study mode."""
    study_pipeline["classifier"]["initialization"]["head"]["replace"] = "__HEAD__"
    study_pipeline["classifier"]["encoder_batchnorm"] = "__BN__"
    config = _pipeline(
        study_pipeline,
        {
            "__LR__": "0.001",
            "__LR0__": LR_TAG,
            "__HEAD__": "!suggest-categorical {choices: [linear, full]}",
            "__BN__": "!coupled-sweep {target: classifier.initialization.head.replace, values: [eval, adapt]}",
        },
        tmp_path,
        studies={"classifier": tuning()},
    )
    result = TwoStageExperiment(config).run()
    report = study_report(
        _encoder_study_root(
            study_pipeline["output_directory"], result.selected_supcon.best_checkpoint
        )
    )
    assert all(
        set(trial["params"])
        == {"classifier.train_kwargs.lr0", "classifier.initialization.head.replace"}
        for trial in report["trials"]
    )
    for call in recorded_trials.classifier:
        replace_mode = call.configuration.initialization["head"]["replace"]
        assert (
            call.configuration.encoder_batchnorm
            == {"linear": "eval", "full": "adapt"}[replace_mode]
        )


@pytest.mark.parametrize(
    "studies",
    [
        {},
        {"supcon": None, "classifier": None},
    ],
)
def test_a_pipeline_tuning_section_must_name_a_stage(
    study_pipeline, recorded_trials, tmp_path, studies
):
    """An empty tuning section is a mistake, not a silent grid sweep."""
    with pytest.raises(ValueError, match="tuning"):
        _pipeline(
            study_pipeline,
            {"__LR__": "0.001", "__LR0__": "0.004"},
            tmp_path,
            studies=studies,
        )
    assert recorded_trials.events == []


def test_an_invalid_reference_document_names_the_reference_values(
    supcon_document, recorded_trials, tmp_path
):
    """A cross-field error at the low end still says which values were used."""
    supcon_document.update(
        samples_per_class="__M__", samples_per_epoch=4
    )  # batch = 2 * M
    path = _write(
        tmp_path / "supcon.yaml",
        supcon_document,
        {"__LR__": LR_TAG, "__M__": "!suggest-int {low: 3, high: 4}"},
    )
    with pytest.raises(ValueError, match="samples_per_class.*3"):
        SupConStudy.from_yaml(path).run()
    assert recorded_trials.events == []


def test_resumed_classifier_studies_refuse_a_changed_encoder(
    classifier_document, classifier_assets, recorded_trials, tmp_path
):
    """Trials trained on different encoder bytes are not comparable."""
    classifier_document["tuning"] = tuning(
        n_trials=1, storage=f"sqlite:///{tmp_path / 'optuna.db'}", load_if_exists=True
    )
    path = _write(tmp_path / "classifier.yaml", classifier_document, CLASSIFIER_TAGS)
    ClassifierStudy.from_yaml(path).run()
    ClassifierStudy.from_yaml(path).run()  # same encoder: resumes
    assert len(recorded_trials.classifier) == 2
    with classifier_assets.supcon.open("ab") as stream:  # test-owned fixture file
        stream.write(b"retrained")
    with pytest.raises(ValueError, match="encoder checkpoint"):
        ClassifierStudy.from_yaml(path).run()
    assert len(recorded_trials.classifier) == 2


@pytest.mark.parametrize("conflict", ["foreign_report", "existing_study"])
def test_supcon_study_conflicts_are_found_before_any_training(
    study_pipeline, recorded_trials, tmp_path, conflict
):
    """Storage and output conflicts are reported before the pipeline trains anything."""
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    root = Path(study_pipeline["output_directory"])
    if conflict == "foreign_report":
        (root / "supcon").mkdir(parents=True)
        (root / "supcon" / "study.yaml").write_text(
            yaml.safe_dump({"study_name": "other"})
        )
        match = "study_name"
    elif conflict == "existing_study":
        optuna.create_study(study_name="sc", storage=storage, direction="maximize")
        match = "load_if_exists"
    else:
        raise AssertionError(conflict)
    config = _pipeline(
        study_pipeline,
        {"__LR__": LR_TAG, "__LR0__": "0.004"},
        tmp_path,
        studies={"supcon": tuning(study_name="sc", storage=storage)},
    )
    with pytest.raises(ValueError, match=match):
        TwoStageExperiment(config).run()
    assert recorded_trials.events == []


# --- Decision (b): one classifier study per selected encoder ---


def test_pipeline_classifier_studies_are_kept_per_selected_encoder(
    study_pipeline, recorded_trials, tmp_path
):
    """An unchanged SupCon winner resumes its classifier study; a new winner starts its own."""
    root = Path(study_pipeline["output_directory"])
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    studies = {
        "classifier": tuning(
            study_name="ce", n_trials=2, storage=storage, load_if_exists=True
        )
    }

    def run(supcon_rates):
        config = _pipeline(
            study_pipeline,
            {"__LR__": f"!sweep {{values: {supcon_rates}}}", "__LR0__": LR_TAG},
            tmp_path,
            studies=studies,
        )
        return TwoStageExperiment(config).run()

    first = run([0.001, 0.004])  # run_001 wins: the score peaks at lr=1e-3
    first_root = _encoder_study_root(root, first.selected_supcon.best_checkpoint)
    second = run([0.001, 0.004])  # same winner, same bytes: resumes
    assert (
        _encoder_study_root(root, second.selected_supcon.best_checkpoint) == first_root
    )
    assert [t["number"] for t in study_report(first_root)["trials"]] == [0, 1, 2, 3]
    assert len(second.classifier_results) == 4
    third = run([0.004, 0.001])  # run_002 wins: a different encoder
    third_root = _encoder_study_root(root, third.selected_supcon.best_checkpoint)
    assert third_root != first_root
    assert [t["number"] for t in study_report(third_root)["trials"]] == [0, 1]
    assert len(study_report(first_root)["trials"]) == 4  # earlier study untouched
    for result in (first, third):
        assert all(
            item.output_directory.parent
            == _encoder_study_root(root, result.selected_supcon.best_checkpoint)
            for item in result.classifier_results
        )


def test_resumed_classifier_studies_check_every_searched_encoder(
    classifier_document, classifier_assets, recorded_trials, tmp_path
):
    """When the encoder itself is searched, a change to any choice's bytes is refused."""
    other = tmp_path / "other-encoder.pt"
    other.write_bytes(classifier_assets.supcon.read_bytes())
    classifier_document["initialization"]["encoder"]["checkpoint"] = "__ENC__"
    classifier_document["tuning"] = tuning(
        n_trials=1, storage=f"sqlite:///{tmp_path / 'optuna.db'}", load_if_exists=True
    )
    tags = {
        **CLASSIFIER_TAGS,
        "__ENC__": f"!suggest-categorical {{choices: [{classifier_assets.supcon}, {other}]}}",
    }
    path = _write(tmp_path / "classifier.yaml", classifier_document, tags)
    ClassifierStudy.from_yaml(path).run()
    with other.open("ab") as stream:  # the second, non-reference choice changes
        stream.write(b"retrained")
    with pytest.raises(ValueError, match="encoder checkpoints"):
        ClassifierStudy.from_yaml(path).run()
    assert len(recorded_trials.classifier) == 1
