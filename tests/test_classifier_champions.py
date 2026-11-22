"""Validation champions across classifier sweeps and successive MLflow invocations.

These tests retain real tiny CPU training, checkpoint loading, model-card writing,
and an isolated SQLite registry. Only epoch validation scores are controlled.
"""

from copy import deepcopy
import json
from pathlib import Path
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pytest
import torch
import yaml

from ultralytics import YOLO
from test_classifier_sweeps import _invoke, _summary, sweep_pipeline_document
from test_classifier_training import classifier_assets, classifier_configuration, controlled_scores, training_api
from test_training_tracking import tracking_store


@pytest.fixture(params=["standalone", "two_stage"])
def classifier_trial(
    request, classifier_configuration, sweep_pipeline_document, tracking_store, tmp_path, monkeypatch, controlled_scores
):
    """Configure two real CE runs whose first run/second epoch wins the sweep."""
    entrypoint = request.param
    if entrypoint == "standalone":
        document = classifier_configuration.to_dict()
        stage = document
    elif entrypoint == "two_stage":
        document = deepcopy(sweep_pipeline_document)
        # One SupCon candidate is sufficient here; separate orchestration tests
        # verify that multiple SupCon candidates are not multiplied by CE sweeps.
        document["supcon"].update(epochs=1, optimizer={"lr": 0.001})
        stage = document["classifier"]
    else:
        raise AssertionError(f"Unknown test entry point: {entrypoint}")
    document["output_directory"] = str(tmp_path / f"tracked-{entrypoint}")
    document["mlflow"] = {**tracking_store.configuration, "registered_model_name": tracking_store.registry_name}
    stage.update(
        epochs=3, tracked_metric="balanced_accuracy", tracked_mode="max",
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
        train_kwargs={**stage["train_kwargs"], "lr0": {"type": "sweep", "values": [0.001, 0.002]}},
    )
    observed = controlled_scores
    observed.document, observed.stage, observed.entrypoint = document, stage, entrypoint
    observed.score_sets = {0.001: [0.2, 0.9, 0.3], 0.002: [0.1, 0.8, 0.4]}
    observed.promotions = []

    original_set_alias = MlflowClient.set_registered_model_alias

    def set_alias(self, name, alias, version):
        """Record timing while retaining the real backend's alias semantics."""
        observed.promotions.append((str(version), len(observed.validations)))
        return original_set_alias(self, name, alias, version)

    def run():
        """Run the configured standalone command or two-stage API, not a test-only runner."""
        return _invoke(entrypoint, document, tmp_path / "trial.yaml", monkeypatch)

    observed.run = run
    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", set_alias)
    return observed


def _candidate_tags(trial, *, value):
    """Specify the portable metadata required to compare a historical champion."""
    data = trial.document if trial.entrypoint == "standalone" else trial.document["data"]
    tags = {
        "validation_status": "passed",
        "tracked_metric": trial.stage["tracked_metric"],
        "tracked_mode": trial.stage["tracked_mode"],
        "class_names": json.dumps(sorted(data["class_names"]), separators=(",", ":")),
        "tracked_metric_definition": json.dumps(
            trial.stage["validation_metrics"][trial.stage["tracked_metric"]], sort_keys=True, separators=(",", ":"),
        ),
        "best_value": str(value),
        "validation_dataset": str(Path(data["validation_dataset"]).resolve()),
    }
    revision = trial.document["mlflow"].get("dataset_revision")
    if revision is not None:
        tags["dataset_revision"] = revision
    return tags


def _seed_champion(trial, store, tmp_path, *, value=0.7, tag_updates=None):
    """Create a historical version without requiring unrelated old model weights."""
    tags = _candidate_tags(trial, value=value)
    for key, value in (tag_updates or {}).items():
        if value is None:
            tags.pop(key, None)
        else:
            tags[key] = value
    store.client.create_registered_model(store.registry_name)
    source = tmp_path / "previous-champion"
    source.mkdir()
    version = store.client.create_model_version(store.registry_name, source.as_uri(), tags=tags)
    store.client.set_registered_model_alias(
        store.registry_name, store.configuration["champion_alias"], version.version,
    )
    trial.promotions.clear()
    return version


def _champion(store):
    """Query the actual configured registry alias, not a local selection result."""
    return store.client.get_model_version_by_alias(store.registry_name, store.configuration["champion_alias"])


def _versions(store):
    """Return classifier registry versions keyed by their source training run."""
    versions = store.client.search_model_versions(f"name='{store.registry_name}'")
    by_run = {version.run_id: version for version in versions}
    assert len(by_run) == len(versions), "A classifier must be registered exactly once per run"
    return by_run


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("winner", [0, 1])
def test_register_all_classifiers_and_promote_only_the_sweep_winner_after_completion(
    classifier_trial, tracking_store, mode, winner
):
    """Select the correct registered version regardless of run order or metric direction."""
    trial = classifier_trial
    trial.stage["tracked_mode"] = mode
    if mode == "min":
        trial.score_sets = {0.001: [0.8, 0.1, 0.5], 0.002: [0.9, 0.2, 0.4]}
    if winner == 1:
        trial.score_sets[0.001], trial.score_sets[0.002] = trial.score_sets[0.002], trial.score_sets[0.001]
    result = trial.run()
    summary = _summary(trial.document)
    candidates = summary["classifier_results"]
    selected = summary["selected_classifier"]
    assert len(candidates) == 2
    assert selected["best_checkpoint"] == candidates[winner]["best_checkpoint"]
    assert selected["best_epoch"] == 2
    assert selected["best_value"] == (0.1 if mode == "min" else 0.9)
    versions = _versions(tracking_store)
    assert len(versions) == 2
    candidate_ids = [candidate["mlflow_run_id"] for candidate in candidates]
    assert len(set(candidate_ids)) == 2
    assert set(candidate_ids) == set(versions)
    winning_version = versions[selected["mlflow_run_id"]]
    assert _champion(tracking_store).version == winning_version.version
    assert trial.promotions == [(str(winning_version.version), 6)]
    for candidate in candidates:
        version = versions[candidate["mlflow_run_id"]]
        run = tracking_store.client.get_run(candidate["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        if trial.entrypoint == "two_stage":
            assert run.data.tags["mlflow.parentRunId"] == result.mlflow_run_id
        elif trial.entrypoint == "standalone":
            assert "mlflow.parentRunId" not in run.data.tags
        assert version.tags["tracked_metric"] == candidate["tracked_metric"]
        assert version.tags["tracked_mode"] == candidate["tracked_mode"]
        data = trial.document if trial.entrypoint == "standalone" else trial.document["data"]
        assert json.loads(version.tags["class_names"]) == sorted(data["class_names"])
        assert json.loads(version.tags["tracked_metric_definition"]) == trial.stage["validation_metrics"][candidate["tracked_metric"]]
        assert float(version.tags["best_value"]) == candidate["best_value"]
        assert version.tags["validation_dataset"] == str(Path(candidate["validation_dataset"]).resolve())
        assert version.tags["dataset_revision"] == tracking_store.configuration["dataset_revision"]
        checkpoint = Path(candidate["best_checkpoint"])
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Native finalization gives best.pt the complete run history, not just its epoch.
        assert snapshot["train_results"]["epoch"][-1] == 3
        assert candidate["best_epoch"] == 2
        assert snapshot["train_metrics"]["balanced_accuracy"] == candidate["best_value"]
        native_model = YOLO(str(checkpoint), task="classify").model
        if trial.entrypoint == "two_stage":
            encoder_checkpoint = result.selected_supcon.best_checkpoint
        elif trial.entrypoint == "standalone":
            encoder_checkpoint = Path(trial.stage["initialization"]["encoder"]["checkpoint"])
        assert native_model.provenance["encoder_checkpoint"] == str(encoder_checkpoint.resolve())
        for suffix in ("yaml", "md"):
            card = checkpoint.with_suffix(f".model_card.{suffix}")
            assert card.is_file()
        card = yaml.safe_load(checkpoint.with_suffix(".model_card.yaml").read_text())
        assert str(card["model"]["registry_version"]) == str(version.version)
        assert card["validation"]["epoch"] == 2
        assert card["validation"]["metrics"]["balanced_accuracy"] == candidate["best_value"]
    assert mlflow.active_run() is None


@pytest.mark.parametrize("mode,previous_value,promote", [
    ("max", 0.95, False),
    ("max", 0.9, False),
    ("max", 0.7, True),
    ("min", 0.05, False),
    ("min", 0.1, False),
    ("min", 0.3, True),
])
def test_existing_champion_is_replaced_only_by_strict_improvement(
    classifier_trial, tracking_store, tmp_path, mode, previous_value, promote
):
    """Respect metric direction and retain the historical champion on exact ties."""
    trial = classifier_trial
    trial.stage["tracked_mode"] = mode
    if mode == "min":
        trial.score_sets = {0.001: [0.8, 0.1, 0.5], 0.002: [0.9, 0.2, 0.4]}
    previous = _seed_champion(trial, tracking_store, tmp_path, value=previous_value)
    trial.run()
    summary = _summary(trial.document)
    selected = summary["selected_classifier"]
    winning_version = _versions(tracking_store)[selected["mlflow_run_id"]]
    # The current invocation still has its own winner even when the global
    # champion remains better; those two identities must not be conflated.
    assert selected["best_value"] == (0.9 if mode == "max" else 0.1)
    expected = winning_version.version if promote else previous.version
    assert _champion(tracking_store).version == expected
    assert trial.promotions == ([(str(winning_version.version), 6)] if promote else [])
    assert len(_versions(tracking_store)) == 3


@pytest.mark.parametrize("second_score,promote", [(0.85, False), (0.9, False), (0.95, True)])
@pytest.mark.parametrize("with_revision", [True, False])
def test_champion_metadata_supports_comparison_on_the_next_real_invocation(
    classifier_trial, tracking_store, second_score, promote, with_revision
):
    """Real metadata must support future comparisons, including optional dataset revisions."""
    trial = classifier_trial
    if not with_revision:
        trial.document["mlflow"].pop("dataset_revision")
    trial.run()
    first = _champion(tracking_store)
    first_summary = _summary(trial.document)
    assert first_summary["selected_classifier"]["best_value"] == 0.9
    trial.document["output_directory"] += "-next"
    data = trial.document if trial.entrypoint == "standalone" else trial.document["data"]
    data["class_names"] = list(reversed(data["class_names"]))
    trial.score_sets = {0.001: [0.2, second_score, 0.3], 0.002: [0.1, 0.8, 0.4]}
    trial.promotions.clear()
    trial.run()
    selected = _summary(trial.document)["selected_classifier"]
    winning_version = _versions(tracking_store)[selected["mlflow_run_id"]]
    assert selected["best_value"] == second_score
    assert _champion(tracking_store).version == (winning_version.version if promote else first.version)
    assert len(trial.promotions) == int(promote)
    assert len(_versions(tracking_store)) == 4


@pytest.mark.parametrize("tag_updates", [
    {"tracked_metric": "f1_macro"},
    {"tracked_mode": "min"},
    {"class_names": None},
    {"class_names": json.dumps(["mouse"], separators=(",", ":"))},
    {"tracked_metric_definition": None},
    {"tracked_metric_definition": json.dumps({"function": "accuracy_score"}, sort_keys=True, separators=(",", ":"))},
    {"tracked_metric_definition": json.dumps({"function": "balanced_accuracy_score", "adjusted": True}, sort_keys=True, separators=(",", ":"))},
    {"validation_dataset": "/different/validation"},
    {"dataset_revision": "different-dataset-revision"},
    {"tracked_metric": None},
    {"tracked_mode": None},
    {"best_value": None},
    {"best_value": "not-a-number"},
    {"best_value": "nan"},
    {"best_value": "inf"},
    {"validation_dataset": None},
    {"dataset_revision": None},
    {"validation_status": "not_validated"},
])
def test_incompatible_or_legacy_champion_requires_an_explicit_decision(
    classifier_trial, tracking_store, tmp_path, tag_updates
):
    """Missing/untrustworthy comparison metadata must never permit silent promotion."""
    trial = classifier_trial
    previous = _seed_champion(trial, tracking_store, tmp_path, tag_updates=tag_updates)
    with pytest.raises(ValueError, match="champion"):
        trial.run()
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []
    assert mlflow.active_run() is None


@pytest.mark.parametrize("failure", ["training", "publication", "model_card_attachment"])
def test_later_failure_never_promotes_a_partial_sweep(
    classifier_trial, tracking_store, tmp_path, monkeypatch, failure
):
    """A completed first candidate is not enough to supersede the historical champion."""
    trial = classifier_trial
    previous = _seed_champion(trial, tracking_store, tmp_path, value=0.7)
    if failure == "training":
        trial.fail_at = (0.002, 1)
    elif failure == "publication":
        original_log_model = mlflow.pytorch.log_model
        calls = []

        def log_model(*args, **kwargs):
            """Let the first candidate register, then fail the later publication."""
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("deliberate later publication failure")
            return original_log_model(*args, **kwargs)

        monkeypatch.setattr(mlflow.pytorch, "log_model", log_model)
    elif failure == "model_card_attachment":
        original_update = MlflowClient.update_model_version
        attached_versions = []

        def update_version(self, name, version, description=None, **kwargs):
            """Fail after the final model registers, during its final card-publication step."""
            if name == tracking_store.registry_name and description is not None:
                attached_versions.append(str(version))
                if len(attached_versions) == 2:
                    raise RuntimeError("deliberate later model-card publication failure")
            return original_update(self, name, version, description=description, **kwargs)

        monkeypatch.setattr(MlflowClient, "update_model_version", update_version)
    else:
        raise AssertionError(f"Unknown failure point: {failure}")
    with pytest.raises(RuntimeError, match="deliberate later"):
        trial.run()
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []
    summary_path = Path(trial.document["output_directory"]) / "selection.yaml"
    if summary_path.exists():
        assert yaml.safe_load(summary_path.read_text()).get("selected_classifier") is None
    assert mlflow.active_run() is None


def test_single_classifier_invocation_also_compares_against_the_historical_champion(
    classifier_trial, tracking_store, tmp_path
):
    """Cross-invocation champion protection must not apply only to multi-run sweeps."""
    trial = classifier_trial
    trial.stage["train_kwargs"]["lr0"] = 0.001
    previous = _seed_champion(trial, tracking_store, tmp_path, value=0.95)
    result = trial.run()
    summary = _summary(trial.document)
    assert len(summary["classifier_results"]) == 1
    assert summary["selected_classifier"]["best_value"] == 0.9
    assert _champion(tracking_store).version == previous.version
    assert len(_versions(tracking_store)) == 2
    assert trial.promotions == []
    if trial.entrypoint == "two_stage":
        assert result.classifier_result == result.selected_classifier


@pytest.mark.parametrize("classifier_trial", ["standalone"], indirect=True)
def test_training_only_classifiers_register_without_selecting_or_replacing_a_champion(
    classifier_trial, tracking_store, tmp_path
):
    """Validation requirements apply to promotion, not to the ability to train/register models."""
    trial = classifier_trial
    previous = _seed_champion(trial, tracking_store, tmp_path)
    trial.document.update(
        validation_dataset=None, validation_metrics=None,
        tracked_metric="ce_loss_mean", tracked_mode="min",
    )
    trial.run()
    summary = _summary(trial.document)
    assert len(summary["classifier_results"]) == 2
    assert summary["selected_classifier"] is None
    versions = _versions(tracking_store)
    assert len(versions) == 3
    for result in summary["classifier_results"]:
        assert versions[result["mlflow_run_id"]].tags["validation_status"] == "not_validated"
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []


@pytest.mark.parametrize("error_code", ["INTERNAL_ERROR", "PERMISSION_DENIED", "INVALID_PARAMETER_VALUE"])
def test_registry_lookup_failure_is_not_treated_as_an_absent_champion(
    classifier_trial, tracking_store, tmp_path, monkeypatch, error_code
):
    """Only an absent alias permits first promotion; outages and permission errors propagate."""
    trial = classifier_trial
    previous = _seed_champion(trial, tracking_store, tmp_path)
    original_lookup = MlflowClient.get_model_version_by_alias

    def unavailable(*args, **kwargs):
        """Represent an unavailable backend rather than a missing alias."""
        raise MlflowException("champion backend unavailable", error_code=error_code)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", unavailable)
    with pytest.raises(MlflowException, match="backend unavailable"):
        trial.run()
    champion = original_lookup(tracking_store.client, tracking_store.registry_name, tracking_store.configuration["champion_alias"])
    assert champion.version == previous.version
    assert trial.promotions == []
    assert mlflow.active_run() is None


def test_tracking_without_registration_still_reports_the_current_sweep_winner(
    classifier_trial, tracking_store
):
    """Champion selection is local functionality, not dependent on a model registry."""
    trial = classifier_trial
    trial.document["mlflow"].pop("registered_model_name")
    trial.run()
    summary = _summary(trial.document)
    assert len(summary["classifier_results"]) == 2
    assert summary["selected_classifier"]["best_value"] == 0.9
    assert tracking_store.client.search_registered_models() == []
    assert trial.promotions == []
    assert mlflow.active_run() is None
