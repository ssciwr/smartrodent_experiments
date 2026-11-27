"""Validation champions across classifier studies and successive MLflow invocations.

The comparison rules of ``ClassifierChampionRegistry`` are tested directly against
an isolated real SQLite registry. The trial tests below retain real tiny CPU
training, checkpoint loading and model-card writing through both entry points, and
cover only what depends on that wiring; only epoch validation scores are controlled.
"""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from study_test_support import grid
from training_test_support import classifier_summary, invoke_entrypoint

from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration

METRIC = {"function": "balanced_accuracy_score"}
CLASS_NAMES = ("rat", "mouse")


@pytest.fixture
def registry(tracking_store):
    """A champion registry on the real isolated SQLite registry, without any training."""
    configuration = MlflowConfiguration.from_mapping(
        {
            **tracking_store.configuration,
            "registered_model_name": tracking_store.registry_name,
        }
    )
    tracking_store.client.create_registered_model(tracking_store.registry_name)
    return ClassifierChampionRegistry(
        configuration, METRIC, CLASS_NAMES, client=tracking_store.client
    )


def _result(tmp_path, *, value, mode="max", validated=True, version=None):
    """Describe a completed classifier without training or checkpoint files."""
    return ExperimentResult(
        output_directory=tmp_path,
        last_checkpoint=tmp_path / "last.pt",
        best_checkpoint=tmp_path / "best.pt",
        best_epoch=1,
        tracked_metric="balanced_accuracy",
        tracked_mode=mode,
        best_value=value,
        validation_dataset=tmp_path / "val" if validated else None,
        registered_model_version=version,
    )


def _register(registry, tmp_path, result, *, tag_updates=None, champion=False):
    """Register a version tagged like a validated run; optionally make it champion."""
    tags = {**registry.comparison_tags(result), "validation_status": "passed"}
    for key, value in (tag_updates or {}).items():
        if value is None:
            tags.pop(key, None)
        else:
            tags[key] = value
    source = tmp_path / f"source-{len(registry.client.search_model_versions())}"
    source.mkdir()
    name = registry.configuration.registered_model_name
    version = registry.client.create_model_version(
        name, source.as_uri(), tags={key: str(value) for key, value in tags.items()}
    )
    if champion:
        registry.client.set_registered_model_alias(
            name, registry.configuration.champion_alias, version.version
        )
    return str(version.version)


def _registry_champion(registry):
    """Read the configured alias from the registry itself."""
    config = registry.configuration
    return str(
        registry.client.get_model_version_by_alias(
            config.registered_model_name, config.champion_alias
        ).version
    )


def test_first_validated_classifier_becomes_champion(registry, tmp_path):
    """An absent alias is the only registry state that permits a first promotion."""
    candidate = _result(tmp_path, value=0.5)
    version = _register(registry, tmp_path, candidate)
    candidate = _result(tmp_path, value=0.5, version=version)
    assert registry.promote(candidate) == version
    assert _registry_champion(registry) == version


@pytest.mark.parametrize(
    "mode,incumbent,promote",
    [
        ("max", 0.95, False),
        ("max", 0.9, False),  # exact tie keeps the historical champion
        ("max", 0.7, True),
        ("min", 0.05, False),
        ("min", 0.1, False),
        ("min", 0.3, True),
    ],
)
def test_existing_champion_is_replaced_only_by_strict_improvement(
    registry, tmp_path, mode, incumbent, promote
):
    """Respect metric direction and retain the historical champion on exact ties."""
    previous = _register(
        registry, tmp_path, _result(tmp_path, value=incumbent, mode=mode), champion=True
    )
    value = 0.9 if mode == "max" else 0.1
    version = _register(registry, tmp_path, _result(tmp_path, value=value, mode=mode))
    candidate = _result(tmp_path, value=value, mode=mode, version=version)
    expected = version if promote else previous
    assert registry.promote(candidate) == expected
    assert _registry_champion(registry) == expected


# One case per identity field the comparison relies on, plus unusable scores.
@pytest.mark.parametrize(
    "tag_updates",
    [
        {"tracked_metric": "f1_macro"},
        {"tracked_mode": None},
        {"tracked_mode": "min"},
        {"class_names": json.dumps(["mouse"], separators=(",", ":"))},
        {"tracked_metric_definition": json.dumps({"function": "accuracy_score"})},
        # options are part of the metric identity, not only the function name
        {
            "tracked_metric_definition": json.dumps(
                {"function": "balanced_accuracy_score", "adjusted": True},
                sort_keys=True,
                separators=(",", ":"),
            )
        },
        {"validation_dataset": "/different/validation"},
        {"dataset_revision": "different-dataset-revision"},
        {"dataset_revision": None},
        {"validation_status": "not_validated"},
        {"best_value": None},
        {"best_value": "nan"},
        {"best_value": "inf"},
    ],
)
def test_incompatible_or_legacy_champion_requires_an_explicit_decision(
    registry, tmp_path, tag_updates
):
    """Missing/untrustworthy comparison metadata must never permit silent promotion."""
    previous = _register(
        registry,
        tmp_path,
        _result(tmp_path, value=0.1),
        tag_updates=tag_updates,
        champion=True,
    )
    version = _register(registry, tmp_path, _result(tmp_path, value=0.9))
    with pytest.raises(ValueError, match="champion"):
        registry.promote(_result(tmp_path, value=0.9, version=version))
    assert _registry_champion(registry) == previous


def test_unspecified_revision_rejects_a_champion_with_a_revision(registry, tmp_path):
    """A configuration without a dataset revision cannot match a revisioned champion."""
    previous = _register(
        registry, tmp_path, _result(tmp_path, value=0.1), champion=True
    )
    unrevised = ClassifierChampionRegistry(
        replace(registry.configuration, dataset_revision=None),
        METRIC,
        CLASS_NAMES,
        client=registry.client,
    )
    version = _register(unrevised, tmp_path, _result(tmp_path, value=0.9))
    with pytest.raises(ValueError, match="champion"):
        unrevised.promote(_result(tmp_path, value=0.9, version=version))
    assert _registry_champion(registry) == previous


# INVALID_PARAMETER_VALUE is also how the SQLite registry reports an absent alias;
# with any other message it must propagate like every other registry error.
@pytest.mark.parametrize(
    "error_code", ["INTERNAL_ERROR", "PERMISSION_DENIED", "INVALID_PARAMETER_VALUE"]
)
def test_registry_lookup_failure_is_not_treated_as_an_absent_champion(
    registry, tmp_path, monkeypatch, error_code
):
    """Only an absent alias permits first promotion; outages and permission errors propagate."""
    version = _register(registry, tmp_path, _result(tmp_path, value=0.9))

    def unavailable(*args, **kwargs):
        """Represent an unavailable backend rather than a missing alias."""
        raise MlflowException("champion backend unavailable", error_code=error_code)

    monkeypatch.setattr(registry.client, "get_model_version_by_alias", unavailable)
    with pytest.raises(MlflowException, match="backend unavailable"):
        registry.promote(_result(tmp_path, value=0.9, version=version))


def test_training_only_results_are_never_promoted(registry, tmp_path):
    """Without validation there is nothing to compare, so no champion is selected."""
    assert registry.promote(_result(tmp_path, value=0.9, validated=False)) is None


def test_unregistered_validated_results_cannot_be_promoted(registry, tmp_path):
    """A validated winner must identify its registered version before promotion."""
    with pytest.raises(ValueError, match="registered classifier version"):
        registry.promote(_result(tmp_path, value=0.9))


@pytest.fixture(params=["standalone", "two_stage"])
def classifier_trial(
    request,
    classifier_configuration,
    classifier_input_pipeline_document,
    tracking_store,
    tmp_path,
    monkeypatch,
    controlled_scores,
):
    """Configure a real two-trial CE study whose lr=0.001 trial wins at its second epoch."""
    entrypoint = request.param
    if entrypoint == "standalone":
        document = classifier_configuration.to_dict()
        stage = document
    elif entrypoint == "two_stage":
        document = deepcopy(classifier_input_pipeline_document)
        # One SupCon run is sufficient here; orchestration tests cover SupCon studies.
        document["supcon"].update(epochs=1, optimizer={"lr": 0.001})
        stage = document["classifier"]
    else:
        raise AssertionError(f"Unknown test entry point: {entrypoint}")
    document["output_directory"] = str(tmp_path / f"tracked-{entrypoint}")
    document["mlflow"] = {
        **tracking_store.configuration,
        "registered_model_name": tracking_store.registry_name,
    }
    stage.update(
        epochs=3,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
        train_kwargs={
            **stage["train_kwargs"],
            "lr0": {"type": "suggest-categorical", "choices": [0.001, 0.002]},
        },
    )
    study = grid(study_name="classifier")
    document["tuning"] = study if entrypoint == "standalone" else {"classifier": study}
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
        return invoke_entrypoint(
            entrypoint, document, tmp_path / "trial.yaml", monkeypatch
        )

    def single_run():
        """Train one classifier without a study: a fixed learning rate, no tuning section."""
        stage["train_kwargs"]["lr0"] = 0.001
        document.pop("tuning")

    observed.run, observed.single_run = run, single_run
    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", set_alias)
    return observed


def _candidate_tags(trial, *, value):
    """Specify the portable metadata required to compare a historical champion."""
    data = (
        trial.document if trial.entrypoint == "standalone" else trial.document["data"]
    )
    tags = {
        "validation_status": "passed",
        "tracked_metric": trial.stage["tracked_metric"],
        "tracked_mode": trial.stage["tracked_mode"],
        "class_names": json.dumps(sorted(data["class_names"]), separators=(",", ":")),
        "tracked_metric_definition": json.dumps(
            trial.stage["validation_metrics"][trial.stage["tracked_metric"]],
            sort_keys=True,
            separators=(",", ":"),
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
    for key, tag_value in (tag_updates or {}).items():
        if tag_value is None:
            tags.pop(key, None)
        else:
            tags[key] = tag_value
    store.client.create_registered_model(store.registry_name)
    source = tmp_path / "previous-champion"
    source.mkdir()
    version = store.client.create_model_version(
        store.registry_name, source.as_uri(), tags=tags
    )
    store.client.set_registered_model_alias(
        store.registry_name,
        store.configuration["champion_alias"],
        version.version,
    )
    trial.promotions.clear()
    return version


def _champion(store):
    """Query the actual configured registry alias, not a local selection result."""
    return store.client.get_model_version_by_alias(
        store.registry_name, store.configuration["champion_alias"]
    )


def _versions(store):
    """Return classifier registry versions keyed by their source training run."""
    versions = store.client.search_model_versions(f"name='{store.registry_name}'")
    by_run = {version.run_id: version for version in versions}
    assert len(by_run) == len(versions), (
        "A classifier must be registered exactly once per run"
    )
    return by_run


def test_register_all_classifiers_and_promote_only_the_study_winner_after_completion(
    classifier_trial, tracking_store
):
    """Register every candidate and promote the study winner's version after completion.

    Winner selection by metric direction is tested in test_hyperparameter_search.py and
    by the comparison tests above; this checks the registration and promotion wiring
    with the lr=0.002 trial winning a "min" metric.
    """
    trial = classifier_trial
    trial.stage["tracked_mode"] = "min"
    trial.score_sets = {0.001: [0.9, 0.2, 0.4], 0.002: [0.8, 0.1, 0.5]}
    result = trial.run()
    summary = classifier_summary(trial.document, trial.entrypoint)
    candidates = summary["classifier_results"]
    selected = summary["selected_classifier"]
    assert len(candidates) == 2
    assert selected in candidates
    assert selected["best_epoch"] == 2
    assert selected["best_value"] == 0.1
    versions = _versions(tracking_store)
    assert len(versions) == 2
    candidate_ids = [candidate["mlflow_run_id"] for candidate in candidates]
    assert len(set(candidate_ids)) == 2
    assert set(candidate_ids) == set(versions)
    winning_version = versions[selected["mlflow_run_id"]]
    assert _champion(tracking_store).version == winning_version.version
    assert trial.promotions == [(str(winning_version.version), 6)]
    # The champion's comparison tags are proven usable by the next-invocation test.
    parents = set()
    for candidate in candidates:
        version = versions[candidate["mlflow_run_id"]]
        run = tracking_store.client.get_run(candidate["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        parents.add(run.data.tags["mlflow.parentRunId"])
        checkpoint = Path(candidate["best_checkpoint"])
        card = yaml.safe_load(checkpoint.with_suffix(".model_card.yaml").read_text())
        assert str(card["model"]["registry_version"]) == str(version.version)
    # Every trial is a child of one run: the study's own, or the pipeline's.
    assert len(parents) == 1
    if trial.entrypoint == "two_stage":
        assert parents == {result.mlflow_run_id}
    assert mlflow.active_run() is None


# Both entry points write their own metadata; the optional revision is checked once.
@pytest.mark.parametrize(
    "classifier_trial,with_revision",
    [("standalone", True), ("two_stage", True), ("standalone", False)],
    indirect=["classifier_trial"],
)
def test_champion_metadata_supports_comparison_on_the_next_real_invocation(
    classifier_trial, tracking_store, with_revision
):
    """Real metadata must support future comparisons, including optional dataset revisions.

    The second invocation ties the first: that only keeps the champion if the
    published tags are complete and the stored score was not rounded down.
    """
    trial = classifier_trial
    if not with_revision:
        trial.document["mlflow"].pop("dataset_revision")
    trial.run()
    first = _champion(tracking_store)
    first_summary = classifier_summary(trial.document, trial.entrypoint)
    assert first_summary["selected_classifier"]["best_value"] == 0.9
    trial.document["output_directory"] += "-next"
    data = (
        trial.document if trial.entrypoint == "standalone" else trial.document["data"]
    )
    data["class_names"] = list(reversed(data["class_names"]))
    trial.score_sets = {0.001: [0.2, 0.9, 0.3], 0.002: [0.1, 0.8, 0.4]}
    trial.promotions.clear()
    trial.run()
    selected = classifier_summary(trial.document, trial.entrypoint)[
        "selected_classifier"
    ]
    assert selected["best_value"] == 0.9
    assert _champion(tracking_store).version == first.version
    assert trial.promotions == []
    assert len(_versions(tracking_store)) == 4


def test_incompatible_champion_stops_the_run_without_promotion(
    classifier_trial, tracking_store, tmp_path
):
    """The comparison error reaches the caller, and every owned run is closed."""
    trial = classifier_trial
    previous = _seed_champion(
        trial,
        tracking_store,
        tmp_path,
        tag_updates={"validation_dataset": "/different/validation"},
    )
    with pytest.raises(ValueError, match="champion"):
        trial.run()
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []
    assert mlflow.active_run() is None


# Study failure handling is shared by both entry points; the two-stage pipeline
# additionally owns a parent run, so one failure mode is checked there too.
@pytest.mark.parametrize(
    "classifier_trial,failure",
    [
        ("standalone", "training"),
        ("standalone", "publication"),
        ("standalone", "model_card_attachment"),
        ("two_stage", "training"),
    ],
    indirect=["classifier_trial"],
)
def test_later_failure_never_promotes_a_partial_study(
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
                    raise RuntimeError(
                        "deliberate later model-card publication failure"
                    )
            return original_update(
                self, name, version, description=description, **kwargs
            )

        monkeypatch.setattr(MlflowClient, "update_model_version", update_version)
    else:
        raise AssertionError(f"Unknown failure point: {failure}")
    with pytest.raises(RuntimeError, match="deliberate later"):
        trial.run()
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []
    if trial.entrypoint == "two_stage":
        summary_path = Path(trial.document["output_directory"]) / "selection.yaml"
        assert yaml.safe_load(summary_path.read_text())["selected_classifier"] is None
    assert mlflow.active_run() is None


def test_single_classifier_invocation_also_compares_against_the_historical_champion(
    classifier_trial, tracking_store, tmp_path
):
    """Cross-invocation champion protection must not apply only to studies."""
    trial = classifier_trial
    trial.single_run()
    previous = _seed_champion(trial, tracking_store, tmp_path, value=0.95)
    result = trial.run()
    versions = _versions(tracking_store)
    assert len(versions) == 2
    (new,) = [version for version in versions.values() if version.run_id]
    assert new.tags["best_value"] == "0.9"
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []
    if trial.entrypoint == "two_stage":
        assert result.classifier_results == (result.selected_classifier,)


@pytest.mark.parametrize("classifier_trial", ["standalone"], indirect=True)
def test_training_only_classifiers_register_without_selecting_or_replacing_a_champion(
    classifier_trial, tracking_store, tmp_path
):
    """Validation requirements apply to promotion, not to the ability to train/register models."""
    trial = classifier_trial
    trial.single_run()
    previous = _seed_champion(trial, tracking_store, tmp_path)
    trial.document.update(
        validation_dataset=None,
        validation_metrics=None,
        tracked_metric="ce_loss_mean",
        tracked_mode="min",
    )
    trial.run()
    versions = _versions(tracking_store)
    assert len(versions) == 2
    (new,) = [version for version in versions.values() if version.run_id]
    assert new.tags["validation_status"] == "not_validated"
    assert _champion(tracking_store).version == previous.version
    assert trial.promotions == []


@pytest.mark.parametrize("classifier_trial", ["standalone"], indirect=True)
def test_tracking_without_registration_still_reports_the_current_study_winner(
    classifier_trial, tracking_store
):
    """Champion selection is local functionality, not dependent on a model registry."""
    trial = classifier_trial
    trial.document["mlflow"].pop("registered_model_name")
    trial.run()
    summary = classifier_summary(trial.document, trial.entrypoint)
    assert len(summary["classifier_results"]) == 2
    assert summary["selected_classifier"]["best_value"] == 0.9
    assert tracking_store.client.search_registered_models() == []
    assert trial.promotions == []
    assert mlflow.active_run() is None
