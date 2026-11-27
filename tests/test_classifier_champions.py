"""Validation champions across classifier studies and successive MLflow invocations.

The comparison rules of ``ClassifierChampionRegistry`` are tested directly against
an isolated real SQLite registry."""

import json
from dataclasses import replace

import pytest
from mlflow.exceptions import MlflowException

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
