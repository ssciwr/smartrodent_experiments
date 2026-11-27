"""Behavioral contracts for the MLflow settings shared by both stages."""

from smartrodent.experiment_support import MlflowConfiguration


def test_shared_mlflow_configuration_retains_optional_fields_and_tags():
    """Shared tracking configuration preserves current names, defaults, and provenance."""
    settings = {
        "tracking_uri": "sqlite:///unused-test.db",
        "experiment_name": "two-stage",
        "registered_model_name": "encoder",
        "model_artifact_name": "species-encoder",
        "champion_alias": "candidate",
        "dataset_revision": "data-revision-123",
    }
    config = MlflowConfiguration.from_mapping(settings)
    for name, value in settings.items():
        assert getattr(config, name) == value
    tags = config.run_tags()
    assert tags["dataset_revision"] == "data-revision-123"
    assert tags["git_commit"]
    assert MlflowConfiguration.from_mapping(None) is None
    default = MlflowConfiguration.from_mapping(
        {"tracking_uri": "sqlite:///unused-test.db", "experiment_name": "two-stage"}
    )
    assert default.registered_model_name is None
    assert default.champion_alias == "champion"
