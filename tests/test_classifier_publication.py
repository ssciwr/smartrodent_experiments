"""MLflow publication of classifiers: unchanged champion identity and native best.pt."""

from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from classifier_task_test_support import (
    CLASS_NAMES,
    classifier_document,
    pipeline_document,
    supcon_document,
)

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.stage_configurations import (
    ClassifierConfiguration,
    SupConConfiguration,
)
from smartrodent.supcon_training import SupConExperiment

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

import json

from classifier_task_test_support import (
    encoder_spec,
)
from mlflow import MlflowClient

# The registry format of existing champions (D11), written out literally so that
# a change to the new code cannot silently change what the tests compare against.
CE_IDENTITY = {"function": "torch.nn.functional.cross_entropy", "reduction": "mean"}


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _experiment(tmp_path, store, **overrides):
    """Prepare a tiny registered classifier experiment."""
    from smartrodent.classifier_training import ClassifierExperiment
    from smartrodent.stage_configurations import ClassifierConfiguration

    document = classifier_document(tmp_path, mlflow=store.configuration, **overrides)
    return ClassifierExperiment(ClassifierConfiguration.from_mapping(document))


def _register_old_champion(tmp_path, store, *, best_value: float) -> str:
    """Register a version tagged exactly as earlier releases tagged their champions."""
    client: MlflowClient = store.client
    name = store.registry_name
    client.create_registered_model(name)
    tags = {
        "tracked_metric": "ce_loss_mean",
        "tracked_mode": "min",
        "tracked_metric_definition": json.dumps(
            CE_IDENTITY, sort_keys=True, separators=(",", ":")
        ),
        "class_names": json.dumps(sorted(CLASS_NAMES), separators=(",", ":")),
        "best_value": str(best_value),
        "validation_dataset": str((tmp_path / "data" / "val").resolve()),
        "dataset_revision": store.configuration["dataset_revision"],
        "validation_status": "passed",
    }
    source = tmp_path / "old-source"
    source.mkdir()
    version = client.create_model_version(name, source.as_uri(), tags=tags)
    client.set_registered_model_alias(name, "approved", version.version)
    return str(version.version)


def _champion(store) -> str:
    """Read the configured alias from the registry."""
    return str(
        store.client.get_model_version_by_alias(store.registry_name, "approved").version
    )


@pytest.mark.parametrize(
    "incumbent_value,improves", [(1.0e6, True), (-1.0, False)], ids=["worse", "better"]
)
def test_an_old_format_champion_is_compared_with_the_new_result(
    tmp_path,
    mlflow_store,
    incumbent_value,
    improves,
):
    """The comparison is neither skipped nor rejected as incompatible metadata."""
    old = _register_old_champion(tmp_path, mlflow_store, best_value=incumbent_value)
    experiment = _experiment(tmp_path, mlflow_store)

    result = experiment.run_with_result()

    assert result.registered_model_version is not None
    assert result.registered_model_version != old
    assert _champion(mlflow_store) == (
        result.registered_model_version if improves else old
    )


def test_new_versions_carry_the_unchanged_comparison_tags(tmp_path, mlflow_store):
    """Published tags keep the names and formats later releases compare against."""
    result = _experiment(tmp_path, mlflow_store).run_with_result()

    tags = mlflow_store.client.get_model_version(
        mlflow_store.registry_name, result.registered_model_version
    ).tags
    assert json.loads(tags["tracked_metric_definition"]) == CE_IDENTITY
    assert tags["tracked_metric"] == "ce_loss_mean"
    assert tags["tracked_mode"] == "min"
    assert tags["validation_status"] == "passed"
    assert tags["dataset_revision"] == mlflow_store.configuration["dataset_revision"]
    assert tags["validation_dataset"] == str((tmp_path / "data" / "val").resolve())
    assert json.loads(tags["class_names"]) == sorted(CLASS_NAMES)


def test_native_run_logs_best_pt_beside_the_checkpoint_with_the_registered_model(
    tmp_path,
    mlflow_store,
):
    """StageRunner writes best.pt via ``to_native``; publication logs it."""
    experiment = _experiment(
        tmp_path,
        mlflow_store,
        encoder=encoder_spec("NativeTinyEncoder"),
        native_checkpoint=True,
    )

    result = experiment.run_with_result()

    assert (result.output_directory / "best.pt").exists()
    assert result.best_checkpoint.name == "best.ckpt"
    logged = {
        artifact.path
        for artifact in mlflow_store.client.list_artifacts(
            result.mlflow_run_id, "best_model"
        )
    }
    assert {"best_model/best.pt", "best_model/best.ckpt"} <= logged
    assert result.registered_model_version is not None


def test_a_single_pipeline_classifier_run_is_compared_with_the_historical_champion(
    tmp_path, mlflow_store
):
    """Without a study, the pipeline's one classifier still faces the champion."""

    from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment

    old = _register_old_champion(tmp_path, mlflow_store, best_value=-1.0)
    document = pipeline_document(tmp_path, mlflow=mlflow_store.configuration)

    result = TwoStageExperiment(TwoStageConfiguration.from_mapping(document)).run()

    summary = yaml.safe_load(
        (Path(document["output_directory"]) / "selection.yaml").read_text()
    )
    assert summary["registered_champion_version"] == old
    assert result.selected_classifier.registered_model_version != old
    assert _champion(mlflow_store) == old


def test_training_only_classifier_registers_without_touching_the_champion(
    tmp_path, mlflow_store
):
    """No validation means no comparison: a version appears, the alias stays."""
    old = _register_old_champion(tmp_path, mlflow_store, best_value=1.0e6)
    experiment = _experiment(tmp_path, mlflow_store, validation_dataset=None)

    result = experiment.run_with_result()

    assert result.registered_model_version not in (None, old)
    assert _champion(mlflow_store) == old
    version = mlflow_store.client.get_model_version(
        mlflow_store.registry_name, result.registered_model_version
    )
    assert version.tags["validation_status"] == "not_validated"


def test_training_only_first_classifier_does_not_become_champion(
    tmp_path, mlflow_store
):
    """Without any champion, an unvalidated run does not create the alias."""
    from mlflow.exceptions import MlflowException

    result = _experiment(
        tmp_path, mlflow_store, validation_dataset=None
    ).run_with_result()

    assert result.registered_model_version is not None
    with pytest.raises(MlflowException):
        mlflow_store.client.get_model_version_by_alias(
            mlflow_store.registry_name, "approved"
        )


def test_registered_supcon_run_publishes_its_encoder(tmp_path, mlflow_store):
    """SupCon registers the encoder, exports ONNX and logs its artifacts."""
    document = supcon_document(
        tmp_path, mlflow=mlflow_store.configuration, onnx_export=ONNX
    )
    result = SupConExperiment(
        SupConConfiguration.from_mapping(document)
    ).run_with_result()

    assert result.registered_model_version == "1"
    assert result.onnx_model.is_file()
    logged = {
        a.path
        for a in mlflow_store.client.list_artifacts(result.mlflow_run_id, "best_model")
    }
    assert {"best_model/best.ckpt", "best_model/best.onnx"} <= logged


def test_unregistered_tracked_run_logs_its_best_artifacts(tmp_path, tracking_store):
    """Without a registry name the artifacts are still logged to the run."""
    document = classifier_document(tmp_path, mlflow=tracking_store.configuration)
    result = ClassifierExperiment(
        ClassifierConfiguration.from_mapping(document)
    ).run_with_result()

    assert result.registered_model_version is None
    paths = {
        a.path
        for a in tracking_store.client.list_artifacts(
            result.mlflow_run_id, "best_model"
        )
    }
    assert "best_model/best.ckpt" in paths
    assert (
        ClassifierExperiment.promote(
            ClassifierConfiguration.from_mapping(document), result
        )
        is None
    )


def test_mlflow_runs_must_be_owned_and_explicit(tmp_path, tracking_store, monkeypatch):
    """Resumption by environment, unrelated active runs and absent parents are refused."""
    from smartrodent.experiment_support import MlflowConfiguration

    settings = MlflowConfiguration.from_mapping(tracking_store.configuration)
    with pytest.raises(ValueError, match="parent run"):
        settings.start_run("child", parent_run_id="absent")
    with mlflow.start_run(), pytest.raises(ValueError, match="unrelated"):
        settings.start_run("other")
    monkeypatch.setenv("MLFLOW_RUN_ID", "x")
    with pytest.raises(ValueError, match="MLFLOW_RUN_ID"):
        settings.start_run("resumed")
