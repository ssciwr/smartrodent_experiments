"""MLflow hierarchy, selected-encoder provenance, and stage-safe publication contracts."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import utils as registry_utils
import pytest
import torch
import yaml

from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration
from ultralytics import YOLO
from ultralytics.models.yolo.classify.val import ClassificationValidator
from smartrodent.classifier_training import ClassifierConfiguration, ClassifierExperiment
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment
# Reuse the already calibrated temporary CPU data/configuration fixture.
from test_two_stage_training import pipeline_document


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    """Use a real isolated SQLite tracking/registry backend and local artifact root."""
    assert mlflow.active_run() is None
    previous_tracking = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_id = client.create_experiment(
        "m6-tests", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_tracking_uri(uri)
    # get_registry_uri() resolves a None fallback, so restoring its return value
    # would create an explicit global override and leak into later client tests.
    # Monkeypatch snapshots/restores the raw setting, including that None state.
    monkeypatch.setattr(registry_utils, "_registry_uri", uri)
    mlflow.set_experiment(experiment_id=experiment_id)
    configuration = {
        "tracking_uri": uri, "experiment_name": "m6-tests",
        "model_artifact_name": "species-classifier", "champion_alias": "approved",
        "dataset_revision": "dvc-fixture-123",
    }
    try:
        yield SimpleNamespace(
            client=client, experiment_id=experiment_id, configuration=configuration,
            registry_name="m6-classifier", downloads=tmp_path / "downloads",
        )
    finally:
        # Cleanup only runs owned by this isolated test; caller-preservation tests
        # assert the caller's state before their own context exits.
        while mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(previous_tracking)


@pytest.fixture
def scripted_validation(monkeypatch):
    """Control ordinary epoch metrics only; retain real optimizers/models and MLflow."""
    observed = SimpleNamespace(supcon=[], classifier=[], failure=None)
    counts = {}

    def supcon_validate(self, model, loader, criterion):
        """Make candidate two/epoch two win the configured metric, not SupCon loss."""
        config = self.configuration
        index = int(config.optimizer["lr"] == 0.002)
        key = str(config.output_directory)
        epoch = counts.get(key, 0)
        counts[key] = epoch + 1
        observed.supcon.append((key, epoch + 1))
        if observed.failure == "supcon" and index == 1:
            raise RuntimeError("deliberate SupCon validation failure")
        values = [[0.3, 0.8, 0.5], [0.2, 0.9, 0.6]]
        if config.effective_tracked_mode == "min":
            values = [[0.7, 0.2, 0.5], [0.8, 0.1, 0.4]]
        return {"balanced_accuracy": values[index][epoch], "supcon_loss_mean": [0.1, 0.2, 0.3][epoch]}

    original = ClassificationValidator.__call__

    def classifier_validate(self, trainer=None, model=None):
        """Control returned epoch metrics after actual native classification validation."""
        metrics = original(self, trainer=trainer, model=model)
        if trainer is None:
            return metrics
        observed.classifier.append(Path(trainer.save_dir))
        if observed.failure == "classifier":
            raise RuntimeError("deliberate classifier validation failure")
        epoch = trainer.epoch
        metrics.update(balanced_accuracy=[0.1, 0.9, 0.4][epoch], ce_loss_mean=[0.1, 0.2, 0.3][epoch])
        return metrics

    monkeypatch.setattr(SupConExperiment, "_validate", supcon_validate)
    monkeypatch.setattr(ClassificationValidator, "__call__", classifier_validate)
    return observed


@pytest.fixture
def tracking_document(pipeline_document):
    """Use two three-epoch candidates plus CE, with distinct stage objectives/directions."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"]["values"] = [0.001, 0.002]
    document["classifier"].update(
        epochs=3, tracked_metric="balanced_accuracy", tracked_mode="max",
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
    )
    return document


@pytest.fixture
def standalone_configuration(pipeline_document, tmp_path):
    """Train a real local SupCon source, then configure standalone CE without pipeline context."""
    source_values = deepcopy(pipeline_document["supcon"])
    source_values.update(pipeline_document["data"])
    source_values.update(
        epochs=1, optimizer={"lr": 0.001},
        output_directory=str(tmp_path / "standalone-source"),
    )
    source = SupConExperiment(TrainingConfiguration.from_mapping(source_values)).run_with_result()
    values = deepcopy(pipeline_document["classifier"])
    values.update(pipeline_document["data"])
    values.update(
        epochs=3, output_directory=str(tmp_path / "standalone-ce"),
        tracked_metric="balanced_accuracy", tracked_mode="max",
        validation_metrics={"balanced_accuracy": {"function": "balanced_accuracy_score"}},
    )
    values["initialization"]["encoder"] = {"source": "supcon", "checkpoint": str(source.best_checkpoint)}
    return ClassifierConfiguration.from_mapping(values)


def _run_tracking_document(document, store, *, register=False):
    """Enable shared tracking without mutating the reusable document fixture."""
    values = deepcopy(document)
    values["mlflow"] = deepcopy(store.configuration)
    if register:
        values["mlflow"]["registered_model_name"] = store.registry_name
    config = TwoStageConfiguration.from_mapping(values)
    return config, TwoStageExperiment(config).run()


def _download(store, run_id, artifact):
    """Read back a real published artifact, not its local source file."""
    store.downloads.mkdir(exist_ok=True)
    return Path(store.client.download_artifacts(run_id, artifact, str(store.downloads)))


def _expected_encoder_provenance(result):
    """Require the selected run's original configured objective, not a sign-inverted score."""
    selected = result.selected_supcon
    return {
        "encoder_checkpoint": str(selected.best_checkpoint.resolve()),
        "encoder_checkpoint_sha256": hashlib.sha256(selected.best_checkpoint.read_bytes()).hexdigest(),
        "encoder_run_id": selected.mlflow_run_id,
        "encoder_epoch": selected.best_epoch,
        "encoder_tracked_metric": selected.tracked_metric,
        "encoder_tracked_mode": selected.tracked_mode,
        "encoder_best_value": selected.best_value,
        "pipeline_run_id": result.mlflow_run_id,
    }


def _assert_registered_classifier(store, result):
    """Check real registry identity, selected-epoch inference, and associated model cards."""
    versions = store.client.search_model_versions(f"name='{store.registry_name}'")
    assert len(versions) == 1
    version = versions[0]
    assert version.run_id == result.mlflow_run_id
    assert version.tags["best_epoch"] == str(result.best_epoch)
    assert version.tags["dataset_revision"] == store.configuration["dataset_revision"]
    assert version.tags["git_commit"]
    card_path = result.best_checkpoint.with_suffix(".model_card.yaml")
    card = yaml.safe_load(card_path.read_text())
    assert card["model"]["name"] == store.registry_name
    assert str(card["model"]["registry_version"]) == str(version.version)
    assert card["provenance"]["run_id"] == result.mlflow_run_id
    logged_model = store.client.get_logged_model(card["provenance"]["model_id"])
    assert logged_model.name == store.configuration["model_artifact_name"]
    assert logged_model.source_run_id == result.mlflow_run_id
    assert version.source in {f"models:/{logged_model.model_id}", logged_model.artifact_location}
    assert version.description == card_path.with_suffix(".md").read_text()
    for suffix in ("yaml", "md"):
        filename = f"best.model_card.{suffix}"
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/{filename}",
            dst_path=str(store.downloads),
        )
        assert Path(downloaded).read_bytes() == (result.best_checkpoint.parent / filename).read_bytes()
    registered = mlflow.pytorch.load_model(f"models:/{store.registry_name}/{version.version}")
    portable = YOLO(str(result.best_checkpoint), task="classify").model.eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        expected = portable(images)
        actual = registered(images)
    assert isinstance(actual, tuple) and isinstance(expected, tuple)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape == (2, 2)
        torch.testing.assert_close(actual_tensor, expected_tensor)
    assert registered.names == portable.names == {0: "mouse", 1: "rat"}
    return version


@pytest.mark.parametrize("tracking", [False, True])
def test_artifact_only_runs_retain_selected_metadata_without_extra_model_restore(
    standalone_configuration, tracking_store, scripted_validation, monkeypatch, tracking
):
    """Cards reuse native-finalized metadata without another model reconstruction."""
    import smartrodent.classifier_model as classifier_models

    restores = []
    original = classifier_models.load_checkpoint

    def restore(weight, *args, **kwargs):
        restores.append(Path(weight).resolve())
        return original(weight, *args, **kwargs)

    monkeypatch.setattr(classifier_models, "load_checkpoint", restore)
    config = replace(standalone_configuration, mlflow=MlflowConfiguration(**tracking_store.configuration) if tracking else None)
    result = ClassifierExperiment(config, encoder_provenance={"encoder_run_id": "selected-encoder-run"}).run_with_result()
    assert restores == []
    assert result.best_epoch == 2 and result.best_value == 0.9
    card = yaml.safe_load(result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
    assert card["classes"] == {0: "mouse", 1: "rat"}
    assert card["validation"]["epoch"] == 2
    assert card["validation"]["metrics"]["balanced_accuracy"] == 0.9
    assert card["provenance"]["encoder_run_id"] == "selected-encoder-run"
    assert card["provenance"]["checkpoint_sha256"] == hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
    checkpoint = YOLO(str(result.best_checkpoint), task="classify").model
    assert checkpoint.names == card["classes"]
    assert checkpoint.provenance["encoder_run_id"] == "selected-encoder-run"
    if tracking:
        run = tracking_store.client.get_run(result.mlflow_run_id)
        assert run.info.status == "FINISHED" and run.data.metrics["best_epoch"] == 2
        downloaded = _download(tracking_store, result.mlflow_run_id, "best_model/best.model_card.yaml")
        assert downloaded.read_bytes() == result.best_checkpoint.with_suffix(".model_card.yaml").read_bytes()
    else:
        assert result.mlflow_run_id is None


def test_corrupt_native_best_checkpoint_cannot_be_reported_or_published(
    standalone_configuration, monkeypatch
):
    """Skipping unnecessary model reconstruction must not hide a corrupt selected file."""
    from pickle import UnpicklingError
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    original = ClassificationTrainer.save_model
    writes = []

    def save(self):
        original(self)
        writes.append(self.best)
        self.best.write_bytes(b"corrupt native best checkpoint")

    monkeypatch.setattr(ClassificationTrainer, "save_model", save)
    config = replace(standalone_configuration, epochs=1)
    with pytest.raises((ValueError, UnpicklingError), match="checkpoint|finaliz|pickle"):
        ClassifierExperiment(config).run_with_result()
    assert writes == [config.output_directory / "weights/best.pt"]
    assert not (config.output_directory / "weights/best.model_card.yaml").exists()
    assert not (config.output_directory / "weights/best.model_card.md").exists()


@pytest.mark.parametrize("register", [False, True])
def test_shared_tracking_configuration_round_trips_before_training(
    tracking_document, tracking_store, tmp_path, register
):
    """One optional root config owns tracking; no backend run or output is created by parsing."""
    document = tracking_document
    document["mlflow"] = deepcopy(tracking_store.configuration)
    if register:
        document["mlflow"]["registered_model_name"] = tracking_store.registry_name
    before = deepcopy(document)
    config = TwoStageConfiguration.from_mapping(document)
    assert config.mlflow == MlflowConfiguration.from_mapping(before["mlflow"])
    assert document == before
    path = tmp_path / "tracking.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    restored = TwoStageConfiguration.from_yaml(path)
    assert restored.to_dict() == config.to_dict()
    assert restored.mlflow == config.mlflow
    assert tracking_store.client.search_runs([tracking_store.experiment_id]) == []
    assert not Path(document["output_directory"]).exists()


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("register", [False, True])
def test_real_pipeline_tracks_one_parent_and_stage_children_with_selected_provenance(
    tracking_document, tracking_store, scripted_validation, mode, register
):
    """Metric-selected epochs, registry model, cards and tracking artifacts must describe one result."""
    tracking_document["supcon"]["tracked_mode"] = mode
    config, result = _run_tracking_document(tracking_document, tracking_store, register=register)
    assert len(scripted_validation.supcon) == 6 and len(scripted_validation.classifier) == 3
    assert result.selected_supcon == result.supcon_results[1]
    assert result.selected_supcon.best_epoch == result.classifier_result.best_epoch == 2
    assert result.selected_supcon.best_value == (0.1 if mode == "min" else 0.9)
    assert result.classifier_result.best_value == 0.9
    child_ids = [item.mlflow_run_id for item in (*result.supcon_results, result.classifier_result)]
    assert all(child_ids) and result.mlflow_run_id not in child_ids
    runs = tracking_store.client.search_runs([tracking_store.experiment_id])
    assert len(runs) == 4 and {run.info.run_id for run in runs} == {result.mlflow_run_id, *child_ids}
    assert all(run.info.status == "FINISHED" for run in runs)
    parent = tracking_store.client.get_run(result.mlflow_run_id)
    assert "mlflow.parentRunId" not in parent.data.tags
    for run in runs:
        assert run.data.tags["git_commit"] == parent.data.tags["git_commit"]
        assert run.data.tags["dataset_revision"] == tracking_store.configuration["dataset_revision"]
        if run.info.run_id in child_ids:
            assert run.data.tags["mlflow.parentRunId"] == result.mlflow_run_id
    for candidate in result.supcon_results:
        history = tracking_store.client.get_metric_history(candidate.mlflow_run_id, "validation/balanced_accuracy")
        assert [item.step for item in history] == [1, 2, 3]
        assert tracking_store.client.get_run(candidate.mlflow_run_id).data.metrics["best_validation_balanced_accuracy"] == candidate.best_value
        card = yaml.safe_load((candidate.output_directory / "best.model_card.yaml").read_text())
        assert card["model"]["task"] == "contrastive-encoder"
        assert card["model"]["registry_version"] is None  # No untrained classifier registry versions.
    ce_run_id = result.classifier_result.mlflow_run_id
    ce_history = tracking_store.client.get_metric_history(ce_run_id, "validation/balanced_accuracy")
    assert [item.step for item in ce_history] == [1, 2, 3]
    assert [item.value for item in ce_history] == [0.1, 0.9, 0.4]
    assert tracking_store.client.get_run(ce_run_id).data.metrics["best_validation_balanced_accuracy"] == 0.9
    provenance = _expected_encoder_provenance(result)
    for checkpoint in (result.classifier_result.best_checkpoint, result.classifier_result.last_checkpoint):
        model = YOLO(str(checkpoint), task="classify").model
        assert provenance.items() <= model.provenance.items()
    card = yaml.safe_load(result.classifier_result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
    assert provenance.items() <= card["provenance"].items()
    assert card["model"]["task"] == "image-classification"
    assert card["validation"]["epoch"] == card["provenance"]["epoch"] == 2
    assert card["provenance"]["git_commit"] == parent.data.tags["git_commit"]
    assert card["provenance"]["dataset_revision"] == tracking_store.configuration["dataset_revision"]
    assert card["provenance"]["checkpoint_sha256"] == hashlib.sha256(result.classifier_result.best_checkpoint.read_bytes()).hexdigest()
    for child in (*result.supcon_results, result.classifier_result):
        for filename in ("best.pt", "best.model_card.yaml", "best.model_card.md"):
            downloaded = _download(tracking_store, child.mlflow_run_id, f"best_model/{filename}")
            assert downloaded.read_bytes() == (child.best_checkpoint.parent / filename).read_bytes()
    for filename in ("config.yaml", "selection.yaml"):
        assert _download(tracking_store, result.mlflow_run_id, filename).read_bytes() == (result.output_directory / filename).read_bytes()
    summary = yaml.safe_load((result.output_directory / "selection.yaml").read_text())
    assert summary["mlflow_run_id"] == result.mlflow_run_id
    assert summary["selected_supcon"]["mlflow_run_id"] == result.selected_supcon.mlflow_run_id
    assert summary["classifier_result"]["mlflow_run_id"] == result.classifier_result.mlflow_run_id
    assert yaml.safe_load((result.output_directory / "config.yaml").read_text()) == config.to_dict()
    if register:
        version = _assert_registered_classifier(tracking_store, result.classifier_result)
        assert {model.name for model in tracking_store.client.search_registered_models()} == {tracking_store.registry_name}
        assert version.tags["validation_status"] == "passed"
        assert version.tags["encoder_run_id"] == result.selected_supcon.mlflow_run_id
        assert version.tags["encoder_checkpoint_sha256"] == provenance["encoder_checkpoint_sha256"]
        assert tracking_store.client.get_model_version_by_alias(tracking_store.registry_name, tracking_store.configuration["champion_alias"]).version == version.version
    else:
        assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None


def test_untracked_pipeline_keeps_selected_provenance_without_touching_callers_tracking(
    tracking_document, tracking_store, scripted_validation
):
    """Provenance is local functionality too; disabled tracking neither nests nor logs into a caller."""
    with mlflow.start_run(run_name="unrelated-caller") as caller:
        mlflow.set_tag("caller_tag", "unchanged")
        before = tracking_store.client.get_run(caller.info.run_id)
        result = TwoStageExperiment(TwoStageConfiguration.from_mapping(tracking_document)).run()
        assert result.mlflow_run_id is None
        assert all(item.mlflow_run_id is None for item in (*result.supcon_results, result.classifier_result))
        provenance = _expected_encoder_provenance(result)
        for checkpoint in (result.classifier_result.best_checkpoint, result.classifier_result.last_checkpoint):
            assert provenance.items() <= YOLO(str(checkpoint), task="classify").model.provenance.items()
        card = yaml.safe_load(result.classifier_result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
        assert provenance.items() <= card["provenance"].items()
        assert mlflow.active_run().info.run_id == caller.info.run_id
        after = tracking_store.client.get_run(caller.info.run_id)
        assert after.info.status == "RUNNING"
        assert after.data.metrics == before.data.metrics and after.data.params == before.data.params
        assert after.data.tags == before.data.tags
        assert tracking_store.client.list_artifacts(caller.info.run_id) == []
        assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    assert mlflow.active_run() is None


@pytest.mark.parametrize("validation", [False, True])
def test_standalone_ce_registers_its_best_classifier_and_only_promotes_validated_models(
    standalone_configuration, tracking_store, scripted_validation, tmp_path, validation
):
    """Standalone publication remains root-level; no-val runs cannot replace a validated champion."""
    configuration = standalone_configuration
    previous_version = None
    if not validation:
        configuration = replace(
            configuration, validation_dataset=None, tracked_metric="ce_loss_mean",
            tracked_mode="min", validation_metrics=None,
        )
        tracking_store.client.create_registered_model(tracking_store.registry_name)
        prior = tmp_path / "prior-validated-model"
        prior.mkdir()
        previous_version = tracking_store.client.create_model_version(tracking_store.registry_name, prior.as_uri()).version
        tracking_store.client.set_registered_model_alias(tracking_store.registry_name, tracking_store.configuration["champion_alias"], previous_version)
    configuration = replace(configuration, mlflow=MlflowConfiguration(
        **tracking_store.configuration, registered_model_name=tracking_store.registry_name,
    ))
    before = deepcopy(configuration.to_dict())
    result = ClassifierExperiment(configuration).run_with_result()
    assert configuration.to_dict() == before
    assert result.mlflow_run_id
    run = tracking_store.client.get_run(result.mlflow_run_id)
    assert run.info.status == "FINISHED" and "mlflow.parentRunId" not in run.data.tags
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    versions = tracking_store.client.search_model_versions(f"name='{tracking_store.registry_name}'")
    published = [version for version in versions if version.run_id == result.mlflow_run_id]
    assert len(published) == 1
    version = published[0]
    assert version.tags["validation_status"] == ("passed" if validation else "not_validated")
    if validation:
        assert result.best_epoch == 2
        _assert_registered_classifier(tracking_store, result)
        expected_alias = version.version
    else:
        card = yaml.safe_load(result.best_checkpoint.with_suffix(".model_card.yaml").read_text())
        assert card["validation"]["status"] == "not_validated"
        assert str(card["model"]["registry_version"]) == str(version.version)
        expected_alias = previous_version
    assert tracking_store.client.get_model_version_by_alias(tracking_store.registry_name, tracking_store.configuration["champion_alias"]).version == expected_alias
    assert mlflow.active_run() is None


def test_standalone_ce_tracking_without_registration_remains_a_single_completed_run(
    standalone_configuration, tracking_store, scripted_validation
):
    """Adding pipeline contexts must not require one or silently register standalone tracking runs."""
    config = replace(standalone_configuration, mlflow=MlflowConfiguration(**tracking_store.configuration))
    result = ClassifierExperiment(config).run_with_result()
    run = tracking_store.client.get_run(result.mlflow_run_id)
    assert run.info.status == "FINISHED" and "mlflow.parentRunId" not in run.data.tags
    assert run.data.metrics["best_epoch"] == 2 and run.data.metrics["best_validation_balanced_accuracy"] == 0.9
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None


def test_standalone_supcon_registry_publication_remains_an_explicit_encoder(
    pipeline_document, tracking_store
):
    """Existing standalone encoder registration stays available, never labeled as a CE classifier."""
    values = deepcopy(pipeline_document["supcon"])
    values.update(pipeline_document["data"])
    values.update(epochs=1, optimizer={"lr": 0.001}, output_directory=str(Path(pipeline_document["output_directory"]).parent / "standalone-supcon"))
    encoder_name = "m6-supcon-encoder"
    values["mlflow"] = {**tracking_store.configuration, "registered_model_name": encoder_name}
    result = SupConExperiment(TrainingConfiguration.from_mapping(values)).run_with_result()
    versions = tracking_store.client.search_model_versions(f"name='{encoder_name}'")
    assert len(versions) == 1 and versions[0].run_id == result.mlflow_run_id
    card = yaml.safe_load((result.output_directory / "best.model_card.yaml").read_text())
    assert card["model"]["task"] == "contrastive-encoder"
    assert str(card["model"]["registry_version"]) == str(versions[0].version)
    assert tracking_store.client.get_run(result.mlflow_run_id).info.status == "FINISHED"
    assert mlflow.active_run() is None


@pytest.mark.parametrize("failure", ["supcon", "classifier", "publication"])
def test_pipeline_failure_closes_owned_runs_and_preserves_the_existing_champion(
    tracking_document, tracking_store, scripted_validation, monkeypatch, tmp_path, failure
):
    """A failed candidate/CE/publication is never skipped, promoted, or reported as completed."""
    tracking_store.client.create_registered_model(tracking_store.registry_name)
    prior = tmp_path / "prior-model"
    prior.mkdir()
    previous = tracking_store.client.create_model_version(tracking_store.registry_name, prior.as_uri()).version
    tracking_store.client.set_registered_model_alias(tracking_store.registry_name, tracking_store.configuration["champion_alias"], previous)
    scripted_validation.failure = failure
    if failure == "publication":
        def fail_publication(*args, **kwargs):
            """Simulate an external MLflow model-publication failure after successful CE training."""
            raise RuntimeError("deliberate classifier publication failure")
        monkeypatch.setattr(mlflow.pytorch, "log_model", fail_publication)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run_tracking_document(tracking_document, tracking_store, register=True)
    runs = tracking_store.client.search_runs([tracking_store.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(parents) == 1 and parents[0].info.status == "FAILED"
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parents[0].info.run_id]
    assert len(children) == (2 if failure == "supcon" else 3)
    assert [run.info.status for run in children].count("FAILED") == 1
    assert all(run.info.status in {"FINISHED", "FAILED"} for run in runs)
    if failure == "supcon":
        assert scripted_validation.classifier == []
    versions = tracking_store.client.search_model_versions(f"name='{tracking_store.registry_name}'")
    assert len(versions) == 1 and versions[0].version == previous
    assert tracking_store.client.get_model_version_by_alias(tracking_store.registry_name, tracking_store.configuration["champion_alias"]).version == previous
    summary = Path(tracking_document["output_directory"]) / "selection.yaml"
    if summary.exists():
        assert yaml.safe_load(summary.read_text()).get("classifier_result") is None
    assert mlflow.active_run() is None


def test_tracked_pipeline_rejects_an_unrelated_active_run_before_creating_outputs(
    tracking_document, tracking_store
):
    """Only the pipeline owns its parent run; never hijack, end, or attach to the caller's run."""
    document = deepcopy(tracking_document)
    document["mlflow"] = tracking_store.configuration
    configuration = TwoStageConfiguration.from_mapping(document)
    with mlflow.start_run(run_name="unrelated-caller") as caller:
        with pytest.raises((ValueError, RuntimeError, MlflowException), match="active|existing|run"):
            TwoStageExperiment(configuration).run()
        assert mlflow.active_run().info.run_id == caller.info.run_id
        assert tracking_store.client.get_run(caller.info.run_id).info.status == "RUNNING"
        assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
        assert not Path(document["output_directory"]).exists()
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"experiment_name": ""}, "experiment_name"),
        ({"registered_model_name": ""}, "registered_model_name"),
        ({"tracking_uri": {"type": "sweep", "values": ["sqlite:///a.db", "sqlite:///b.db"]}}, "mlflow|tracking_uri|sweep"),
        ({"unexpected": True}, "unexpected"),
    ],
)
def test_invalid_shared_tracking_settings_fail_before_runs_or_training(
    tracking_document, tracking_store, updates, message
):
    """New root-level tracking support must retain complete prevalidation and strict parsing."""
    tracking_document["mlflow"] = {**tracking_store.configuration, **updates}
    with pytest.raises((TypeError, ValueError), match=message):
        TwoStageConfiguration.from_mapping(tracking_document)
    assert tracking_store.client.search_runs([tracking_store.experiment_id]) == []
    assert not Path(tracking_document["output_directory"]).exists()


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    revision=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12),
    alias=st.sampled_from(["champion", "candidate"]),
    register=st.booleans(),
)
def test_tracking_metadata_round_trip_preserves_optional_registry_identity(
    tracking_document, tracking_store, revision, alias, register
):
    """Serialization cannot silently reset registry aliases or dataset provenance."""
    document = deepcopy(tracking_document)
    document["mlflow"] = {**tracking_store.configuration, "dataset_revision": revision, "champion_alias": alias}
    if register:
        document["mlflow"]["registered_model_name"] = tracking_store.registry_name
    config = TwoStageConfiguration.from_mapping(document)
    restored = TwoStageConfiguration.from_mapping(config.to_dict())
    assert restored.mlflow.dataset_revision == revision
    assert restored.mlflow.champion_alias == alias
    assert restored.mlflow.registered_model_name == (tracking_store.registry_name if register else None)


def test_tracking_rejects_environment_run_resumption_without_modifying_the_existing_run(
    tracking_document, tracking_store, monkeypatch, tmp_path
):
    """An environment run ID is not authorization to resume, modify, or close someone else's run."""
    caller = tracking_store.client.create_run(tracking_store.experiment_id, tags={"owner": "caller"})
    tracking_store.client.log_param(caller.info.run_id, "caller_parameter", "unchanged")
    sentinel = tmp_path / "caller-artifact.txt"
    sentinel.write_text("unchanged caller artifact")
    tracking_store.client.log_artifact(caller.info.run_id, str(sentinel))
    before = tracking_store.client.get_run(caller.info.run_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", caller.info.run_id)
    document = deepcopy(tracking_document)
    document["mlflow"] = tracking_store.configuration
    config = TwoStageConfiguration.from_mapping(document)
    assert mlflow.active_run() is None
    with pytest.raises(ValueError, match="run|resum"):
        TwoStageExperiment(config).run()
    after = tracking_store.client.get_run(caller.info.run_id)
    assert after.info.status == before.info.status == "RUNNING"
    assert after.info.end_time == before.info.end_time
    assert after.data.params == before.data.params
    assert after.data.metrics == before.data.metrics
    assert after.data.tags == before.data.tags
    assert [item.path for item in tracking_store.client.list_artifacts(caller.info.run_id)] == [sentinel.name]
    assert _download(tracking_store, caller.info.run_id, sentinel.name).read_bytes() == sentinel.read_bytes()
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    assert mlflow.active_run() is None
    assert not Path(document["output_directory"]).exists()
