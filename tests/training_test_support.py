"""Shared tiny CPU training fixtures and helpers for the training test modules.

Registered as a pytest plugin in ``conftest.py``: test modules request these
fixtures by parameter name and never import them, so no test module depends on
another test module. Plain helper functions are imported explicitly.
"""

import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.pytorch
import pytest
import torch
import yaml
from mlflow import MlflowClient
from mlflow.tracking._model_registry import utils as registry_utils
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from study_test_support import grid
from ultralytics import YOLO
from ultralytics.models.yolo.classify.val import ClassificationValidator

from scripts import train_classifier
from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
)
from smartrodent.experiment_results import ExperimentResult
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration
from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment


@pytest.fixture
def classifier_assets(tmp_path):
    """Create tiny images and native and SupCon encoder inputs."""
    torch.set_num_threads(2)
    for split, counts in [("train", (5, 5)), ("val", (3, 2))]:
        for label, (name, count) in enumerate(
            zip(("mouse", "rat"), counts, strict=True)
        ):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (32, 32), (30 + 20 * index, 255 * label, 80)).save(
                    directory / f"{index}.png"
                )
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "nn.Dropout", [0.5]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        source = YoloSupConModel(str(architecture), num_classes=2, output_features=4)
        source.requires_grad_(True)
        source.train()
        source.label_names = {"mouse": 0, "rat": 1}
        source.full_model.model.names = {0: "mouse", 1: "rat"}
        optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        _, embeddings = source(torch.rand(4, 3, 32, 32))
        SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
        optimizer.step()
        for state in optimizer.state.values():
            state["step"].fill_(99)
        supcon = tmp_path / "supcon-best.pt"
        torch.save(
            {
                "epoch": 99,
                "architecture": deepcopy(source.full_model.model.yaml),
                # Stage-specific: the classifier's own freeze setting must win over this.
                "model_configuration": {
                    "dropout": 0.0,
                    "output_features": 4,
                    "num_classes": 2,
                    "freeze": [0],
                },
                "model_state_dict": source.state_dict(),
                "label_names": dict(source.label_names),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": {"best": -123.0, "num_bad_epochs": 50},
            },
            supcon,
        )
        source.finalize()
        native = tmp_path / "native-cls.pt"
        yolo = YOLO(str(architecture), task="classify")
        yolo.model.names = {0: "mouse", 1: "rat"}
        yolo.save(native)
    return SimpleNamespace(
        train=tmp_path / "train",
        val=tmp_path / "val",
        supcon=supcon,
        ultralytics=native,
        architecture=architecture,
    )


@pytest.fixture
def classifier_configuration(classifier_assets, tmp_path):
    """Keep shared data/batching settings while passing CE-specific options natively."""
    return ClassifierConfiguration(
        train_dataset=classifier_assets.train,
        validation_dataset=classifier_assets.val,
        output_directory=tmp_path / "ce-output",
        class_names=("rat", "mouse"),
        initialization={
            "encoder": {
                "source": "supcon",
                "checkpoint": str(classifier_assets.supcon),
            },
            "head": {"replace": "linear", "source": "random"},
        },
        # Fixture network layers: 0=Conv, 1=Dropout (encoder), 2=Classify (head).
        # Freezing both encoder layers is valid for linear and full replacement.
        freeze=2,
        encoder_batchnorm="eval",
        transforms={
            "image_size": 32,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
            "random_grayscale": {"p": 0.0},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
            "random_horizontal_flip": {"p": 0.0},
            "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
        },
        train_kwargs={
            "optimizer": "AdamW",
            "lr0": 0.002,
            "lrf": 1.0,
            "weight_decay": 0.0,
            "warmup_epochs": 0.0,
            "nbs": 1,
            "patience": 0,
            "amp": False,
            "plots": False,
        },
        epochs=3,
        batch_size=4,
        workers=0,
        device="cpu",
        seed=42,
        tracked_metric="ce_loss_mean",
        tracked_mode="min",
    )


@pytest.fixture
def controlled_scores(monkeypatch):
    """Control only returned epoch scores after real native validation has completed."""
    observed = SimpleNamespace(score_sets={}, epochs={}, validations=[], fail_at=None)
    original = ClassificationValidator.__call__

    def validate(self, trainer=None, model=None):
        metrics = original(self, trainer=trainer, model=model)
        # Native final best-model evaluation is not another training epoch.
        if trainer is None:
            return metrics
        directory = str(trainer.save_dir)
        epoch = trainer.epoch + 1
        observed.epochs[directory] = epoch
        observed.validations.append((directory, epoch))
        rate = trainer.args.lr0
        if observed.fail_at == (rate, epoch):
            raise RuntimeError("deliberate later classifier failure")
        if rate in observed.score_sets:
            metrics["balanced_accuracy"] = observed.score_sets[rate][epoch - 1]
            # Deliberately disagree with the requested score and direction.
            metrics["ce_loss_mean"] = 0.1 * epoch
        return metrics

    monkeypatch.setattr(ClassificationValidator, "__call__", validate)
    return observed


@pytest.fixture
def experiment_configuration(tmp_path):
    """Create temporary images and a tiny Ultralytics architecture without downloads."""
    dataset = tmp_path / "images"
    for name, color in (("mouse", "red"), ("rat", "blue")):
        directory = dataset / name
        directory.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (16, 16), color).save(directory / f"{index}.png")
        (directory / "notes.txt").write_text("Ignore non-image files")
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    return TrainingConfiguration(
        train_dataset=dataset,
        output_directory=tmp_path / "outputs",
        class_names=("rat", "mouse"),
        model_name=str(architecture),
        epochs=2,
        classes_per_batch=2,
        samples_per_class=2,
        samples_per_epoch=4,
        workers=0,
        device="cpu",
        output_features=4,
        transforms={
            "image_size": 16,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.2},
            "random_grayscale": {"p": 0.2},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.2},
            "random_horizontal_flip": {"p": 0.2},
            "random_resized_crop": {"scale": [0.75, 1.0], "ratio": [0.75, 1.33]},
        },
        optimizer={"lr": 0.001},
    )


@pytest.fixture
def validation_configuration(experiment_configuration, tmp_path):
    """Create identifiable validation images only inside the temporary test fixture."""
    validation = tmp_path / "validation-images"
    for label, name in enumerate(sorted(experiment_configuration.class_names)):
        directory = validation / name
        directory.mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (16, 16), (20 + 30 * index, 255 * label, 80)).save(
                directory / f"{index}.png"
            )
    return replace(experiment_configuration, validation_dataset=validation)


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
        "tracking_uri": uri,
        "experiment_name": "m6-tests",
        "model_artifact_name": "species-classifier",
        "champion_alias": "approved",
        "dataset_revision": "dvc-fixture-123",
    }
    try:
        yield SimpleNamespace(
            client=client,
            experiment_id=experiment_id,
            configuration=configuration,
            registry_name="m6-classifier",
            downloads=tmp_path / "downloads",
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
        """Make lr=0.002 at epoch two win the configured metric, not SupCon loss.

        The second candidate to run fails when a SupCon failure is requested.
        """
        config = self.configuration
        index = int(config.optimizer["lr"] == 0.002)
        key = str(config.output_directory)
        epoch = counts.get(key, 0)
        counts[key] = epoch + 1
        observed.supcon.append((key, epoch + 1))
        if observed.failure == "supcon" and len(counts) == 2:
            raise RuntimeError("deliberate SupCon validation failure")
        values = [[0.3, 0.8, 0.5], [0.2, 0.9, 0.6]]
        if config.effective_tracked_mode == "min":
            values = [[0.7, 0.2, 0.5], [0.8, 0.1, 0.4]]
        return {
            "balanced_accuracy": values[index][epoch],
            "supcon_loss_mean": [0.1, 0.2, 0.3][epoch],
        }

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
        metrics.update(
            balanced_accuracy=[0.1, 0.9, 0.4][epoch],
            ce_loss_mean=[0.1, 0.2, 0.3][epoch],
        )
        return metrics

    monkeypatch.setattr(SupConExperiment, "_validate", supcon_validate)
    monkeypatch.setattr(ClassificationValidator, "__call__", classifier_validate)
    return observed


def download_artifact(store, run_id, artifact):
    """Read back a real published artifact, not its local source file.

    Args:
        store: The ``tracking_store`` fixture value.
        run_id: MLflow run that logged the artifact.
        artifact: Artifact path inside that run.

    Returns:
        Local path of the downloaded copy.
    """
    store.downloads.mkdir(exist_ok=True)
    return Path(store.client.download_artifacts(run_id, artifact, str(store.downloads)))


def assert_registered_classifier(store, result):
    """Check real registry identity, selected-epoch inference, and associated model cards.

    Args:
        store: The ``tracking_store`` fixture value.
        result: Experiment result whose best checkpoint was registered.

    Returns:
        The single registered model version.
    """
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
    assert version.source in {
        f"models:/{logged_model.model_id}",
        logged_model.artifact_location,
    }
    assert version.description == card_path.with_suffix(".md").read_text()
    for suffix in ("yaml", "md"):
        filename = f"best.model_card.{suffix}"
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/{filename}",
            dst_path=str(store.downloads),
        )
        assert (
            Path(downloaded).read_bytes()
            == (result.best_checkpoint.parent / filename).read_bytes()
        )
    registered = mlflow.pytorch.load_model(
        f"models:/{store.registry_name}/{version.version}"
    )
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


@pytest.fixture
def pipeline_document(tmp_path):
    """Build both existing splits and a tiny local model, never touching real data."""
    torch.set_num_threads(2)
    for split in ("train", "val"):
        for label, name in enumerate(("mouse", "rat")):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (16, 16), (40 + 20 * index, label * 220, 80)).save(
                    directory / f"{index}.png"
                )
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    transforms = {
        "image_size": 16,
        "resize": {"antialias": True},
        "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
        "random_grayscale": {"p": 0.0},
        "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
        "random_horizontal_flip": {"p": 0.0},
        "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
    }
    classifier_transforms = deepcopy(transforms)
    classifier_transforms["random_horizontal_flip"]["p"] = 0.3
    return {
        "output_directory": str(tmp_path / "pipeline"),
        "data": {
            "train_dataset": str(tmp_path / "train"),
            "validation_dataset": str(tmp_path / "val"),
            "class_names": ["rat", "mouse"],
        },
        "supcon": {
            "model_name": str(architecture),
            "epochs": 3,
            "transforms": deepcopy(transforms),
            "optimizer": {
                "lr": {"type": "suggest-categorical", "choices": [0.001, 0.002, 0.003]}
            },
            "classes_per_batch": 2,
            "samples_per_class": 2,
            "samples_per_epoch": 8,
            "workers": 0,
            "device": "cpu",
            "seed": 11,
            "dropout": 0.0,
            "output_features": 4,
            "temperature": 0.1,
            "tracked_metric": "balanced_accuracy",
            "tracked_mode": "max",
            "validation_metrics": {
                "balanced_accuracy": {"function": "balanced_accuracy_score"}
            },
        },
        "classifier": {
            "initialization": {
                "encoder": {"source": "selected_supcon"},
                "head": {"replace": "linear", "source": "random"},
            },
            # Tiny network: 0=Conv encoder, 1=Classify head; valid for both head modes.
            "freeze": 1,
            "encoder_batchnorm": "eval",
            "epochs": 2,
            "batch_size": 3,
            "workers": 0,
            "device": "cpu",
            "seed": 29,
            "transforms": classifier_transforms,
            "train_kwargs": {
                "optimizer": "AdamW",
                "lr0": 0.004,
                "lrf": 1.0,
                "warmup_epochs": 0.0,
                "nbs": 1,
                "patience": 0,
                "amp": False,
                "plots": False,
            },
            "tracked_metric": "ce_loss_mean",
            "tracked_mode": "min",
        },
        "tuning": {"supcon": grid(study_name="supcon")},
    }


@pytest.fixture
def recorded_stages(monkeypatch):
    """Replace only stage execution, retaining real configuration validation and ranking."""
    observed = SimpleNamespace(
        supcon=[],
        classifier=[],
        results=[],
        events=[],
        values=[0.2, 0.8, 0.5],
        fail_at=None,
        fail_classifier=False,
        overrides={},
    )

    def run_supcon(self, *, parent_run_id=None, epoch_callback=None):
        """Publish distinguishable best/last bytes and report original configured scores."""
        config = self.configuration
        index = len(observed.supcon)
        observed.supcon.append(config)
        observed.events.append("supcon")
        if index == observed.fail_at:
            raise RuntimeError("deliberate SupCon candidate failure")
        directory = Path(config.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        best, last = directory / "best.pt", directory / "last.pt"
        torch.save({"candidate": index, "epoch": 2}, best)
        torch.save({"candidate": index, "epoch": 3}, last)
        result = ExperimentResult(
            output_directory=directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=2,
            best_value=observed.values[index],
            tracked_metric=config.effective_tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
        )
        result = replace(result, **observed.overrides.get(index, {}))
        observed.results.append(result)
        return result

    def run_classifier(
        self, *, parent_run_id=None, defer_promotion=False, epoch_callback=None
    ):
        """Record the actual handoff configuration without requiring fake SupCon weights to load."""
        config = self.configuration
        observed.classifier.append(config)
        observed.events.append("classifier")
        if observed.fail_classifier:
            raise RuntimeError("deliberate classifier failure")
        directory = Path(config.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "weights").mkdir(exist_ok=True)
        best, last = directory / "weights/best.pt", directory / "weights/last.pt"
        torch.save({"epoch": 1}, best)
        torch.save({"epoch": 2}, last)
        return ExperimentResult(
            output_directory=directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=1,
            best_value=0.4,
            tracked_metric=config.tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
        )

    monkeypatch.setattr(SupConExperiment, "run_with_result", run_supcon)
    monkeypatch.setattr(ClassifierExperiment, "run_with_result", run_classifier)
    return observed


def serialized_result(result):
    """Specify the portable result fields, allowing additional summary metadata.

    Args:
        result: Experiment result to describe.

    Returns:
        The fields a saved selection summary must contain for ``result``.
    """
    return {
        "output_directory": str(result.output_directory),
        "last_checkpoint": str(result.last_checkpoint),
        "best_checkpoint": str(result.best_checkpoint),
        "best_epoch": result.best_epoch,
        "best_value": result.best_value,
        "tracked_metric": result.tracked_metric,
        "tracked_mode": result.tracked_mode,
        "validation_dataset": (
            str(result.validation_dataset) if result.validation_dataset else None
        ),
        "mlflow_run_id": result.mlflow_run_id,
    }


def invoke_entrypoint(entrypoint, document, config_path, monkeypatch):
    """Exercise the real entry point, a standalone script or the two-stage pipeline.

    Args:
        entrypoint: ``"standalone"`` (train_classifier script) or ``"two_stage"``.
        document: Configuration mapping written to ``config_path``.
        config_path: Where the configuration file is written.
        monkeypatch: Pytest monkeypatch, used to set ``sys.argv``.

    Returns:
        Whatever the entry point returns.

    Raises:
        AssertionError: For an unknown ``entrypoint``.
    """
    config_path.write_text(yaml.safe_dump(document))
    if entrypoint == "standalone":
        monkeypatch.setattr(sys, "argv", ["train_classifier.py", str(config_path)])
        return train_classifier.main()
    elif entrypoint == "two_stage":
        configuration = TwoStageConfiguration.from_yaml(config_path)
        return TwoStageExperiment(configuration).run()
    else:
        raise AssertionError(f"Unknown test entry point: {entrypoint}")


def classifier_summary(document, entrypoint):
    """Read the classifier candidates and winner from the entry point's portable report.

    Args:
        document: Configuration mapping whose ``output_directory`` holds the report.
        entrypoint: ``"standalone"`` (a study's ``study.yaml``) or ``"two_stage"``
            (the pipeline's ``selection.yaml``).

    Returns:
        ``classifier_results`` (completed candidates in order) and ``selected_classifier``.

    Raises:
        AssertionError: For an unknown ``entrypoint``.
    """
    root = Path(document["output_directory"])
    if entrypoint == "two_stage":
        summary = yaml.safe_load((root / "selection.yaml").read_text())
        return {
            "classifier_results": summary["classifier_results"],
            "selected_classifier": summary["selected_classifier"],
        }
    elif entrypoint == "standalone":
        report = yaml.safe_load((root / "study.yaml").read_text())
        selected = report["selected_trial"]
        return {
            "classifier_results": [
                trial["result"] for trial in report["trials"] if trial["result"]
            ],
            "selected_classifier": selected["result"] if selected else None,
        }
    else:
        raise AssertionError(f"Unknown test entry point: {entrypoint}")


@pytest.fixture
def classifier_input_pipeline_document(classifier_configuration, tmp_path):
    """Share the classifier fixture's inputs rather than creating colliding image directories."""
    stage = classifier_configuration.to_dict()
    data = {
        key: stage.pop(key)
        for key in ("train_dataset", "validation_dataset", "class_names")
    }
    stage.pop("output_directory")
    stage.pop("mlflow")
    stage["initialization"]["encoder"] = {"source": "selected_supcon"}
    # The pipeline's tiny network (0=Conv, 1=Classify) is shallower than the fixture's.
    stage["freeze"] = 1
    architecture = tmp_path / "pipeline-model.yaml"
    architecture.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    return {
        "output_directory": str(tmp_path / "pipeline"),
        "data": data,
        "supcon": {
            "model_name": str(architecture),
            "epochs": 3,
            "transforms": deepcopy(stage["transforms"]),
            "optimizer": {
                "lr": {"type": "suggest-categorical", "choices": [0.001, 0.002, 0.003]}
            },
            "classes_per_batch": 2,
            "samples_per_class": 2,
            "samples_per_epoch": 8,
            "workers": 0,
            "device": "cpu",
            "seed": 11,
            "dropout": 0.0,
            "output_features": 4,
            "temperature": 0.1,
            "tracked_metric": "balanced_accuracy",
            "tracked_mode": "max",
            "validation_metrics": {
                "balanced_accuracy": {"function": "balanced_accuracy_score"}
            },
        },
        "classifier": stage,
        "tuning": {"supcon": grid(study_name="supcon")},
    }


def native_checkpoint(architecture, path, *, width=8, names=None):
    """Write a standard Ultralytics checkpoint and account for its FP16 save conversion.

    Args:
        architecture: Tiny classifier architecture YAML to start from.
        path: Destination checkpoint path.
        width: Output channels of the first convolution.
        names: Class-index-to-name mapping; defaults to mouse/rat.

    Also writes the widened architecture next to it as ``path.with_suffix(".yaml")``.

    Returns:
        Namespace with ``path``, the reloaded eval ``network`` and ``source_format``.
    """
    document = yaml.safe_load(architecture.read_text())
    document["backbone"][0][3][0] = width
    classes = names if names is not None else {0: "mouse", 1: "rat"}
    document["nc"] = len(classes)
    document["head"][-1][3][0] = len(classes)
    config_path = path.with_suffix(".yaml")
    config_path.write_text(yaml.safe_dump(document))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yolo = YOLO(str(config_path), task="classify")
        yolo.model.names = classes
        head = yolo.model.model[-1]
        with torch.no_grad():
            head.conv.conv.weight.fill_(0.25)
            head.conv.bn.running_mean.fill_(1.25)
            head.conv.bn.running_var.fill_(0.75)
            head.linear.weight.fill_(0.125)
            head.linear.bias.copy_(torch.linspace(-0.5, 0.5, len(classes)))
        head.drop.p = 0.35
        yolo.save(path)
        # Compare to what is actually stored, not to pre-quantization FP32 values.
        network = YOLO(str(path), task="classify").model.eval()
    return SimpleNamespace(path=path, network=network, source_format="ultralytics")


# Ultralytics' native exporter rejects files below 0.1 MB, so classifier exports
# need wider networks than the 8-channel fixtures used elsewhere.
EXPORT_WIDTH = 32


def widened_architecture(architecture, path):
    """Write a copy of a tiny architecture with an exportable first-layer width.

    Args:
        architecture: Tiny architecture YAML to copy.
        path: Destination of the widened copy.

    Returns:
        ``path``.
    """
    document = yaml.safe_load(architecture.read_text())
    document["backbone"][0][3][0] = EXPORT_WIDTH
    path.write_text(yaml.safe_dump(document))
    return path


def wide_native_encoder(architecture, path):
    """Write a native classifier checkpoint large enough for native ONNX export.

    Args:
        architecture: Tiny classifier architecture YAML to start from.
        path: Destination checkpoint path.

    Returns:
        Path of the written checkpoint.
    """
    return native_checkpoint(architecture, path, width=EXPORT_WIDTH).path
