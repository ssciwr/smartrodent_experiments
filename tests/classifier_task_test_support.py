"""Tiny CPU encoders, data and runners shared by the classifier and cutover tests.

Encoders are referenced by ``module:attr`` component specs, so this module must stay
importable by name from the tests directory.
"""

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlflow
import optuna
import pytest
import torch
from mlflow import MlflowClient
from mlflow.tracking._model_registry import utils as registry_utils
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODULE = "classifier_task_test_support"
CLASS_NAMES = ("mouse", "rat", "shrew")
LABEL_NAMES = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 8


class TinyEncoder(nn.Module):
    """Pool RGB channels into four features; optionally start from a weights file."""

    feature_dim = 4

    def __init__(self, source: str | None = None, fail: bool = False):
        super().__init__()
        self.linear = nn.Linear(3, self.feature_dim)
        self.fail = fail
        if source is not None:
            self.load_state_dict(torch.load(source, weights_only=True))

    def checkpoint_parameters(self) -> dict:
        """Rebuild from architecture alone once checkpoint weights are available."""
        return {"fail": self.fail}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image; a failing encoder raises instead."""
        if self.fail:
            raise RuntimeError("encoder failure")
        return self.linear(images.mean(dim=(2, 3)))


class NativeTinyEncoder(TinyEncoder):
    """A tiny encoder that offers the optional native export capability."""

    def to_native(self, head: nn.Linear, label_names: dict) -> nn.Module:
        """Return a module that carries the encoder, the head and the label names."""
        native = nn.Sequential(self, head)
        native.names = {index: name for name, index in label_names.items()}
        return native


class LayeredEncoder(nn.Module):
    """Three named linear layers, so prefixes can select some and spare others."""

    feature_dim = 4

    def __init__(self, source: str | None = None, hidden: int = 4):
        super().__init__()
        self.hidden = hidden
        self.first = nn.Linear(3, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.third = nn.Linear(hidden, 4)
        if source is not None:
            self.load_state_dict(torch.load(source, weights_only=True))

    def checkpoint_parameters(self) -> dict:
        """Rebuild from architecture alone once checkpoint weights are available."""
        return {"hidden": self.hidden}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image."""
        features = self.first(images.mean(dim=(2, 3)))
        return self.third(torch.relu(self.second(torch.relu(features))))


def encoder_spec(name: str = "TinyEncoder", **parameters) -> dict:
    """Describe a support-module encoder as a component spec mapping."""
    return {"name": f"{MODULE}:{name}", "parameters": parameters}


def sgd(lr: float = 0.05) -> dict:
    """Describe plain SGD, whose zero learning rate leaves weights untouched."""
    return {"name": "SGD", "parameters": {"lr": lr}}


def classifier_task(*, encoder=None, lr: float = 0.05, **options):
    """Build the planned ClassifierTask with the three-class label mapping."""
    from smartrodent.classifier_task import ClassifierTask

    return ClassifierTask(
        encoder=encoder_spec() if encoder is None else encoder,
        label_names=LABEL_NAMES,
        optimizer=sgd(lr),
        **options,
    )


def tensor_data(labels: list[int], *, seed: int = 0) -> TensorDataset:
    """Return ``(label, image)`` pairs with class-dependent colors, in memory."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(len(labels), 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    targets = torch.tensor(labels)
    for channel in range(3):
        images[targets == channel, channel] += 1.0
    return TensorDataset(targets, images.clamp(max=1.0))


def loaders(
    labels: list[int] | None = None, *, batch_size: int = 4
) -> tuple[DataLoader, DataLoader]:
    """Return ordered training and validation loaders over the same small split."""
    labels = labels if labels is not None else [0, 1, 2] * 3 + [0, 1]
    dataset = tensor_data(labels)
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
    )


def runner(output_directory, *, epochs: int = 2, early_stopping=None, **artifacts):
    """Use the real Lightning runner with quiet CPU settings and bounded epochs."""
    from smartrodent.output_paths import RunArtifacts
    from smartrodent.stage_runner import StageRunner

    run_artifacts = RunArtifacts(Path(output_directory), **artifacts)
    stage = StageRunner(
        run_artifacts,
        trainer={
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": epochs,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        early_stopping=early_stopping,
    )
    return stage, run_artifacts


def fit(stage, task, labels=None, *, batch_size: int = 4, **kwargs):
    """Fit a task on the small split, using it for validation as well."""
    train, validation = loaders(labels, batch_size=batch_size)
    return stage.fit(
        task,
        train_loader=train,
        validation_loader=validation,
        validation_dataset=Path("validation-split"),
        **kwargs,
    )


def sample_inputs(count: int = 2) -> torch.Tensor:
    """Return fixed RGB inputs in [0, 1] for comparing models."""
    generator = torch.Generator().manual_seed(11)
    return torch.rand(count, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)


def write_class_folders(root: Path) -> tuple[Path, Path]:
    """Write tiny class-folder train and val splits under ``root/data``."""
    for split in ("train", "val"):
        for channel, class_name in enumerate(CLASS_NAMES):
            folder = root / "data" / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                color = [10, 20, 30]
                color[channel] = 128 + index * 20
                Image.new("RGB", (IMAGE_SIZE,) * 2, tuple(color)).save(
                    folder / f"{index}.png"
                )
    return root / "data" / "train", root / "data" / "val"


TRAINER = {
    "accelerator": "cpu",
    "devices": 1,
    "max_epochs": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def supcon_document(root: Path, **overrides) -> dict:
    """Assumed standalone SupConConfiguration document over the tiny data."""
    train, validation = write_class_folders(root)
    return {
        "train_dataset": str(train),
        "validation_dataset": str(validation),
        "output_directory": str(root / "supcon"),
        "class_names": list(CLASS_NAMES),
        "transforms": {"image_size": IMAGE_SIZE},
        "encoder": encoder_spec(),
        "optimizer": sgd(0.05),
        "trainer": dict(TRAINER),
        "classes_per_batch": 2,
        "samples_per_class": 2,
        "samples_per_epoch": 8,
        "workers": 0,
        "seed": 42,
        "dropout": 0.0,
        "output_features": 4,
    } | overrides


def classifier_document(root: Path, **overrides) -> dict:
    """Assumed standalone ClassifierConfiguration document over the tiny data."""
    train, validation = write_class_folders(root)
    return {
        "train_dataset": str(train),
        "validation_dataset": str(validation),
        "output_directory": str(root / "classifier"),
        "class_names": list(CLASS_NAMES),
        "transforms": {"image_size": IMAGE_SIZE},
        "encoder": encoder_spec(),
        "optimizer": sgd(0.05),
        "trainer": dict(TRAINER),
        "batch_size": 4,
        "workers": 0,
        "seed": 1,
    } | overrides


def pipeline_document(root: Path, **overrides) -> dict:
    """Assumed two-stage document: shared data, two stage sections, no CE encoder."""
    supcon = supcon_document(root)
    classifier = classifier_document(root)
    shared = {
        key: supcon.pop(key)
        for key in ("train_dataset", "validation_dataset", "class_names")
    }
    for key in ("train_dataset", "validation_dataset", "class_names"):
        classifier.pop(key)
    for stage in (supcon, classifier):
        stage.pop("output_directory")
    classifier.pop("encoder")
    return {
        "output_directory": str(root / "pipeline"),
        "data": shared,
        "supcon": supcon,
        "classifier": classifier,
    } | overrides


def tagged_document(document: dict, **tags: str) -> dict:
    """Load a document with study tags substituted for ``'__NAME__'`` placeholders."""
    import yaml
    from study_test_support import load_tagged_yaml

    text = yaml.safe_dump(document)
    for name, tag in tags.items():
        text = re.sub(rf"'?__{name}__'?", lambda _, tag=tag: tag, text)
    return load_tagged_yaml(text)


def train_supcon(directory: Path, *, encoder: dict | None = None):
    """Train a tiny SupCon task and return its result.

    Uses the reviewed 2b+2c task, runner and sampler-backed data.
    """
    from smartrodent.supcon_task import SupConData, SupConTask
    from smartrodent.transforms import ImageTransforms, TransformSettings

    train, validation = write_class_folders(directory)
    data = SupConData(
        train_dataset=train,
        validation_dataset=validation,
        class_names=CLASS_NAMES,
        transforms=ImageTransforms(TransformSettings(image_size=IMAGE_SIZE)),
        classes_per_batch=2,
        samples_per_class=2,
        samples_per_epoch=8,
        workers=0,
        seed=42,
    )
    task = SupConTask(
        encoder=encoder or encoder_spec(),
        optimizer=sgd(0.05),
        dropout=0.0,
        output_features=4,
    )
    stage, _ = runner(directory / "supcon", epochs=1)
    result = stage.fit(
        task,
        train_loader=data.train_dataloader(),
        validation_loader=data.val_dataloader(),
        validation_dataset=data.validation_dataset.data_dir,
    )
    return task, result


def scratch_directory() -> tempfile.TemporaryDirectory:
    """Create a scratch directory for Hypothesis examples, which cannot use fixtures."""
    return tempfile.TemporaryDirectory()


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
def mlflow_store(tracking_store):
    """The tracking store with registration enabled in its configuration mapping."""
    return SimpleNamespace(
        client=tracking_store.client,
        registry_name=tracking_store.registry_name,
        configuration={
            **tracking_store.configuration,
            "registered_model_name": tracking_store.registry_name,
        },
    )


def native_checkpoint(architecture: Path, path: Path, *, width=8, names=None):
    """Write a standard Ultralytics classifier checkpoint with distinctive head weights.

    Args:
        architecture: Tiny classifier architecture YAML to start from.
        path: Destination checkpoint path.
        width: Output channels of the first convolution.
        names: Class-index-to-name mapping; defaults to mouse/rat.

    Returns:
        Namespace with ``path`` and the reloaded eval ``network``.
    """
    import yaml
    from ultralytics import YOLO

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
    return SimpleNamespace(path=path, network=network)


class ExplodingPruner(optuna.pruners.BasePruner):
    """A pruner whose check fails unexpectedly for the second trial of a study."""

    def prune(self, study, trial) -> bool:
        """Raise for trial 1; never prune otherwise."""
        if trial.number == 1:
            raise RuntimeError("pruner failure")
        return False
