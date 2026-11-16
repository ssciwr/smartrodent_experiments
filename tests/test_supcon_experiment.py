"""Exercise the standalone experiment, library modules, and portable checkpoints."""

from dataclasses import replace
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from scripts import train_supcon
from scripts.train_supcon import SupConExperiment, TrainingConfiguration
from smartrodent.supcon_datasets import ImageDataset, MultiViewImageDataset, collate_views
from smartrodent.supcon_model import YoloSupConModel


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
    architecture.write_text(yaml.safe_dump({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
        "head": [[-1, 1, "Classify", [2]]],
    }))
    return TrainingConfiguration(
        train_dataset=dataset, output_directory=tmp_path / "outputs",
        class_names=("rat", "mouse"), model_name=str(architecture),
        epochs=2, image_size=16, classes_per_batch=2, samples_per_class=2,
        samples_per_epoch=4, workers=0, device="cpu", output_features=4,
    )


def test_library_datasets_pickle_and_multiview_batching(experiment_configuration):
    """Imported datasets preserve image labels through pickling and collation."""
    config = experiment_configuration
    transform = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])
    single = ImageDataset(config.train_dataset, config.class_names, transform)
    assert len(single) == 6 and single[0][0] == 0
    dataset = MultiViewImageDataset(config.train_dataset, config.class_names, transform)
    restored = pickle.loads(pickle.dumps(dataset))
    labels, images = collate_views([restored[0], restored[3]])
    assert labels.tolist() == [0, 1, 0, 1]
    assert images.shape == (4, 3, 16, 16)
    assert 0 <= images.min() <= images.max() <= 1


def test_wrapper_propagates_gradients_and_respects_modes(experiment_configuration):
    """SupCon updates the encoder and projector, leaving the classifier outside the loss."""
    torch.set_num_threads(2)
    model = YoloSupConModel(experiment_configuration.model_name, num_classes=2, output_features=4)
    model.requires_grad_(True)
    model.train()
    assert model.training and model.full_model.model.training and model.projector.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    encoder = next(model.full_model.model.model[0].parameters())
    before = encoder.detach().clone()
    logits, embeddings = model(torch.rand(4, 3, 16, 16))
    assert logits.shape == (4, 2) and embeddings.shape == (4, 4)
    np.testing.assert_allclose(embeddings.detach().norm(dim=1).numpy(), 1, atol=1e-5)
    SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
    assert encoder.grad is not None and model.projector[-1].weight.grad is not None
    assert model.head.linear.weight.grad is None
    optimizer.step()
    assert not torch.equal(before, encoder)
    model.eval()
    assert not model.training and not model.full_model.model.training and not model.projector.training
    model.finalize()
    assert model.embeddings is None


def test_experiment_saves_reusable_checkpoints(experiment_configuration):
    """A fresh session can restore architecture, class mapping, and exact outputs."""
    torch.set_num_threads(2)
    config = experiment_configuration
    result = SupConExperiment(config).run()
    assert result.name == "epoch_0002.pt"
    assert (config.output_directory / "epoch_0001.pt").is_file()
    checkpoint = torch.load(result, weights_only=True)
    assert checkpoint["epoch"] == 2 and len(checkpoint["history"]) == 2
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["label_names"] == {"mouse": 0, "rat": 1}
    assert all(np.isfinite(row["std"]) for row in checkpoint["history"])
    history = yaml.safe_load((config.output_directory / "history.yaml").read_text())
    assert len(history) == 2 and history[-1]["epoch"] == 2
    # Rebuilding must use the embedded architecture, not reopen original weights/YAML.
    Path(config.model_name).unlink()
    first = YoloSupConModel.from_checkpoint(result)
    second = YoloSupConModel.from_checkpoint(result)
    assert first.label_names == checkpoint["label_names"]
    assert first.training_configuration["image_size"] == 16
    assert not first.training and not first.projector.training
    for name, parameter in first.state_dict().items():
        assert torch.equal(parameter, checkpoint["model_state_dict"][name])
    images = torch.rand(4, 3, 16, 16)
    with torch.inference_mode():
        first_logits, first_features = first(images)
        second_logits, second_features = second(images)
    torch.testing.assert_close(first_logits[0], second_logits[0])
    torch.testing.assert_close(first_features, second_features)
    first.finalize()
    second.finalize()


def test_freezing_is_applied_after_gradient_enable(experiment_configuration, monkeypatch):
    """Requested frozen layers stay fixed while the projection branch learns."""
    torch.set_num_threads(2)
    created = []
    original = train_supcon.YoloSupConModel

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append((model, next(model.full_model.model.model[0].parameters()).detach().clone()))
        return model

    monkeypatch.setattr(train_supcon, "YoloSupConModel", create_model)
    config = replace(experiment_configuration, epochs=1, freeze=(0,))
    checkpoint = SupConExperiment(config).run()
    model, before = created[0]
    parameter = next(model.full_model.model.model[0].parameters())
    assert not parameter.requires_grad and torch.equal(parameter, before)
    assert torch.load(checkpoint, weights_only=True)["model_configuration"]["freeze"] == [0]
    assert model.embeddings is None and not model.training


def test_experiment_finalizes_after_loss_failure(experiment_configuration, monkeypatch):
    """Exceptions propagate while the training feature hook and modes are finalized."""
    torch.set_num_threads(2)
    created = []
    original = train_supcon.YoloSupConModel

    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(model)
        return model

    class FailingLoss(torch.nn.Module):
        """Simulate a failure after encoder and projector forward execution."""

        def forward(self, embeddings, labels):
            """Raise an explicit training error."""
            raise RuntimeError("deliberate loss failure")

    monkeypatch.setattr(train_supcon, "YoloSupConModel", create_model)
    monkeypatch.setattr(train_supcon, "SupConLoss", lambda **kwargs: FailingLoss())
    with pytest.raises(RuntimeError, match="deliberate"):
        SupConExperiment(experiment_configuration).run()
    assert created[0].embeddings is None and not created[0].training


def test_experiment_rejects_nonfinite_epoch_loss(experiment_configuration, monkeypatch):
    """A nonfinite objective cannot be recorded as a successful checkpoint."""
    torch.set_num_threads(2)

    class NonfiniteLoss(torch.nn.Module):
        """Return a graph-connected NaN loss."""

        def forward(self, embeddings, labels):
            """Keep a gradient path while deliberately making the value invalid."""
            return embeddings.sum() * float("nan")

    monkeypatch.setattr(train_supcon, "SupConLoss", lambda **kwargs: NonfiniteLoss())
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        SupConExperiment(experiment_configuration).run()
    assert not list(experiment_configuration.output_directory.glob("*.pt"))


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"epochs": 0}, "positive integers"),
        ({"epochs": True}, "positive integers"),
        ({"workers": -1}, "workers"),
        ({"class_names": "mouse"}, "sequence"),
        ({"class_names": ("", "rat")}, "nonempty strings"),
        ({"class_names": ("rat", "rat")}, "unique"),
        ({"classes_per_batch": 1}, "between 2"),
        ({"classes_per_batch": 3}, "between 2"),
        ({"samples_per_epoch": 1}, "full batch"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"temperature": 0}, "temperature"),
        ({"dropout": 2}, "dropout"),
        ({"freeze": (-1,)}, "freeze"),
    ],
)
def test_invalid_training_configuration_fails(experiment_configuration, updates, message):
    """Invalid experiment settings fail before image loading and training."""
    with pytest.raises(ValueError, match=message):
        replace(experiment_configuration, **updates)


def test_output_directory_cannot_be_inside_input(experiment_configuration):
    """The experiment never writes checkpoints into its input images."""
    with pytest.raises(ValueError, match="outside"):
        replace(experiment_configuration, output_directory=experiment_configuration.train_dataset / "outputs")


def test_missing_classes_fail_before_training(experiment_configuration):
    """P-K batches cannot silently omit an empty represented class."""
    empty = experiment_configuration.train_dataset / "rat"
    for path in empty.glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="nonempty classes"):
        SupConExperiment(experiment_configuration).run()


@pytest.mark.parametrize("document", ["[]", "class_names: mouse\ntrain_dataset: images\noutput_directory: outputs\n"])
def test_yaml_requires_mapping_and_class_list(tmp_path, document):
    """Malformed YAML structure fails without model construction."""
    path = tmp_path / "config.yaml"
    path.write_text(document)
    with pytest.raises(ValueError):
        TrainingConfiguration.from_yaml(path)


def test_invalid_checkpoint_metadata_fails(tmp_path):
    """Old or unrelated tensor files are rejected with a format explanation."""
    path = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="train_supcon.py"):
        YoloSupConModel.from_checkpoint(path)


def test_script_cli_trains_without_notebook_imports(experiment_configuration, tmp_path):
    """The actual script can run a CPU epoch and save a loadable checkpoint."""
    configuration = replace(experiment_configuration, epochs=1)
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Final checkpoint:" in result.stdout
    assert (configuration.output_directory / "epoch_0001.pt").is_file()


def test_script_cli_supports_multiprocessing_workers(experiment_configuration, tmp_path):
    """Imported datasets and collation work in fresh worker processes."""
    configuration = replace(experiment_configuration, epochs=1, workers=1)
    config_path = tmp_path / "workers.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        capture_output=True, text=True, timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (configuration.output_directory / "epoch_0001.pt").is_file()
