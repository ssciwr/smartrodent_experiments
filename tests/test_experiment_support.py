"""Behavioral contracts for infrastructure shared by SupCon and CE experiments.

Existing SupCon tests provide the integration/regression guard for these shared
components. Seeded pixel comparisons preserve the original augmentation behavior.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st
from PIL import Image


@pytest.fixture
def support():
    """Load shared experiment components for their behavioral tests."""
    from smartrodent import experiment_support

    return experiment_support


@pytest.fixture
def transform_settings():
    """Disable randomness unless a test explicitly asks for augmentation."""
    return {
        "image_size": 8,
        "resize": {"antialias": True},
        "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
        "random_grayscale": {"p": 0.0},
        "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
        "random_horizontal_flip": {"p": 0.0},
        "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
    }


def test_checkpoint_store_preserves_best_while_replacing_last(support, tmp_path):
    """The best snapshot survives later epochs, including optimizer and history state."""
    output = tmp_path / "checkpoints"
    store = support.CheckpointStore(output)
    first = {
        "epoch": 1,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"step": 1},
        "scheduler_state_dict": None,
        "history": [{"epoch": 1, "loss": 0.5}],
    }
    assert store.save(first, is_best=True) == output / "last.pt"
    second = deepcopy(first)
    second["epoch"] = 2
    second["model_state_dict"]["weight"].fill_(2.0)
    second["optimizer_state_dict"]["step"] = 2
    second["history"].append({"epoch": 2, "loss": 0.7})
    assert store.save(second, is_best=False) == output / "last.pt"

    best = torch.load(output / "best.pt", weights_only=True)
    last = torch.load(output / "last.pt", weights_only=True)
    assert best["epoch"] == 1 and last["epoch"] == 2
    torch.testing.assert_close(best["model_state_dict"]["weight"], torch.tensor([1.0]))
    torch.testing.assert_close(last["model_state_dict"]["weight"], torch.tensor([2.0]))
    assert best["optimizer_state_dict"] == {"step": 1}
    assert last["optimizer_state_dict"] == {"step": 2}
    assert best["history"] == first["history"]
    assert last["history"] == second["history"]
    assert best["scheduler_state_dict"] is None
    assert {path.name for path in output.iterdir()} == {"last.pt", "best.pt"}


def test_checkpoint_store_updates_best_without_pruning_unrelated_files(support, tmp_path):
    """A new best replaces the old best, without touching existing user artifacts."""
    sentinel = tmp_path / "notes.txt"
    sentinel.write_text("keep me")
    store = support.CheckpointStore(tmp_path)
    for epoch in range(1, 5):
        store.save({"epoch": epoch}, is_best=epoch in (1, 3))

    assert torch.load(tmp_path / "best.pt", weights_only=True) == {"epoch": 3}
    assert torch.load(tmp_path / "last.pt", weights_only=True) == {"epoch": 4}
    assert sentinel.read_text() == "keep me"
    assert {path.name for path in tmp_path.iterdir()} == {
        "best.pt", "last.pt", "notes.txt"
    }


@pytest.mark.parametrize("has_previous_checkpoint", [False, True])
def test_checkpoint_write_failure_leaves_no_partial_checkpoint(
    support, tmp_path, monkeypatch, has_previous_checkpoint
):
    """Failed serialization cannot truncate a previously completed snapshot."""
    store = support.CheckpointStore(tmp_path)
    if has_previous_checkpoint:
        store.save({"epoch": 1}, is_best=True)
    completed_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def fail_save(snapshot, destination, *args, **kwargs):
        """Write some bytes before simulating a storage-quota exception."""
        if hasattr(destination, "write"):
            destination.write(b"partial checkpoint")
        else:
            from pathlib import Path

            Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("simulated storage quota exceeded")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="quota exceeded"):
        store.save({"epoch": 2}, is_best=True)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == completed_files


def test_checkpoint_publication_failure_preserves_completed_snapshot(
    support, tmp_path, monkeypatch
):
    """A failed atomic publication also removes temporary files and preserves the target."""
    from pathlib import Path

    store = support.CheckpointStore(tmp_path)
    store.save({"epoch": 1}, is_best=True)
    completed_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    original_replace = Path.replace

    def fail_last_publication(source, target):
        """Fail the final replacement rather than the earlier serialization."""
        if Path(target) == tmp_path / "last.pt":
            raise OSError("simulated publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_last_publication)
    with pytest.raises(OSError, match="publication failure"):
        store.save({"epoch": 2}, is_best=False)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == completed_files


def test_best_publication_failure_preserves_previous_best(
    support, tmp_path, monkeypatch
):
    """Best publication is atomic independently of publication of the latest epoch."""
    from pathlib import Path

    store = support.CheckpointStore(tmp_path)
    store.save({"epoch": 1}, is_best=True)
    previous_best = (tmp_path / "best.pt").read_bytes()
    original_replace = Path.replace

    def fail_best_publication(source, target):
        """Interrupt replacement of best, after the new snapshot has been written."""
        if Path(target) == tmp_path / "best.pt":
            raise OSError("simulated best publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_best_publication)
    with pytest.raises(OSError, match="best publication failure"):
        store.save({"epoch": 2}, is_best=True)
    assert (tmp_path / "best.pt").read_bytes() == previous_best
    assert torch.load(tmp_path / "best.pt", weights_only=True) == {"epoch": 1}
    # Each file is atomic; the pair is not required to roll back as a transaction.
    assert torch.load(tmp_path / "last.pt", weights_only=True)["epoch"] in (1, 2)
    assert {path.name for path in tmp_path.iterdir()} == {"last.pt", "best.pt"}


@pytest.mark.parametrize("phase", ["training", "validation"])
@pytest.mark.parametrize("size", [(8, 8), (13, 5)])
def test_image_transforms_preserve_rgb_scale_and_output_shape(
    support, transform_settings, phase, size
):
    """Both pipelines produce CHW float tensors with the existing zero-to-one scaling."""
    transforms = support.ImageTransforms(transform_settings)
    pipeline = getattr(transforms, phase)()
    result = pipeline(Image.new("RGB", size, (255, 128, 0)))
    assert result.shape == (3, 8, 8)
    assert result.dtype == torch.float32
    expected = torch.tensor([1.0, 128 / 255, 0.0])[:, None, None].expand_as(result)
    torch.testing.assert_close(result, expected)


def test_validation_transforms_ignore_training_augmentations(support, transform_settings):
    """Guaranteed training grayscale and flips must not leak into evaluation."""
    transform_settings["random_grayscale"]["p"] = 1.0
    transform_settings["random_horizontal_flip"]["p"] = 1.0
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4] = [255, 0, 0]
    pixels[:, 4:] = [0, 0, 255]
    image = Image.fromarray(pixels)
    transforms = support.ImageTransforms(transform_settings)
    expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
    first = transforms.validation()(image)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(transforms.validation()(image), first)
    augmented = transforms.training()(image)
    torch.testing.assert_close(augmented[0], augmented[1])
    torch.testing.assert_close(augmented[1], augmented[2])
    assert not torch.equal(augmented, first)


def test_training_transform_samples_each_view_and_replays_with_seed(
    support, transform_settings
):
    """Repeated calls draw independent augmentations, reproducible with the same seed."""
    transform_settings["random_horizontal_flip"]["p"] = 0.5
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    image = Image.fromarray(pixels)
    transform = support.ImageTransforms(transform_settings).training()
    # Isolate this test's RNG consumption from other experiment tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = [transform(image) for _ in range(12)]
        torch.manual_seed(42)
        replay = [transform(image) for _ in range(12)]
    assert any(not torch.equal(first[0], item) for item in first[1:])
    for expected, actual in zip(first, replay, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("seed", [11, 42, 91])
def test_training_transform_preserves_legacy_augmentation_pipeline(
    support, transform_settings, seed
):
    """Extracting augmentation construction must preserve the existing pixel outputs."""
    from torchvision.transforms.v2 import (
        Compose,
        PILToTensor,
        RandomAdjustSharpness,
        RandomGrayscale,
        RandomHorizontalFlip,
        RandomPerspective,
        RandomResizedCrop,
        Resize,
        ToDtype,
    )

    settings = transform_settings
    settings["random_adjust_sharpness"]["p"] = 1.0
    settings["random_grayscale"]["p"] = 0.5
    settings["random_perspective"]["p"] = 1.0
    settings["random_horizontal_flip"]["p"] = 0.5
    settings["random_resized_crop"] = {"scale": [0.5, 0.9], "ratio": [0.75, 1.33]}
    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    )
    # This explicit reference is intentionally independent of the shared builder.
    # It pins the old script's operation order and options through the extraction.
    size = settings["image_size"]
    legacy = Compose(
        [
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Resize((size, size), **settings["resize"]),
            RandomAdjustSharpness(**settings["random_adjust_sharpness"]),
            RandomGrayscale(**settings["random_grayscale"]),
            RandomPerspective(**settings["random_perspective"]),
            RandomHorizontalFlip(**settings["random_horizontal_flip"]),
            RandomResizedCrop((size, size), **settings["random_resized_crop"]),
        ]
    )
    extracted = support.ImageTransforms(settings).training()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected = legacy(image)
        torch.manual_seed(seed)
        actual = extracted(image)
    torch.testing.assert_close(actual, expected)


def test_metric_resolution_honors_function_options_without_mutation(support):
    """Named scorers accept label arrays and preserve caller-owned configuration."""
    settings = {
        "accuracy": {"function": "accuracy_score"},
        "macro_f1": {"function": "f1_score", "average": "macro", "zero_division": 0},
    }
    original = deepcopy(settings)
    scorers = support.resolve_validation_metrics(settings)
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 1])
    assert set(scorers) == {"accuracy", "macro_f1"}
    assert scorers["accuracy"](targets, predictions) == 0.75
    assert scorers["macro_f1"](targets, predictions) == pytest.approx(11 / 15)
    assert settings == original


@pytest.mark.parametrize("settings", [None, {}])
def test_metric_resolution_allows_no_optional_metrics(support, settings):
    """Loss-only experiments need no classification scorers."""
    assert support.resolve_validation_metrics(settings) == {}


@pytest.mark.parametrize(
    "settings,error,message",
    [
        ([], TypeError, "validation_metrics"),
        ({"": {"function": "accuracy_score"}}, ValueError, "names"),
        ({"accuracy": []}, TypeError, "accuracy"),
        ({"accuracy": {"function": "not_a_metric"}}, ValueError, "accuracy"),
        ({"accuracy": {"function": "accuracy_score", "invalid": True}}, ValueError, "accuracy"),
        ({"f1": {"function": "f1_score", "average": None}}, ValueError, "f1"),
    ],
)
def test_invalid_metric_specs_fail_before_training(support, settings, error, message):
    """Unknown functions, bad options, and nonscalar metrics have explicit failures."""
    with pytest.raises(error, match=message):
        support.resolve_validation_metrics(settings)


def test_shared_scheduler_builds_independent_native_state(support):
    """Runs sharing a configuration must not share optimizer or scheduler state."""
    settings = {"name": "ReduceLROnPlateau", "parameters": {"factor": 0.5, "patience": 0}}
    original = deepcopy(settings)
    config = support.SchedulerConfiguration.from_mapping(settings)
    first_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    second_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    first = config.build(first_optimizer)
    second = config.build(second_optimizer)
    first.step(1.0)
    first.step(1.0)
    second.step(1.0)
    assert first_optimizer.param_groups[0]["lr"] == 0.0005
    assert second_optimizer.param_groups[0]["lr"] == 0.001
    assert settings == original
    assert support.SchedulerConfiguration.from_mapping(None) is None


@given(
    values=st.lists(st.integers(-20, 20), min_size=1, max_size=30),
    patience=st.integers(1, 5),
    tolerance=st.integers(0, 3),
    comparison=st.sampled_from(["less", "greater"]),
)
def test_shared_early_stopping_compares_against_best_not_previous(
    values, patience, tolerance, comparison
):
    """The extracted stopper retains tolerance and consecutive-miss behavior."""
    from smartrodent import experiment_support

    stopper = experiment_support.EarlyStopping.from_mapping(
        {"patience": patience, "tolerance": tolerance, "comparison": comparison}
    )
    best = values[0]
    misses = 0
    assert not stopper(best)
    for value in values[1:]:
        if comparison == "less":
            improved = value < best - tolerance
        else:
            improved = value > best + tolerance
        if improved:
            best = value
            misses = 0
        else:
            misses += 1
        assert stopper(value) == (misses == patience)
        if misses == patience:
            break


def test_shared_mlflow_configuration_retains_optional_fields_and_tags(support):
    """Shared tracking configuration preserves current names, defaults, and provenance."""
    settings = {
        "tracking_uri": "sqlite:///unused-test.db",
        "experiment_name": "two-stage",
        "registered_model_name": "encoder",
        "model_artifact_name": "species-encoder",
        "champion_alias": "candidate",
        "dataset_revision": "data-revision-123",
    }
    config = support.MlflowConfiguration.from_mapping(settings)
    for name, value in settings.items():
        assert getattr(config, name) == value
    tags = config.run_tags()
    assert tags["dataset_revision"] == "data-revision-123"
    assert tags["git_commit"]
    assert support.MlflowConfiguration.from_mapping(None) is None
    default = support.MlflowConfiguration.from_mapping(
        {"tracking_uri": "sqlite:///unused-test.db", "experiment_name": "two-stage"}
    )
    assert default.registered_model_name is None
    assert default.champion_alias == "champion"
