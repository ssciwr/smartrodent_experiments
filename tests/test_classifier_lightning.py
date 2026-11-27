"""Exercise the Lightning classifier: training, loss, reinitialization and export."""

import csv
from pathlib import Path

import pytest
import torch
from classifier_task_test_support import (
    LABEL_NAMES,
    classifier_task,
    encoder_spec,
    fit,
    runner,
    sample_inputs,
    scratch_directory,
    train_supcon,
)
from hypothesis import example, given, settings
from hypothesis import strategies as st
from torch.nn import functional as F


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _metric_rows(path: Path, name: str) -> list[float]:
    """Read the values of one metric from the trainer's CSV log, one per epoch."""
    with path.open(newline="") as stream:
        return [float(row[name]) for row in csv.DictReader(stream) if row[name]]


def test_classifier_trains_and_restores_from_its_best_checkpoint(tmp_path):
    """The best checkpoint rebuilds the classifier once the encoder source is gone."""
    from smartrodent.classifier_task import ClassifierTask
    from smartrodent.experiment_results import Objective

    source = tmp_path / "original.pth"
    torch.manual_seed(3)
    torch.save(classifier_task().encoder.state_dict(), source)
    task = classifier_task(encoder=encoder_spec(source=str(source)))
    initial = task.encoder.linear.weight.detach().clone()
    stage, artifacts = runner(tmp_path / "run", epochs=1)

    result = fit(stage, task)

    assert not torch.equal(initial, task.encoder.linear.weight)
    assert result.best_checkpoint == artifacts.best_checkpoint
    assert result.best_checkpoint.exists() and result.last_checkpoint.exists()
    assert result.objective == Objective("ce_loss_mean", "min") == task.objective
    rows = _metric_rows(artifacts.metrics_csv, "ce_loss_mean")
    assert result.best_value == pytest.approx(min(rows))
    source.unlink()
    restored = ClassifierTask.load_from_checkpoint(
        result.best_checkpoint, map_location="cpu"
    )
    inputs = sample_inputs()
    with torch.inference_mode():
        torch.testing.assert_close(restored.eval()(inputs), task.eval()(inputs))
        assert restored(inputs).shape == (2, len(LABEL_NAMES))


def test_classifier_starts_from_a_supcon_checkpoint_and_restores_without_it(tmp_path):
    """CE accepts a SupCon best.ckpt as encoder and later needs neither file."""
    from smartrodent.classifier_task import ClassifierTask
    from smartrodent.supcon_task import SupConTask

    _, supcon = train_supcon(tmp_path)
    task = classifier_task(encoder={"checkpoint": str(supcon.best_checkpoint)})
    inputs = sample_inputs()
    pretrained = SupConTask.load_from_checkpoint(
        supcon.best_checkpoint, map_location="cpu"
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            task.encoder.eval()(inputs), pretrained.encoder.eval()(inputs)
        )

    stage, _ = runner(tmp_path / "classifier", epochs=1)
    result = fit(stage, task)
    supcon.best_checkpoint.unlink()
    restored = ClassifierTask.load_from_checkpoint(
        result.best_checkpoint, map_location="cpu"
    )
    with torch.inference_mode():
        torch.testing.assert_close(restored.eval()(inputs), task.eval()(inputs))


@settings(max_examples=8, deadline=None)
@example(labels=[0, 1, 2, 0, 1, 2, 0], batch_size=3)
@given(
    labels=st.lists(st.integers(0, 2), min_size=3, max_size=14),
    batch_size=st.integers(1, 14),
)
def test_ce_loss_is_the_whole_split_mean_for_any_batch_size(labels, batch_size):
    """An uneven final batch must not weigh its samples more than the others."""
    from classifier_task_test_support import tensor_data

    with scratch_directory() as directory:
        task = classifier_task(lr=0.0)
        stage, artifacts = runner(Path(directory) / "run", epochs=1)
        fit(stage, task, labels, batch_size=batch_size)
        logged = _metric_rows(artifacts.metrics_csv, "ce_loss_mean")

    targets, images = tensor_data(labels).tensors
    task.eval()
    with torch.inference_mode():
        expected = F.cross_entropy(task(images), targets).item()
    assert logged == [pytest.approx(expected, abs=1e-5)]


def _initializing_weights(path: Path) -> dict[str, torch.Tensor]:
    """Save and return the weights that a configured encoder starts from."""
    from classifier_task_test_support import LayeredEncoder

    torch.manual_seed(5)
    state = LayeredEncoder().state_dict()
    torch.save(state, path)
    return state


def _requested(name: str, prefixes) -> bool:
    """Whether a parameter name is selected by any prefix, on dotted boundaries."""
    return any(name == p or name.startswith(f"{p}.") for p in prefixes)


@settings(max_examples=10, deadline=None)
@given(
    prefixes=st.sets(
        st.sampled_from(
            ["first", "second", "third", "first.weight", "second.bias", "third.weight"]
        ),
        min_size=1,
    )
)
def test_random_reinitialization_alters_only_the_requested_parameters(prefixes):
    """Parameters under a requested prefix are redrawn; every other one is kept.

    A parameter-name prefix such as ``first.weight`` must not redraw its sibling
    ``first.bias``.
    """
    with scratch_directory() as directory:
        source = Path(directory) / "weights.pth"
        original = _initializing_weights(source)
        task = classifier_task(
            encoder=encoder_spec("LayeredEncoder", source=str(source)),
            lr=0.0,
            reinitialization={"prefixes": sorted(prefixes), "source": "random"},
        )
        stage, _ = runner(Path(directory) / "run", epochs=1)
        fit(stage, task)
        trained = task.encoder.state_dict()

    for name, value in original.items():
        if _requested(name, prefixes):
            assert not torch.equal(trained[name], value), name
        else:
            assert torch.equal(trained[name], value), name


def test_native_export_request_is_rejected_when_the_encoder_cannot_export():
    """The request is validated at construction, long before any training."""
    with pytest.raises(ValueError, match="native"):
        classifier_task(native_checkpoint=True)


def test_native_export_request_is_accepted_when_the_encoder_can_export():
    """An encoder with ``to_native`` satisfies the request."""
    task = classifier_task(
        encoder=encoder_spec("NativeTinyEncoder"), native_checkpoint=True
    )
    assert task.encoder.feature_dim == 4
