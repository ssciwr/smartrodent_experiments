"""Exercise SupCon data, training, and recovery on a tiny CPU encoder."""

import csv
import math
import statistics
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from pytorch_metric_learning.losses import SupConLoss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyEncoder(nn.Module):
    """Pool RGB channels and optionally initialize them from a local weights file."""

    feature_dim = 4

    def __init__(self, source: str | None = None, nonfinite: bool = False):
        super().__init__()
        self.linear = nn.Linear(3, self.feature_dim)
        self.nonfinite = nonfinite
        if source is not None:
            self.load_state_dict(torch.load(source, weights_only=True))

    def checkpoint_parameters(self) -> dict:
        """Return the constructor arguments needed when checkpoint weights are available."""
        return {"nonfinite": self.nonfinite}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image, with optional injected nonfinite output."""
        features = self.linear(images.mean(dim=(2, 3)))
        if self.nonfinite:
            return features * math.nan
        return features


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _build_data(
    root: Path, *, validation_counts=None, validation=True, **loader_options
):
    """Build class-folder splits; validation image counts per class are adjustable."""
    from smartrodent.supcon_task import SupConData
    from smartrodent.transforms import ImageTransforms, TransformSettings

    class_names = ("mouse", "rat", "shrew")
    counts = {
        "train": {name: 4 for name in class_names},
        "val": validation_counts or {name: 4 for name in class_names},
    }
    for split, per_class in counts.items():
        for channel, class_name in enumerate(class_names):
            folder = root / split / class_name
            folder.mkdir(parents=True)
            for index in range(per_class[class_name]):
                color = [10, 20, 30]
                color[channel] = 128 + index * 20
                Image.new("RGB", (8, 8), tuple(color)).save(folder / f"{index}.png")
    return SupConData(
        train_dataset=root / "train",
        validation_dataset=root / "val" if validation else None,
        class_names=class_names,
        transforms=ImageTransforms(TransformSettings(image_size=8)),
        classes_per_batch=2,
        samples_per_class=2,
        samples_per_epoch=8,
        seed=42,
        **{"workers": 0, **loader_options},
    )


@pytest.fixture
def data(tmp_path):
    """Build two class-folder splits with three classes and several batches."""
    return _build_data(tmp_path)


class BatchNormEncoder(nn.Module):
    """Normalize pooled channels before a linear layer, for freeze-policy tests."""

    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(3)
        self.linear = nn.Linear(3, self.feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image."""
        return self.linear(self.norm(images.mean(dim=(2, 3))))


def _task(*, lr=0.05, encoder="TinyEncoder", freeze=None, **encoder_parameters):
    """Construct the production task with a test encoder and plain SGD."""
    from smartrodent.supcon_task import SupConTask

    return SupConTask(
        encoder={"name": f"{__name__}:{encoder}", "parameters": encoder_parameters},
        freeze=freeze,
        optimizer={"name": "SGD", "parameters": {"lr": lr}},
        dropout=0.0,
        output_features=4,
    )


def _runner(
    output_directory, *, epochs=2, early_stopping=None, sanity_steps=2, **trainer
):
    """Use the real Lightning runner with quiet CPU settings and bounded epochs."""
    from smartrodent.output_paths import RunArtifacts
    from smartrodent.stage_runner import StageRunner

    artifacts = RunArtifacts(output_directory)
    runner = StageRunner(
        artifacts,
        trainer={
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": epochs,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": sanity_steps,
        }
        | trainer,
        early_stopping=early_stopping,
    )
    return runner, artifacts


def _fit(runner, task, data, **kwargs):
    """Run the task with the data fixture's real samplers and collator."""
    return runner.fit(
        task,
        train_loader=data.train_dataloader(),
        validation_loader=data.val_dataloader(),
        validation_dataset=(
            None
            if data.validation_dataset is None
            else data.validation_dataset.data_dir
        ),
        **kwargs,
    )


def _loss_rows(path: Path) -> list[dict[str, str]]:
    """Read rows containing the validation objective from the trainer's CSV log."""
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["supcon_loss_mean"]]


def test_worker_loaders_carry_the_configured_prefetch_factor(tmp_path):
    """Multiprocess loaders honour the requested workers and prefetch depth."""
    data = _build_data(tmp_path, workers=1, prefetch_factor=3)

    for loader in (data.train_dataloader(), data.val_dataloader()):
        assert loader.num_workers == 1
        assert loader.prefetch_factor == 3


def test_supcon_keeps_pk_sampling_and_a_fixed_single_view_validation_cohort(data):
    """Training batches pair two views of P-K samples; validation never changes."""
    train_batches = list(data.train_dataloader())
    assert len(train_batches) == 2
    for labels, images in train_batches:
        assert images.shape == (8, 3, 8, 8)
        torch.testing.assert_close(labels[:4], labels[4:])
        _, counts = labels.unique(return_counts=True)
        torch.testing.assert_close(counts, torch.tensor([4, 4]))

    cohort = list(data.val_dataloader())
    assert len(cohort) == 3
    for (labels, images), (same_labels, same_images) in zip(
        cohort, data.val_dataloader(), strict=True
    ):
        assert images.shape == (4, 3, 8, 8)
        _, counts = labels.unique(return_counts=True)
        torch.testing.assert_close(counts, torch.tensor([2, 2]))
        torch.testing.assert_close(labels, same_labels)
        torch.testing.assert_close(images, same_images)


def test_supcon_records_all_validation_loss_statistics(data, tmp_path):
    """Logged min, mean, std and max match the loss over the validation cohort."""
    runner, artifacts = _runner(tmp_path / "stats")
    task = _task()
    initial = task.encoder.linear.weight.detach().clone()
    result = _fit(runner, task, data)
    assert not torch.equal(initial, task.encoder.linear.weight)
    rows = _loss_rows(artifacts.metrics_csv)
    assert len(rows) == 2
    criterion = SupConLoss(temperature=0.1)
    task.eval()
    with torch.inference_mode():
        batch_losses = [
            criterion(task(images), labels).item()
            for labels, images in data.val_dataloader()
        ]
    expected = {
        "min": min(batch_losses),
        "mean": statistics.mean(batch_losses),
        "std": statistics.stdev(batch_losses),
        "max": max(batch_losses),
    }
    for statistic, value in expected.items():
        assert float(rows[-1][f"supcon_loss_{statistic}"]) == pytest.approx(
            value, abs=1e-6
        )
    assert result.tracked_metric == "supcon_loss_mean"
    assert result.tracked_mode == "min"
    assert result.best_value == pytest.approx(
        min(float(row["supcon_loss_mean"]) for row in rows)
    )


ANCHOR_STATISTICS = {
    "supcon_anchor_loss_min": lambda values: values.min(),
    "supcon_anchor_loss_median": lambda values: np.quantile(values, 0.5),
    "supcon_anchor_loss_p95": lambda values: np.quantile(values, 0.95),
    "supcon_anchor_loss_max": lambda values: values.max(),
}


def _anchor_losses_by_definition(embeddings, labels, temperature=0.1) -> list[float]:
    """Return the textbook SupCon loss of every anchor that has a positive.

    Anchor i: -mean over its positives p of [s_ip - logsumexp over a != i of s_ia],
    with s the temperature-scaled cosine similarities of unit-length embeddings. A
    batch of a single class has no negatives; like the objective, which counts such a
    batch as 0, it contributes no anchors.
    """
    if len(set(labels)) < 2:
        return []
    scores = embeddings @ embeddings.T / temperature
    losses = []
    for anchor in range(len(labels)):
        others = [i for i in range(len(labels)) if i != anchor]
        positives = [i for i in others if labels[i] == labels[anchor]]
        if positives:
            log_denominator = torch.logsumexp(scores[anchor, others], dim=0)
            losses.append(-(scores[anchor, positives] - log_denominator).mean().item())
    return losses


def _expected_anchor_statistics(losses: list[float]) -> dict[str, float]:
    """Return each per-anchor statistic of the losses; NaN when there are no anchors."""
    if not losses:
        return {name: float("nan") for name in ANCHOR_STATISTICS}
    values = np.asarray(losses, dtype=np.float64)
    return {name: float(f(values)) for name, f in ANCHOR_STATISTICS.items()}


@contextmanager
def _fit_one_epoch_with_validation_counts(counts: tuple[int, int, int]):
    """Fit one epoch on a validation cohort of the given per-class sizes.

    Yields the trained task, the data and the CSV row of the validation epoch. The
    images exist only inside the context: loaders read them lazily.
    """
    class_names = ("mouse", "rat", "shrew")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = _build_data(
            root / "data", validation_counts=dict(zip(class_names, counts))
        )
        runner, artifacts = _runner(root / "run", epochs=1, sanity_steps=0)
        task = _task()
        _fit(runner, task, data)
        (row,) = _loss_rows(artifacts.metrics_csv)
        yield task, data, row


_VALIDATION_COUNTS = st.tuples(*[st.integers(2, 5)] * 3)


@settings(max_examples=6, deadline=None)
@given(counts=_VALIDATION_COUNTS)
def test_supcon_records_per_anchor_validation_loss_statistics(counts):
    """The logged statistics describe single anchors across the whole cohort."""
    with _fit_one_epoch_with_validation_counts(counts) as (task, data, row):
        task.eval()
        losses = []
        with torch.inference_mode():
            for labels, images in data.val_dataloader():
                losses += _anchor_losses_by_definition(task(images), labels.tolist())
    for name, value in _expected_anchor_statistics(losses).items():
        assert float(row[name]) == pytest.approx(value, abs=1e-5), name


@pytest.mark.parametrize(
    "labels",
    [
        [0, 0, 1],  # one anchor has no positive: it is not part of the statistics
        [0, 1, 2],  # no anchor has a positive: the statistics are undefined
        [0, 0, 0],  # one class only: every anchor has positives but none has a negative
    ],
)
def test_supcon_anchors_without_a_usable_loss_are_left_out_of_the_statistics(
    data, tmp_path, labels
):
    """Lone-class anchors, or a batch without negatives, have no loss to count."""
    generator = torch.Generator().manual_seed(7)
    images = torch.rand(len(labels), 3, 8, 8, generator=generator)
    validation = DataLoader(
        TensorDataset(torch.tensor(labels), images), batch_size=len(labels)
    )
    runner, artifacts = _runner(tmp_path / "lone", epochs=1, sanity_steps=0)
    task = _task()

    runner.fit(
        task,
        train_loader=data.train_dataloader(),
        validation_loader=validation,
        validation_dataset=data.validation_dataset.data_dir,
    )

    task.eval()
    with torch.inference_mode():
        expected = _expected_anchor_statistics(
            _anchor_losses_by_definition(task(images), labels)
        )
    (row,) = _loss_rows(artifacts.metrics_csv)
    for name, value in expected.items():
        assert float(row[name]) == pytest.approx(value, abs=1e-5, nan_ok=True), name


def test_supcon_anchor_statistics_work_with_half_precision_training(data, tmp_path):
    """A half-precision trainer yields half-precision losses, which torch.quantile rejects."""
    runner, artifacts = _runner(
        tmp_path / "bf16", epochs=1, sanity_steps=0, precision="bf16-true"
    )

    _fit(runner, _task(), data)

    (row,) = _loss_rows(artifacts.metrics_csv)
    assert all(math.isfinite(float(row[name])) for name in ANCHOR_STATISTICS)


def test_supcon_stops_on_tied_mean_and_keeps_the_earliest_best_checkpoint(
    data, tmp_path
):
    """An unchanging model ties on every validation pass and stops after one miss."""
    runner, artifacts = _runner(
        tmp_path / "ties", epochs=5, early_stopping={"patience": 1, "min_delta": 0.0}
    )
    epochs = []
    result = _fit(
        runner,
        _task(lr=0.0),
        data,
        epoch_callback=lambda epoch, _value: epochs.append(epoch),
    )

    assert epochs == [1, 2]
    assert result.best_epoch == 1
    assert result.best_checkpoint == artifacts.best_checkpoint
    assert result.last_checkpoint == artifacts.last_checkpoint
    assert torch.load(result.best_checkpoint, weights_only=True)["epoch"] == 0
    assert torch.load(result.last_checkpoint, weights_only=True)["epoch"] == 1


def test_best_supcon_checkpoint_restores_features_after_source_is_removed(
    data, tmp_path
):
    """Stored construction parameters suffice once checkpoint state is available."""
    from smartrodent.supcon_task import SupConTask

    original = tmp_path / "original.pth"
    torch.save(TinyEncoder().state_dict(), original)
    runner, _ = _runner(tmp_path / "restore", epochs=1)
    task = _task(source=str(original))
    result = _fit(runner, task, data)
    inputs = torch.linspace(0, 1, steps=3 * 8 * 8).reshape(1, 3, 8, 8)
    expected = task.encoder.eval()(inputs).detach()
    original.unlink()

    restored = SupConTask.load_from_checkpoint(
        result.best_checkpoint, map_location="cpu"
    )
    with torch.inference_mode():
        torch.testing.assert_close(restored.encoder.eval()(inputs), expected)


def test_nonfinite_supcon_loss_raises_before_checkpointing(data, tmp_path):
    """A broken training batch cannot become a publishable training result."""
    runner, artifacts = _runner(tmp_path / "nonfinite", sanity_steps=0)

    with pytest.raises(FloatingPointError):
        _fit(runner, _task(nonfinite=True), data)

    assert not artifacts.best_checkpoint.exists()


def test_supcon_observes_completed_epochs_once_and_propagates_optuna_pruning(
    data, tmp_path
):
    """Lightning's sanity validation does not report a trial step to Optuna."""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.ThresholdPruner(upper=-1, n_warmup_steps=2),
    )
    trial = study.ask()
    seen = []

    def observe(epoch, value):
        """Report the stage objective until the configured pruner stops the trial."""
        seen.append((epoch, value))
        trial.report(value, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    runner, _ = _runner(tmp_path / "pruned", epochs=5)
    with pytest.raises(optuna.TrialPruned):
        _fit(runner, _task(), data, epoch_callback=observe)

    frozen = study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    assert [epoch for epoch, _ in seen] == [1, 2]
    assert frozen.intermediate_values == dict(seen)
    assert all(math.isfinite(value) for _, value in seen)


@pytest.mark.parametrize("batchnorm", ["eval", "adapt"])
def test_task_freezes_prefixed_parameters_and_follows_the_batchnorm_policy(
    data, tmp_path, batchnorm
):
    """Fitting leaves frozen BatchNorm parameters fixed; its statistics follow policy."""
    task = _task(
        encoder="BatchNormEncoder",
        freeze={"prefixes": ["norm"], "batchnorm": batchnorm},
    )
    norm, linear = task.encoder.norm, task.encoder.linear
    before = {
        "weight": norm.weight.detach().clone(),
        "bias": norm.bias.detach().clone(),
        "mean": norm.running_mean.clone(),
        "variance": norm.running_var.clone(),
        "linear": linear.weight.detach().clone(),
    }

    runner, _ = _runner(tmp_path / "bn", epochs=1)
    _fit(runner, task, data)

    torch.testing.assert_close(norm.weight, before["weight"])
    torch.testing.assert_close(norm.bias, before["bias"])
    assert not torch.equal(linear.weight, before["linear"])
    if batchnorm == "eval":
        torch.testing.assert_close(norm.running_mean, before["mean"])
        torch.testing.assert_close(norm.running_var, before["variance"])
    elif batchnorm == "adapt":
        assert not torch.equal(norm.running_mean, before["mean"])
        assert not torch.equal(norm.running_var, before["variance"])


def test_task_reapplies_the_batchnorm_policy_whenever_training_resumes():
    """Switching back to training mode keeps frozen statistics layers in eval mode."""
    task = _task(
        encoder="BatchNormEncoder", freeze={"prefixes": ["norm"], "batchnorm": "eval"}
    )
    task.eval()
    task.train()

    assert task.encoder.linear.training
    assert not task.encoder.norm.training


def test_training_without_validation_checkpoints_on_the_training_loss(tmp_path):
    """The training mean becomes the objective when no validation split exists."""
    data = _build_data(tmp_path, validation=False)
    runner, artifacts = _runner(tmp_path / "train_only", sanity_steps=0)
    epochs = []

    result = _fit(
        runner, _task(), data, epoch_callback=lambda epoch, _v: epochs.append(epoch)
    )

    with artifacts.metrics_csv.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["supcon_loss_mean"]]
    assert len(rows) == 2
    assert not any(r.get("train_supcon_loss_mean") for r in rows)
    assert epochs == [1, 2]
    assert artifacts.best_checkpoint.exists()
    assert artifacts.last_checkpoint.exists()
    assert result.validation_dataset is None
    assert result.best_value == pytest.approx(
        min(float(r["supcon_loss_mean"]) for r in rows)
    )


@pytest.fixture
def tracking(tmp_path):
    """Point MLflow at a temporary SQLite store and leave no run active."""
    import mlflow

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    # Earlier tests may leave a process-wide experiment ID absent from this store.
    mlflow.set_experiment("stage-runner")
    try:
        yield mlflow
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous)


def test_stage_logs_to_the_callers_active_mlflow_run_and_leaves_it_open(
    data, tmp_path, tracking
):
    """Metrics reach the caller's run, which stays active for later publication."""
    with tracking.start_run() as run:
        runner, _ = _runner(tmp_path / "tracked", epochs=2)
        result = _fit(runner, _task(), data, mlflow_run_id=run.info.run_id)

        assert tracking.active_run().info.run_id == run.info.run_id
        history = tracking.MlflowClient().get_metric_history(
            run.info.run_id, "supcon_loss_mean"
        )
    assert result.mlflow_run_id == run.info.run_id
    assert len(history) == 2


@pytest.mark.parametrize("active", [True, False])
def test_stage_rejects_an_mlflow_run_that_is_not_the_active_one(
    data, tmp_path, tracking, active
):
    """Tracking requires the caller's own active run."""
    runner, _ = _runner(tmp_path / "mismatch")
    if active:
        tracking.start_run()
    with pytest.raises(ValueError, match="active MLflow run"):
        _fit(runner, _task(), data, mlflow_run_id="not-the-active-run")


@pytest.mark.parametrize("key", ["callbacks", "logger", "default_root_dir"])
def test_stage_runner_rejects_trainer_settings_it_owns(tmp_path, key):
    """Callbacks, logger and root directory are chosen by the runner alone."""
    with pytest.raises(ValueError, match=key):
        _runner(tmp_path / "reserved", **{key: None})


def test_stage_runner_requires_an_explicit_epoch_limit(tmp_path):
    """Lightning's silent 1000-epoch default is never used."""
    from smartrodent.output_paths import RunArtifacts
    from smartrodent.stage_runner import StageRunner

    with pytest.raises(ValueError, match="max_epochs"):
        StageRunner(RunArtifacts(tmp_path), trainer={"accelerator": "cpu"})


def test_a_run_that_trains_no_epoch_is_an_error(data, tmp_path):
    """Zero epochs yield no checkpoint, which must not look like a result."""
    runner, _ = _runner(tmp_path / "empty", epochs=0, sanity_steps=0)

    with pytest.raises(RuntimeError, match="checkpoint"):
        _fit(runner, _task(), data)


@pytest.mark.parametrize(
    "counts,message",
    [
        ({"mouse": 1, "rat": 1, "shrew": 1}, "full batch"),
        ({"mouse": 4, "rat": 0, "shrew": 0}, "too few nonempty classes"),
    ],
)
def test_validation_split_must_fill_a_batch_of_enough_classes(
    tmp_path, counts, message
):
    """An unusable validation cohort fails before training starts."""
    with pytest.raises(ValueError, match=message):
        _build_data(tmp_path, validation_counts=counts)
