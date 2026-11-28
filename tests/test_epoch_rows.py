"""The CSV metrics log has one complete row per epoch, with or without validation."""

import csv
import math
from pathlib import Path

import pytest
import torch
from classifier_task_test_support import (
    CLASS_NAMES,
    IMAGE_SIZE,
    LABEL_NAMES,
    classifier_task,
    encoder_spec,
    fit,
    loaders,
    runner,
    scratch_directory,
    sgd,
    write_class_folders,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from smartrodent.classifier_task import ClassifierTask
from smartrodent.supcon_task import SupConTask


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read the header and the rows of a metrics CSV."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames), rows


def _train_classifier(
    directory: Path,
    *,
    epochs: int,
    validate: bool,
    sanity_steps: int = 0,
    task_class=None,
    **trainer_options,
) -> Path:
    """Fit a tiny classifier (of ``task_class``) and return its metrics CSV."""
    stage, artifacts = runner(directory / "run", epochs=epochs)
    stage.trainer.update(num_sanity_val_steps=sanity_steps, **trainer_options)
    if task_class is None:
        task = classifier_task()
    else:
        task = task_class(
            encoder=encoder_spec(), label_names=LABEL_NAMES, optimizer=sgd(0.05)
        )
    if validate:
        fit(stage, task)
    else:
        train, _ = loaders()
        stage.fit(task, train_loader=train)
    return artifacts.metrics_csv


def _train_supcon(
    directory: Path,
    *,
    epochs: int,
    validate: bool,
    sanity_steps: int = 0,
    task_class=None,
    **trainer_options,
) -> Path:
    """Fit a tiny SupCon task (of ``task_class``) and return its metrics CSV."""
    from smartrodent.supcon_task import SupConData, SupConTask
    from smartrodent.transforms import ImageTransforms, TransformSettings

    train, validation = write_class_folders(directory)
    data = SupConData(
        train_dataset=train,
        validation_dataset=validation if validate else None,
        class_names=CLASS_NAMES,
        transforms=ImageTransforms(TransformSettings(image_size=IMAGE_SIZE)),
        classes_per_batch=2,
        samples_per_class=2,
        samples_per_epoch=8,
        workers=0,
        seed=42,
    )
    task = (task_class or SupConTask)(
        encoder=encoder_spec(), optimizer=sgd(0.05), dropout=0.0, output_features=4
    )
    stage, artifacts = runner(directory / "supcon", epochs=epochs)
    stage.trainer.update(num_sanity_val_steps=sanity_steps, **trainer_options)
    stage.fit(
        task,
        train_loader=data.train_dataloader(),
        validation_loader=data.val_dataloader(),
        validation_dataset=data.validation_dataset.data_dir if validate else None,
    )
    return artifacts.metrics_csv


def _recording(base, covered):
    """Subclass ``base`` to note every logged metric and, as validation starts, how
    much training the pending train mean covers (``covered(task)``)."""
    events: list[tuple] = []

    class Recording(base):
        """Record what the logged train means are made of."""

        def on_validation_epoch_end(self):
            """Note the accumulated training size, unless this is the sanity pass."""
            if not self.trainer.sanity_checking:
                events.append(("covered", covered(self)))
            super().on_validation_epoch_end()

        def log(self, name, value, *args, **kwargs):
            """Note the metric and its value, then log as usual."""
            events.append(("log", name, float(value)))
            return super().log(name, value, *args, **kwargs)

    return Recording, events


TRAINERS = {"classifier": _train_classifier, "supcon": _train_supcon}
TRAIN_COLUMN = {"classifier": "train_ce_loss_mean", "supcon": "train_supcon_loss_mean"}
OBJECTIVE_COLUMN = {"classifier": "ce_loss_mean", "supcon": "supcon_loss_mean"}
LOSS_STATISTIC_COLUMNS = {
    "classifier": {
        "ce_loss_min",
        "ce_loss_median",
        "ce_loss_p95",
        "ce_loss_max",
        "ce_loss_correct_mean",
        "ce_loss_wrong_mean",
    },
    "supcon": {
        "supcon_anchor_loss_min",
        "supcon_anchor_loss_median",
        "supcon_anchor_loss_p95",
        "supcon_anchor_loss_max",
    },
}


@pytest.mark.parametrize("task", ["classifier", "supcon"])
def test_validated_run_logs_train_and_validation_metrics_in_one_row_per_epoch(
    tmp_path, task
):
    """Train loss shares the validation row instead of adding a half-empty one."""
    path = TRAINERS[task](tmp_path, epochs=2, validate=True)

    columns, rows = _rows(path)
    assert len(rows) == 2
    assert all(value != "" for row in rows for value in row.values())
    expected = {TRAIN_COLUMN[task], OBJECTIVE_COLUMN[task]}
    assert expected | LOSS_STATISTIC_COLUMNS[task] <= set(columns)


@pytest.mark.parametrize("task", ["classifier", "supcon"])
def test_run_without_validation_logs_one_row_per_epoch(tmp_path, task):
    """The objective is the training mean, logged once at the end of each epoch."""
    path = TRAINERS[task](tmp_path, epochs=2, validate=False)

    columns, rows = _rows(path)
    assert len(rows) == 2
    assert all(value != "" for row in rows for value in row.values())
    assert OBJECTIVE_COLUMN[task] in columns
    assert TRAIN_COLUMN[task] not in columns


@pytest.mark.parametrize("task", ["classifier", "supcon"])
@pytest.mark.parametrize("validate", [True, False])
@settings(max_examples=3, deadline=None)
@given(epochs=st.integers(min_value=1, max_value=3))
def test_rows_equal_epochs(task, validate, epochs):
    """Whatever the epoch count, the log has exactly that many rows."""
    with scratch_directory() as directory:
        path = TRAINERS[task](Path(directory), epochs=epochs, validate=validate)
        _, rows = _rows(path)

    assert len(rows) == epochs


@pytest.mark.parametrize("task", ["classifier", "supcon"])
def test_sanity_check_before_training_does_not_disturb_the_epoch_rows(tmp_path, task):
    """The sanity pass validates before any training loss exists to be logged."""
    path = TRAINERS[task](tmp_path, epochs=2, validate=True, sanity_steps=1)

    columns, rows = _rows(path)
    assert len(rows) == 2
    assert all(value != "" for row in rows for value in row.values())
    assert TRAIN_COLUMN[task] in columns


@pytest.mark.parametrize(
    ("task", "covered", "per_epoch"),
    [
        # 11 training samples per epoch; 2 batches of 4 source images per epoch.
        ("classifier", lambda task: task._training_count, 11),
        ("supcon", lambda task: len(task._training_losses), 2),
    ],
)
def test_logged_train_mean_covers_one_epoch_when_validation_skips_epochs(
    tmp_path, task, covered, per_epoch
):
    """With validation every second epoch, a train mean still spans one epoch only."""
    recording, events = _recording(
        {"classifier": ClassifierTask, "supcon": SupConTask}[task], covered
    )
    TRAINERS[task](
        tmp_path,
        epochs=4,
        validate=True,
        task_class=recording,
        check_val_every_n_epoch=2,
    )

    # Epochs without validation log no train row; the others cover just their own data.
    assert [event[1] for event in events if event[0] == "covered"] == [per_epoch] * 2


def test_classifier_sanity_check_logs_no_train_metric_before_training(tmp_path):
    """The sanity pass has no training batches, so it must not log an empty mean (NaN)."""
    recording, events = _recording(ClassifierTask, lambda task: task._training_count)
    _train_classifier(
        tmp_path, epochs=1, validate=True, sanity_steps=1, task_class=recording
    )

    train_values = [
        e[2] for e in events if e[0] == "log" and e[1] == "train_ce_loss_mean"
    ]
    assert len(train_values) == 1
    assert all(math.isfinite(value) for value in train_values)
