"""Contracts for run results and ranking by the configured tracking metric."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smartrodent.experiment_results import ExperimentResult, select_best_experiment


def _result(index, value, **updates):
    """Describe a completed run without creating or scanning checkpoint files."""
    output = Path("/nonexistent-smartrodent-results") / f"run_{index:03d}"
    fields = {
        "output_directory": output,
        "last_checkpoint": output / "last.pt",
        "best_checkpoint": output / "best.pt",
        "best_epoch": index + 1,
        "tracked_metric": "balanced_accuracy",
        "tracked_mode": "max",
        "best_value": value,
        "validation_dataset": Path("/nonexistent-smartrodent-data/val"),
        "mlflow_run_id": None,
    }
    return ExperimentResult(**(fields | updates))


@pytest.mark.parametrize("mode,winner", [("min", 1), ("max", 0)])
def test_winner_uses_configured_tracking_direction_and_best_checkpoint(mode, winner):
    """The metric's configured direction, not a loss-name heuristic, chooses the winner."""
    results = [
        _result(0, 0.9, tracked_mode=mode),
        _result(1, 0.2, tracked_mode=mode),
        _result(2, 0.5, tracked_mode=mode),
    ]
    selected = select_best_experiment(results)
    assert selected is results[winner]
    assert selected.best_checkpoint.name == "best.pt"
    assert selected.best_checkpoint != selected.last_checkpoint
    assert selected.best_epoch == winner + 1
    assert selected.output_directory == results[winner].output_directory


def test_ranking_accepts_a_single_validated_result():
    """A one-candidate pipeline needs no directory-scanning special case."""
    result = _result(0, 0.25)
    assert select_best_experiment([result]) is result


@given(
    values=st.lists(st.integers(-20, 20), min_size=1, max_size=30),
    mode=st.sampled_from(["min", "max"]),
)
def test_ranking_respects_direction_and_keeps_first_exact_tie(values, mode):
    """Sweep order deterministically breaks ties, including negative-valued metrics."""
    results = [
        _result(index, value, tracked_mode=mode) for index, value in enumerate(values)
    ]
    best_value = min(values) if mode == "min" else max(values)
    expected_index = values.index(best_value)
    assert select_best_experiment(results) is results[expected_index]


def test_empty_sweep_cannot_produce_a_winner():
    """Missing results fail explicitly rather than silently selecting a path."""
    with pytest.raises(ValueError, match="empty|at least one"):
        select_best_experiment([])


def test_training_only_result_is_not_eligible_for_automatic_ranking():
    """No-validation training is valid standalone but cannot win a pipeline sweep."""
    result = _result(0, 0.1, validation_dataset=None)
    with pytest.raises(ValueError, match="validation"):
        select_best_experiment([result])


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"tracked_metric": "supcon_loss_mean"}, "metric"),
        ({"tracked_mode": "min"}, "mode"),
        ({"validation_dataset": Path("/different-data/val")}, "validation"),
    ],
)
def test_incompatible_tracked_results_cannot_be_ranked_together(updates, message):
    """Runs must report the same tracking metric, direction, and validation split."""
    with pytest.raises(ValueError, match=message):
        select_best_experiment(
            [
                _result(0, 0.4),
                _result(1, 0.2, **updates),
            ]
        )
