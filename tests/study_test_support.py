"""Shared helpers for the Optuna study test modules."""

from pathlib import Path

import optuna
import pytest
import yaml


@pytest.fixture(autouse=True)
def quiet_optuna():
    """Keep Optuna's per-trial INFO logging out of test output."""
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(verbosity)


STUDY_NAME = "test-study"


def tuning(**changes) -> dict:
    """Return a minimal valid, deterministic `tuning:` section; every relevant key is explicit."""
    values = {
        "study_name": STUDY_NAME,
        "storage": None,
        "n_trials": 3,
        "sampler": {"name": "RandomSampler", "parameters": {"seed": 3}},
        "pruner": None,
    }
    values.update(changes)
    return values


def study_report(root: str | Path) -> dict:
    """Read a study's portable report rather than inspecting Optuna internals."""
    return yaml.safe_load((Path(root) / "study.yaml").read_text())
