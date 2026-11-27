"""Shared Optuna-study and tagged-YAML helpers for the study and config tests.

Registered as a pytest plugin in ``conftest.py``, so its autouse ``quiet_optuna``
fixture applies to the whole suite; the helpers are imported explicitly.
"""

from pathlib import Path

import optuna
import pytest
import yaml

from smartrodent import config_loading


@pytest.fixture(autouse=True)
def quiet_optuna():
    """Keep Optuna's per-trial INFO logging out of test output."""
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(verbosity)


STUDY_NAME = "test-study"


def tuning(**changes) -> dict:
    """Return a minimal valid, deterministic `tuning:` section; every relevant key is explicit.

    Args:
        **changes: Keys overriding the defaults.

    Returns:
        The `tuning:` mapping.
    """
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
    """Read a study's portable report rather than inspecting Optuna internals.

    Args:
        root: Study output directory.

    Returns:
        The parsed ``study.yaml``.
    """
    return yaml.safe_load((Path(root) / "study.yaml").read_text())


def load_tagged_yaml(text: str):
    """Load YAML with the project's custom tags, exactly like every entry point.

    Args:
        text: YAML document text.

    Returns:
        The loaded document.
    """
    return config_loading.parse_yaml(text)


def grid(**changes) -> dict:
    """Return a `tuning:` section that runs every combination of its categoricals once.

    Args:
        **changes: Keys overriding the grid defaults.

    Returns:
        The `tuning:` mapping; n_trials is an upper bound the exhaustive sampler never reaches.
    """
    return tuning(
        **{
            "n_trials": 20,
            "sampler": {"name": "BruteForceSampler", "parameters": {"seed": 0}},
            **changes,
        }
    )
