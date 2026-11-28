"""Shared fixtures for the whole suite.

Fixtures used by more than one test module live here or in the support plugins
below; test modules request them by parameter name and never import them, so no
test module depends on another test module. Plugin autouse fixtures (such as
``quiet_optuna``) therefore apply to every test.
"""

import logging
import os

import pytest
from PIL import Image

# The suite is CPU-only, as in CI. Torch's optimizer step initializes CUDA whenever a
# GPU is visible, even for CPU tensors, so hide it for every test and subprocess.
# An explicit CUDA_VISIBLE_DEVICES (for opt-in GPU markers) takes precedence.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytest_plugins = [
    "bundle_test_support",
    "classifier_task_test_support",
    "http_test_support",
    "study_test_support",
]


@pytest.fixture
def images(tmp_path):
    """Create two differently colored images with identical base filenames."""
    paths = []
    for species, color in [("mouse", "red"), ("shrew", "blue")]:
        directory = tmp_path / "input" / species
        directory.mkdir(parents=True)
        path = directory / "observation.png"
        image = Image.new("RGB", (20, 10), "black")
        image.paste(color, (5, 2, 15, 8))
        image.save(path)
        paths.append(path)
    return paths


@pytest.fixture
def package_logger():
    """The ``smartrodent`` logger; it and the third-party loggers that entry points
    configure are restored afterwards, as is Python's warnings capture."""
    names = ("smartrodent", "lightning.pytorch", "lightning.fabric", "py.warnings")
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in names
    }
    yield logging.getLogger("smartrodent")
    logging.captureWarnings(False)
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        for handler in set(logger.handlers) - set(handlers):
            handler.close()
        logger.handlers, logger.level, logger.propagate = handlers, level, propagate
