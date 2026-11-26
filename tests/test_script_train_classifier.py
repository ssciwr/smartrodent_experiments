"""Entry-point behavior of scripts/train_classifier.py."""

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from ultralytics import YOLO


@pytest.mark.parametrize("workers", [0, 1])
def test_classifier_cli_trains_a_native_cpu_checkpoint_without_touching_inputs(
    classifier_configuration, tmp_path, workers
):
    """The actual CLI uses native training and cannot create caches inside input splits."""
    config = replace(classifier_configuration, epochs=1, workers=workers)
    path = tmp_path / "cli.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(config.output_directory)
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_classifier.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Final checkpoint:" in completed.stdout
    assert YOLO(
        str(config.output_directory / "weights" / "best.pt"), task="classify"
    ).model.names == {0: "mouse", 1: "rat"}
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_relative_to(config.output_directory)
    }
    assert after == inputs
