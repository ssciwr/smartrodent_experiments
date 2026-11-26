"""Entry-point behavior of scripts/train_supcon.py."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml


def test_script_cli_trains_without_notebook_imports(experiment_configuration, tmp_path):
    """The actual script can run a CPU epoch and save a loadable checkpoint."""
    configuration = replace(experiment_configuration, epochs=1)
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Final checkpoint:" in result.stdout
    assert (configuration.output_directory / "last.pt").is_file()
    assert (configuration.output_directory / "best.pt").is_file()


def test_script_cli_supports_multiprocessing_workers(
    experiment_configuration, tmp_path
):
    """Imported datasets and collation work in fresh worker processes."""
    configuration = replace(experiment_configuration, epochs=1, workers=1)
    config_path = tmp_path / "workers.yaml"
    config_path.write_text(yaml.safe_dump(configuration.to_dict()))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_supcon.py"
    result = subprocess.run(
        [sys.executable, str(script), str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (configuration.output_directory / "last.pt").is_file()
    assert (configuration.output_directory / "best.pt").is_file()
