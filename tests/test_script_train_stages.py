"""Entry-point behavior of the three training scripts on tiny CPU documents."""

import os
import subprocess
import sys
from pathlib import Path

import yaml
from classifier_task_test_support import (
    classifier_document,
    pipeline_document,
    supcon_document,
    train_supcon,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script: str, document: dict, tmp_path: Path) -> str:
    """Run one script on a document and return its output."""
    path = tmp_path / f"{script}.yaml"
    path.write_text(yaml.safe_dump(document))
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py"), str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            # The tiny test encoders are components of the tests directory.
            "PYTHONPATH": str(Path(__file__).parent),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_supcon_script_trains_and_writes_checkpoints(tmp_path):
    """The script runs a CPU epoch and leaves best and last checkpoints."""
    document = supcon_document(tmp_path)
    output = _run("train_supcon", document, tmp_path)

    assert "Final checkpoint:" in output
    directory = Path(document["output_directory"])
    assert (directory / "best.ckpt").is_file() and (directory / "last.ckpt").is_file()


def test_classifier_script_trains_on_a_supcon_checkpoint(tmp_path):
    """The classifier script restores its encoder from a SupCon best.ckpt."""
    supcon = train_supcon(tmp_path)[1]
    document = classifier_document(
        tmp_path, encoder={"checkpoint": str(supcon.best_checkpoint)}
    )
    output = _run("train_classifier", document, tmp_path)

    assert "Final checkpoint:" in output
    assert (Path(document["output_directory"]) / "best.ckpt").is_file()


def test_two_stage_script_runs_the_pipeline(tmp_path):
    """Both stages run and the pipeline reports the selected classifier."""
    document = pipeline_document(tmp_path)
    output = _run("train_two_stage", document, tmp_path)

    assert "Selected classifier checkpoint:" in output
    root = Path(document["output_directory"])
    assert (root / "selection.yaml").is_file()
    assert (root / "classifier" / "best.ckpt").is_file()
