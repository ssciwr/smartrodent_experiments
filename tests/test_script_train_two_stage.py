"""Entry-point behavior of scripts/train_two_stage.py."""

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml
from ultralytics import YOLO


def test_two_stage_cli_runs_the_actual_cpu_pipeline(pipeline_document, tmp_path):
    """The new thin entry point must execute both stages without downloads or test data."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"] = 0.001
    document["supcon"]["epochs"] = document["classifier"]["epochs"] = 1
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(document))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_two_stage.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Final classifier checkpoint:" in completed.stdout
    root = Path(document["output_directory"])
    assert (root / "supcon" / "run_001" / "best.pt").is_file()
    assert (root / "selection.yaml").is_file()
    restored = YOLO(
        str(root / "classifier" / "weights" / "best.pt"), task="classify"
    ).model
    assert restored.names == {0: "mouse", 1: "rat"}
