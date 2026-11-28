"""Stage and trial banners, decoded parameters in the logs, and the scripts' log file."""

import hashlib
import json
import logging
import os
import string
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import optuna
import pytest
import yaml
from classifier_task_test_support import pipeline_document
from hypothesis import given, settings
from hypothesis import strategies as st
from study_test_support import load_tagged_yaml, tuning

from smartrodent.experiment_results import ExperimentResult
from smartrodent.hyperparameter_search import (
    HyperparameterStudy,
    StudyConfiguration,
    StudyObjective,
)
from smartrodent.search_space import SearchSpace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class _Collector(logging.Handler):
    """Keep the records of the package logger for inspection."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.messages: list[tuple[int, str]] = []

    def emit(self, record):
        """Store level and merged message."""
        self.messages.append((record.levelno, record.getMessage()))


@contextmanager
def _logged():
    """Collect the package's INFO records; usable inside Hypothesis examples."""
    logger = logging.getLogger("smartrodent")
    collector, level = _Collector(), logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)
    try:
        yield collector.messages
    finally:
        logger.removeHandler(collector)
        logger.setLevel(level)


class _Runner:
    """A trial runner that reports a fixed score, or raises the scripted failure."""

    def __init__(self, validation: Path, *, score=0.5, failures=None):
        self.validation, self.score = validation, score
        self.failures = failures or {}
        self.documents: list[dict] = []

    def __call__(self, document, directory, epoch_callback):
        index = len(self.documents)
        self.documents.append(document)
        if index in self.failures:
            raise self.failures[index]
        directory.mkdir(parents=True)
        best, last = directory / "best.pt", directory / "last.pt"
        best.write_text("best")
        last.write_text("last")
        return ExperimentResult(
            output_directory=directory,
            last_checkpoint=last,
            best_checkpoint=best,
            best_epoch=1,
            best_value=self.score,
            tracked_metric="balanced_accuracy",
            tracked_mode="max",
            validation_dataset=self.validation,
        )


def _study(root: Path, space: str, runner: _Runner, **settings_) -> HyperparameterStudy:
    """Build a real study around the scripted runner."""
    return HyperparameterStudy(
        SearchSpace.from_document(load_tagged_yaml(space)),
        StudyConfiguration.from_mapping(tuning(**settings_)),
        runner,
        output_directory=root / "study",
        objective=StudyObjective(
            tracked_metric="balanced_accuracy",
            tracked_mode="max",
            validation_dataset=runner.validation,
        ),
    )


SPACE = """
lr: !suggest-float {low: 0.001, high: 1.0}
head: !suggest-categorical {choices: [linear, full]}
"""


def test_pipeline_logs_stage_banners_around_the_selection_summary(tmp_path):
    """SupCon banner, the selected encoder's summary, then the classifier banner."""
    from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment

    document = pipeline_document(tmp_path)
    with _logged() as records:
        result = TwoStageExperiment(TwoStageConfiguration.from_mapping(document)).run()

    messages = [message for _, message in records]
    first = next(i for i, m in enumerate(messages) if "Stage 1/2: SupCon" in m)
    last = next(i for i, m in enumerate(messages) if "Stage 2/2: classifier" in m)
    summary = [i for i, m in enumerate(messages) if "sha256" in m]
    assert len(summary) == 1 and first < summary[0] < last
    selected = result.selected_supcon
    digest = hashlib.sha256(selected.best_checkpoint.read_bytes()).hexdigest()
    line = messages[summary[0]]
    assert selected.tracked_metric in line
    assert f"{selected.best_value:.6g}" in line
    assert f"epoch {selected.best_epoch}" in line
    assert digest[:12] in line and digest[:13] not in line


@settings(max_examples=6, deadline=None)
@given(n_trials=st.integers(min_value=1, max_value=4))
def test_every_trial_logs_its_position_in_order(n_trials):
    """``trial k/n`` appears for k = 1..n, once as start and once as finish."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = _Runner(root / "val")
        with _logged() as records:
            _study(root, SPACE, runner, n_trials=n_trials).run()

    messages = [message for _, message in records]
    for k in range(1, n_trials + 1):
        mentions = [i for i, m in enumerate(messages) if f"trial {k}/{n_trials} " in m]
        assert len(mentions) == 2, messages
    starts = [
        next(i for i, m in enumerate(messages) if f"trial {k}/{n_trials} " in m)
        for k in range(1, n_trials + 1)
    ]
    assert starts == sorted(starts)


def test_finished_line_reports_the_score_and_decoded_values_without_indices(tmp_path):
    """The categorical is shown as its choice, never as the sampled index."""
    runner = _Runner(tmp_path / "val", score=0.625)
    with _logged() as records:
        _study(tmp_path, SPACE, runner, n_trials=4).run()

    messages = [message for _, message in records]
    for number, document in enumerate(runner.documents, start=1):
        finished = next(m for m in messages if f"trial {number}/4 finished" in m)
        assert f"head={document['head']}" in finished
        assert "0.625" in finished
        assert "head=0" not in finished and "head=1" not in finished
        started = next(m for m in messages if f"trial {number}/4 started" in m)
        assert f"head={document['head']}" in started


@settings(max_examples=15, deadline=None)
@given(
    choices=st.lists(
        st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=6).map(
            lambda text: f"v_{text}"
        ),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
def test_logged_categorical_value_is_the_choice_at_the_sampled_index(choices):
    """Whatever index is sampled, the log names ``choices[index]``."""
    space = f"x: !suggest-categorical {{choices: {json.dumps(choices)}}}"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runner = _Runner(root / "val")
        grid = {
            "n_trials": len(choices),
            "sampler": {"name": "BruteForceSampler", "parameters": {"seed": 0}},
        }
        with _logged() as records:
            _study(root, space, runner, **grid).run()

    messages = [message for _, message in records]
    assert sorted(d["x"] for d in runner.documents) == sorted(choices)
    for number, document in enumerate(runner.documents, start=1):
        started = next(
            m for m in messages if f"trial {number}/{len(choices)} started" in m
        )
        assert started.endswith(f"x={document['x']}")


def test_optuna_does_not_print_its_own_trial_line_with_raw_indices(tmp_path, capfd):
    """Optuna's verbosity is lowered where the study runs, not at import."""
    optuna.logging.set_verbosity(optuna.logging.INFO)
    _study(tmp_path, SPACE, _Runner(tmp_path / "val"), n_trials=2).run()

    assert "finished with value" not in capfd.readouterr().err


def test_failing_trial_logs_its_error_and_the_exception_propagates(tmp_path):
    """An unexpected failure is visible in the log and still stops the study."""
    runner = _Runner(tmp_path / "val", failures={0: RuntimeError("boom in trial")})
    with _logged() as records, pytest.raises(RuntimeError, match="boom in trial"):
        _study(tmp_path, SPACE, runner, n_trials=3).run()

    failures = [m for level, m in records if level >= logging.ERROR]
    assert len(failures) == 1
    assert "trial 1/3" in failures[0] and "boom in trial" in failures[0]


def _run_script(script: str, document: dict, tmp_path: Path):
    """Run a training script on a document and return the finished process."""
    path = tmp_path / f"{script}.yaml"
    path.write_text(yaml.safe_dump(document))
    return subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py"), str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(Path(__file__).parent),
        },
    )


def test_two_stage_script_writes_the_log_file_and_filters_only_the_console(tmp_path):
    """The script sets logging up from the configuration; the file keeps everything."""
    document = pipeline_document(tmp_path, logging={"suppress": ["Stage 1/2"]})
    result = _run_script("train_two_stage", document, tmp_path)

    log_file = Path(document["output_directory"]) / "training.log"
    text = log_file.read_text()
    assert "Stage 1/2: SupCon" in text and "Stage 2/2: classifier" in text
    assert "Stage 1/2" not in result.stderr
    assert "Stage 2/2: classifier" in result.stderr
    assert (
        f"Suppressing 1 console message pattern(s); full output in {log_file}"
        in result.stderr
    )
