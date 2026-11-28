"""Console noise filter, log file and the third-party loggers of the training scripts."""

import logging
import re
import warnings

import pytest
from classifier_task_test_support import (
    classifier_document,
    classifier_task,
    fit,
    pipeline_document,
    runner,
    supcon_document,
)
from hypothesis import given
from hypothesis import strategies as st

THIRD_PARTY = ("lightning.pytorch", "lightning.fabric", "py.warnings")


@pytest.fixture(autouse=True)
def lightning_like_logging(package_logger):
    """Start from the state a script finds after importing Lightning.

    ``package_logger`` restores every logger the setup touches afterwards.
    """
    # Lightning installs its own stderr handler and stops propagation when it is
    # imported without root handlers, which is the situation of a script.
    for name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).handlers = [logging.StreamHandler()]
        logging.getLogger(name).propagate = False


def _setup(tmp_path, suppress=()):
    """Configure logging as a script does and return the log file path."""
    from smartrodent.logging_setup import log_to_console_and_file

    tmp_path.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "training.log"
    log_to_console_and_file(log_file, suppress)
    return log_file


def _flush() -> None:
    """Make file handlers write out so the log file can be read."""
    for name in ("smartrodent", *THIRD_PARTY):
        for handler in logging.getLogger(name).handlers:
            handler.flush()


def test_suppressed_message_is_absent_from_console_and_present_in_the_file(
    tmp_path, capsys
):
    """The filter belongs to the console handler only."""
    log_file = _setup(tmp_path, [r"noisy \d+"])
    logger = logging.getLogger("smartrodent.test")
    logger.info("noisy 42")
    logger.info("useful message")
    _flush()

    console = capsys.readouterr().err
    assert "noisy 42" not in console
    assert "useful message" in console
    text = log_file.read_text()
    assert "noisy 42" in text and "useful message" in text


def test_count_line_reports_the_number_of_patterns_on_console_and_in_the_file(
    tmp_path, capsys
):
    """The first console line names how many patterns are active and where the full log is."""
    log_file = _setup(tmp_path, ["alpha", "beta"])
    _flush()

    expected = f"Suppressing 2 console message pattern(s); full output in {log_file}"
    console_lines = capsys.readouterr().err.splitlines()
    assert expected in console_lines[0]
    assert expected in log_file.read_text()


def test_no_count_line_without_patterns(tmp_path, capsys):
    """An empty pattern list is not worth a message."""
    log_file = _setup(tmp_path, [])
    logging.getLogger("smartrodent.test").info("hello")
    _flush()

    assert "Suppressing" not in capsys.readouterr().err
    assert "Suppressing" not in log_file.read_text()


def test_python_warning_matching_a_pattern_is_suppressed_on_console_only(
    tmp_path, capsys
):
    """Warnings flow through logging, hence through the same filter and file."""
    log_file = _setup(tmp_path, ["module\\(s\\) in eval mode"])
    warnings.simplefilter("always")  # pytest restores the filters after the test
    warnings.warn("3 module(s) in eval mode at the start of training", UserWarning)
    warnings.warn("some other warning", UserWarning)
    _flush()

    console = capsys.readouterr().err
    assert "in eval mode" not in console
    assert "some other warning" in console
    text = log_file.read_text()
    assert "in eval mode" in text and "some other warning" in text


def test_lightning_messages_reach_console_and_file_and_obey_the_filter(
    tmp_path, capsys
):
    """A real fit: Lightning's own logger gets both handlers instead of its private one."""
    log_file = _setup(tmp_path, ["max_epochs"])
    stage, _ = runner(tmp_path / "run", epochs=1)
    fit(stage, classifier_task())
    _flush()

    console = capsys.readouterr().err
    text = log_file.read_text()
    assert "max_epochs" not in console
    assert "max_epochs" in text  # `Trainer.fit` stopped: `max_epochs=1` reached.
    assert "GPU available" in console and "GPU available" in text
    assert console.count("GPU available") == 1  # no second, private handler


@given(
    messages=st.lists(st.text(alphabet="abc ", max_size=8), min_size=1, max_size=6),
    patterns=st.lists(st.text(alphabet="abc", min_size=1, max_size=3), max_size=4),
)
def test_a_message_is_suppressed_exactly_when_some_pattern_matches(messages, patterns):
    """The filter keeps a record iff no pattern is found in its message."""
    from smartrodent.logging_setup import ConsoleNoiseFilter

    noise_filter = ConsoleNoiseFilter(patterns)
    for message in messages:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, message, (), None)
        # The patterns contain no regex metacharacters, so plain containment is the oracle.
        matches = any(pattern in message for pattern in patterns)
        assert noise_filter.filter(record) is (not matches)


def test_filter_matches_regular_expressions_against_the_formatted_message():
    """Arguments are merged into the message before a pattern is searched."""
    from smartrodent.logging_setup import ConsoleNoiseFilter

    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "value %d done", (17,), None
    )
    assert ConsoleNoiseFilter([r"value 1\d done"]).filter(record) is False
    assert ConsoleNoiseFilter([r"value 2\d done"]).filter(record) is True


@pytest.mark.parametrize(
    ("schema", "document"),
    [
        (
            "two_stage",
            lambda root: pipeline_document(root, logging={"suppress": ["("]}),
        ),
        ("supcon", lambda root: supcon_document(root, logging={"suppress": ["["]})),
        (
            "classifier",
            lambda root: classifier_document(root, logging={"suppress": ["*a"]}),
        ),
    ],
)
def test_invalid_regular_expression_fails_at_configuration_load(
    tmp_path, schema, document
):
    """A bad pattern must not survive until the middle of a training run."""
    from smartrodent.stage_configurations import (
        ClassifierConfiguration,
        SupConConfiguration,
    )
    from smartrodent.two_stage_training import TwoStageConfiguration

    load = {
        "two_stage": TwoStageConfiguration.from_mapping,
        "supcon": SupConConfiguration.from_mapping,
        "classifier": ClassifierConfiguration.from_mapping,
    }[schema]
    with pytest.raises(ValueError, match="regular expression|regex|pattern"):
        load(document(tmp_path))


@pytest.mark.parametrize("schema", ["two_stage", "supcon", "classifier"])
def test_logging_section_is_accepted_and_defaults_to_no_patterns(tmp_path, schema):
    """The section is optional everywhere and keeps the patterns in order."""
    from smartrodent.stage_configurations import (
        ClassifierConfiguration,
        SupConConfiguration,
    )
    from smartrodent.two_stage_training import TwoStageConfiguration

    load, make = {
        "two_stage": (TwoStageConfiguration.from_mapping, pipeline_document),
        "supcon": (SupConConfiguration.from_mapping, supcon_document),
        "classifier": (ClassifierConfiguration.from_mapping, classifier_document),
    }[schema]
    assert load(make(tmp_path)).logging.suppress == ()
    configured = load(make(tmp_path, logging={"suppress": ["b", "a"]}))
    assert configured.logging.suppress == ("b", "a")
    assert re.compile(configured.logging.suppress[0])


def test_training_run_logging_writes_training_log_into_a_new_output_directory(tmp_path):
    """The scripts' entry point creates the directory and applies the configured patterns."""
    from smartrodent.logging_setup import LoggingConfiguration, log_training_run

    directory = tmp_path / "missing" / "run"
    log_training_run(directory, LoggingConfiguration(suppress=("alpha",)))
    logging.getLogger("smartrodent.test").info("some message")
    _flush()

    text = (directory / "training.log").read_text()
    assert "some message" in text
    assert "Suppressing 1 console message pattern(s)" in text


def test_repeated_setup_replaces_handlers_and_closes_the_old_ones(tmp_path, capsys):
    """A second call neither duplicates output nor leaks the first call's file handle."""
    first_file = _setup(tmp_path / "first")
    first_handlers = list(logging.getLogger("smartrodent").handlers)
    second_file = _setup(tmp_path / "second")
    logging.getLogger("smartrodent.test").info("only once")
    _flush()

    for name in ("smartrodent", *THIRD_PARTY):
        assert len(logging.getLogger(name).handlers) == 2
    assert capsys.readouterr().err.count("only once") == 1
    assert second_file.read_text().count("only once") == 1
    assert "only once" not in first_file.read_text()
    assert all(
        getattr(handler, "stream", None) is None
        for handler in first_handlers
        if isinstance(handler, logging.FileHandler)
    )
