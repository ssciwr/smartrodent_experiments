"""Classifier experiments: metrics, export and run paths."""

import onnxruntime
import pytest
import torch
from classifier_task_test_support import (
    CLASS_NAMES,
    classifier_document,
    sample_inputs,
)

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.stage_configurations import (
    ClassifierConfiguration,
)

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_classifier_tracks_a_sklearn_metric_and_exports_onnx(tmp_path):
    """A score alias is maximized, and the dynamic export accepts other batch sizes."""
    configuration = ClassifierConfiguration.from_mapping(
        classifier_document(
            tmp_path,
            validation_metrics=METRICS,
            tracked_metric="f1",
            onnx_export=ONNX,
            early_stopping={"patience": 1},
        )
    )
    result = ClassifierExperiment(configuration).run_with_result()

    assert (result.tracked_metric, result.tracked_mode) == ("f1", "max")
    assert configuration.metric_definition == {
        "function": "f1_score",
        "average": "macro",
    }
    session = onnxruntime.InferenceSession(
        str(result.onnx_model), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    probabilities = session.run(None, {name: sample_inputs(3).numpy()})[0]
    assert probabilities.shape == (3, len(CLASS_NAMES))


def test_classifier_run_returns_the_last_checkpoint(tmp_path):
    """``run`` keeps the checkpoint-path API of the standalone scripts."""
    configuration = ClassifierConfiguration.from_mapping(classifier_document(tmp_path))
    assert ClassifierExperiment(configuration).run().name == "last.ckpt"


def test_classifier_without_validation_tracks_the_training_loss(tmp_path):
    """The training mean is the objective when no split is configured."""
    document = classifier_document(tmp_path, validation_dataset=None)
    result = ClassifierExperiment(
        ClassifierConfiguration.from_mapping(document)
    ).run_with_result()
    assert result.validation_dataset is None


def test_an_empty_class_directory_is_rejected(tmp_path):
    """Every requested class needs training images."""
    document = classifier_document(tmp_path)
    for image in (tmp_path / "data" / "train" / CLASS_NAMES[0]).iterdir():
        image.unlink()
    with pytest.raises(ValueError, match="nonempty"):
        ClassifierExperiment(ClassifierConfiguration.from_mapping(document))
