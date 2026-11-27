"""Compare ONNX Runtime outputs with Torch for both Lightning stages."""

from pathlib import Path

import onnxruntime
import pytest
import torch
from classifier_task_test_support import (
    IMAGE_SIZE,
    sample_inputs,
    train_supcon,
)

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

import hashlib

import numpy as np
import onnx
from classifier_task_test_support import (
    classifier_task,
    fit,
    runner,
)
from torch.nn import functional as F
from ultralytics.utils.export.engine import best_onnx_opset


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def supcon(tmp_path):
    """Train a tiny SupCon stage; return its result."""
    return train_supcon(tmp_path)[1]


@pytest.fixture
def classifier(tmp_path, supcon):
    """Train a classifier on the SupCon encoder; return its result."""
    task = classifier_task(encoder={"checkpoint": str(supcon.best_checkpoint)})
    stage, _ = runner(tmp_path / "classifier", epochs=1)
    return fit(stage, task)


def _exporter():
    """Use the shared exporter with its default (simplified, static) settings."""
    from smartrodent.onnx_export import OnnxExporter

    return OnnxExporter({}, IMAGE_SIZE)


def _run(path) -> np.ndarray:
    """Run an exported file as a recipient would, on the shared sample inputs."""
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    return session.run(None, {name: sample_inputs(1).numpy()})[0]


def _metadata(path) -> dict[str, str]:
    """Read the file's metadata properties as a plain mapping."""
    return {e.key: e.value for e in onnx.load(str(path)).metadata_props}


def _sha256(path) -> str:
    """Return the checksum of the checkpoint an export came from."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _opset(path) -> int:
    """Read the default-domain opset the file declares."""
    versions = {e.domain: e.version for e in onnx.load(str(path)).opset_import}
    return versions.get("", versions.get("ai.onnx"))


def test_encoder_export_matches_the_torch_encoder_features(supcon):
    """The SupCon checkpoint's encoder runs identically in ONNX Runtime."""
    path = _exporter().encoder(supcon.best_checkpoint)

    from smartrodent.supcon_task import SupConTask

    restored = SupConTask.load_from_checkpoint(
        supcon.best_checkpoint, map_location="cpu"
    )
    with torch.inference_mode():
        expected = restored.encoder.eval()(sample_inputs(1))
    assert path.suffix == ".onnx"
    np.testing.assert_allclose(_run(path), expected.numpy(), atol=1e-5, rtol=1e-4)
    assert _opset(path) == best_onnx_opset(onnx)
    metadata = _metadata(path)
    assert metadata["opset"] == str(best_onnx_opset(onnx))
    assert metadata["embedding_dim"] == "4"
    assert metadata["imgsz"] == str([IMAGE_SIZE, IMAGE_SIZE])
    assert metadata["checkpoint_sha256"] == _sha256(supcon.best_checkpoint)
    assert "RGB" in metadata["input"]


def test_classifier_export_matches_the_torch_probabilities(classifier):
    """The classifier file outputs class probabilities, not logits."""
    path = _exporter().classifier(classifier.best_checkpoint)

    from smartrodent.classifier_task import ClassifierTask

    restored = ClassifierTask.load_from_checkpoint(
        classifier.best_checkpoint, map_location="cpu"
    )
    with torch.inference_mode():
        expected = F.softmax(restored.eval()(sample_inputs(1)), dim=1)
    assert path.suffix == ".onnx"
    np.testing.assert_allclose(_run(path), expected.numpy(), atol=1e-5, rtol=1e-4)
    assert _opset(path) == best_onnx_opset(onnx)
    metadata = _metadata(path)
    assert metadata["opset"] == str(best_onnx_opset(onnx))
    assert metadata["imgsz"] == str([IMAGE_SIZE, IMAGE_SIZE])
    assert metadata["checkpoint_sha256"] == _sha256(classifier.best_checkpoint)
    assert "RGB" in metadata["input"]


def test_ultralytics_loads_the_classifier_file_and_predicts_the_torch_probabilities(
    classifier,
):
    """At 32 px, the smallest multiple of the stride, ``YOLO(path.onnx)`` really predicts."""
    from ultralytics import YOLO

    from smartrodent.classifier_task import ClassifierTask
    from smartrodent.onnx_export import OnnxExporter

    size = 32
    path = OnnxExporter({}, size).classifier(classifier.best_checkpoint)
    metadata = _metadata(path)
    assert metadata["task"] == "classify"
    assert metadata["batch"] == "1" and metadata["stride"] == "1"

    images = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(5))
    results = YOLO(str(path), task="classify")(images, verbose=False)
    task = ClassifierTask.load_from_checkpoint(
        classifier.best_checkpoint, map_location="cpu"
    ).eval()
    with torch.inference_mode():
        expected = F.softmax(task(images), dim=1)[0]
    assert results[0].names == {0: "mouse", 1: "rat", 2: "shrew"}
    torch.testing.assert_close(
        results[0].probs.data.float().cpu(), expected, atol=1e-4, rtol=1e-3
    )
