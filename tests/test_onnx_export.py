"""ONNX export contracts for trained CE classifiers and SupCon encoders.

The classifier export is Ultralytics' native one (class probabilities); the
SupCon export is the encoder's pooled features. Both are checked numerically
against the PyTorch checkpoint they were exported from.
"""

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import onnxslim
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from training_test_support import wide_native_encoder
from ultralytics import YOLO
from ultralytics.utils.export.engine import best_onnx_opset

from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
)
from smartrodent.experiment_support import ImageTransforms
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration

SUPPORTED_EXPORT_KEYS = {"dynamic", "simplify"}


# A recipient's script: only the ONNX file, its metadata, and the documented input
# contract (RGB, float32 in [0, 1], N x 3 x S x S). Training libraries are blocked,
# so the file must be self-contained.
RECIPIENT_SCRIPT = """
import ast, importlib.abc, json, sys

class BlockTrainingLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "torchvision", "ultralytics", "smartrodent"}:
            raise ImportError(f"recipient has no {name}")
        return None

sys.meta_path.insert(0, BlockTrainingLibraries())

import numpy as np
import onnxruntime
from PIL import Image

model, *images = sys.argv[1:]
session = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
metadata = session.get_modelmeta().custom_metadata_map
height, width = ast.literal_eval(metadata["imgsz"])
# One image per call: without dynamic=True the exported batch size is fixed at 1.
output = np.concatenate([
    session.run(None, {session.get_inputs()[0].name: np.asarray(
        Image.open(image).convert("RGB").resize((width, height), Image.BILINEAR), dtype=np.float32,
    ).transpose(2, 0, 1)[None] / 255})[0]
    for image in images
])
result = {"output": output.tolist()}
if "names" in metadata:
    names = ast.literal_eval(metadata["names"])
    result["labels"] = [names[int(index)] for index in output.argmax(axis=1)]
print(json.dumps(result))
"""


@pytest.fixture
def classifier_case(classifier_configuration, classifier_assets, tmp_path):
    """Run the CE classifier trainer; its native export sits next to best.pt."""
    torch.set_num_threads(2)
    encoder = wide_native_encoder(
        classifier_assets.architecture, tmp_path / "wide-cls.pt"
    )
    base = replace(
        classifier_configuration,
        epochs=1,
        initialization={
            "encoder": {"source": "ultralytics", "checkpoint": str(encoder)},
            "head": {"replace": "linear", "source": "random"},
        },
    )
    return SimpleNamespace(
        base=base,
        image_size=base.transforms["image_size"],
        run=lambda config: ClassifierExperiment(config).run_with_result(),
        expected=lambda config: config.output_directory / "weights" / "best.onnx",
        rebuild=lambda values: ClassifierConfiguration.from_mapping(values),
    )


@pytest.fixture
def supcon_case(experiment_configuration):
    """Run the SupCon trainer; its encoder export sits next to best.pt."""
    torch.set_num_threads(2)
    base = replace(experiment_configuration, epochs=1)
    return SimpleNamespace(
        base=base,
        image_size=base.transforms["image_size"],
        run=lambda config: SupConExperiment(config).run_with_result(),
        expected=lambda config: config.output_directory / "best.onnx",
        rebuild=lambda values: TrainingConfiguration.from_mapping(values),
    )


@pytest.fixture(params=["classifier_case", "supcon_case"])
def export_case(request):
    """Apply the shared export contract to both trainers."""
    return request.getfixturevalue(request.param)


def _session(path):
    """Open an exported file with the runtime a recipient would use."""
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _metadata(path):
    """Read ONNX metadata as a plain mapping."""
    return {entry.key: entry.value for entry in onnx.load(str(path)).metadata_props}


def _images(batch, size):
    """Use preprocessed RGB inputs in [0, 1], the documented input contract."""
    return np.random.default_rng(3).random((batch, 3, size, size), dtype=np.float32)


def test_classifier_export_matches_native_probabilities(classifier_case):
    """The native export computes the same class probabilities as best.pt."""
    config = replace(classifier_case.base, onnx_export={})
    result = classifier_case.run(config)
    assert result.onnx_model == classifier_case.expected(config)
    session = _session(result.onnx_model)
    assert [node.name for node in session.get_inputs()] == ["images"]
    images = _images(1, classifier_case.image_size)
    (probabilities,) = session.run(None, {"images": images})
    reference = YOLO(str(result.best_checkpoint), task="classify").model.float().eval()
    with torch.inference_mode():
        expected, _ = reference(torch.from_numpy(images))
    assert probabilities.shape == (1, 2)
    np.testing.assert_allclose(probabilities, expected.numpy(), rtol=0, atol=1e-5)
    metadata = _metadata(result.onnx_model)
    assert ast.literal_eval(metadata["names"]) == {0: "mouse", 1: "rat"}
    assert ast.literal_eval(metadata["imgsz"]) == [classifier_case.image_size] * 2


def test_supcon_export_matches_pooled_encoder_features(supcon_case):
    """The SupCon export returns the pooled encoder features, not the projection."""
    config = replace(supcon_case.base, onnx_export={})
    result = supcon_case.run(config)
    assert result.onnx_model == supcon_case.expected(config)
    session = _session(result.onnx_model)
    assert [node.name for node in session.get_inputs()] == ["images"]
    assert [node.name for node in session.get_outputs()] == ["embeddings"]
    images = _images(1, supcon_case.image_size)
    (embeddings,) = session.run(None, {"images": images})
    model = YoloSupConModel.from_checkpoint(result.best_checkpoint)
    try:
        with torch.inference_mode():
            model(torch.from_numpy(images))
            expected = model.embeddings
        embedding_dim = model.head.linear.in_features
    finally:
        model.finalize()
    assert embeddings.shape == (1, embedding_dim)
    np.testing.assert_allclose(embeddings, expected.numpy(), rtol=1e-5, atol=1e-5)
    metadata = _metadata(result.onnx_model)
    assert int(metadata["embedding_dim"]) == embedding_dim
    # Same "[S, S]" format as Ultralytics' native classifier export.
    assert ast.literal_eval(metadata["imgsz"]) == [supcon_case.image_size] * 2
    assert (
        metadata["checkpoint_sha256"]
        == hashlib.sha256(result.best_checkpoint.read_bytes()).hexdigest()
    )
    assert "RGB" in metadata["input"] and "[0, 1]" in metadata["input"]


def test_without_export_no_onnx_file_is_written(export_case):
    """Export is opt-in: the default configuration writes no ONNX file."""
    config = export_case.base
    assert config.onnx_export is None
    result = export_case.run(config)
    assert result.onnx_model is None
    assert not list(config.output_directory.rglob("*.onnx"))


def test_dynamic_export_accepts_other_batch_and_image_sizes(export_case):
    """dynamic=True keeps batch, height and width open, like Ultralytics' native export."""
    config = replace(export_case.base, onnx_export={"dynamic": True})
    result = export_case.run(config)
    session = _session(result.onnx_model)
    for size in (export_case.image_size, export_case.image_size + 8):
        (output,) = session.run(None, {"images": _images(3, size)})
        assert output.shape[0] == 3


def test_both_exports_use_the_ultralytics_default_opset(export_case):
    """Classifier and encoder files declare the same opset, Ultralytics' default for this torch."""
    config = replace(export_case.base, onnx_export={"simplify": False})
    result = export_case.run(config)
    versions = {
        entry.domain: entry.version
        for entry in onnx.load(str(result.onnx_model)).opset_import
    }
    assert versions.get("", versions.get("ai.onnx")) == best_onnx_opset(onnx)


# Arbitrary unknown keys are covered by the property test below; here one native
# exporter option that is deliberately not exposed, non-bool values for both
# supported options, and a non-mapping value.
@pytest.mark.parametrize(
    "export", [{"opset": 17}, {"dynamic": "yes"}, {"simplify": 1}, "onnx"]
)
def test_invalid_export_settings_fail_at_configuration(export_case, export):
    """Unsupported options or values fail when the configuration is built, before training."""
    replace(
        export_case.base, onnx_export={}
    )  # The field exists and accepts valid settings.
    with pytest.raises(ValueError, match="onnx_export"):
        replace(export_case.base, onnx_export=export)


def test_export_failure_propagates_instead_of_reporting_a_complete_run(
    export_case, monkeypatch
):
    """A run whose requested export failed is never reported as complete."""

    def fail(*args, **kwargs):
        raise RuntimeError("simulated ONNX export failure")

    monkeypatch.setattr(torch.onnx, "export", fail)
    config = replace(export_case.base, onnx_export={})
    with pytest.raises(RuntimeError, match="export failure"):
        export_case.run(config)
    assert not export_case.expected(config).exists()


def test_simplifier_failure_propagates_and_removes_the_written_file(
    export_case, monkeypatch
):
    """simplify=True never silently yields an unsimplified file, and no partial file remains."""

    def fail(*args, **kwargs):
        # The file is already written when simplification runs.
        assert export_case.expected(config).is_file()
        raise RuntimeError("simulated simplifier failure")

    monkeypatch.setattr(onnxslim, "slim", fail)
    config = replace(export_case.base, onnx_export={"simplify": True})
    with pytest.raises(RuntimeError, match="simplifier failure"):
        export_case.run(config)
    assert not export_case.expected(config).exists()


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    dynamic=st.one_of(st.none(), st.booleans()),
    simplify=st.one_of(st.none(), st.booleans()),
)
def test_valid_export_settings_survive_serialization(export_case, dynamic, simplify):
    """Every supported option combination round-trips through the saved configuration."""
    export = {
        key: value
        for key, value in (("dynamic", dynamic), ("simplify", simplify))
        if value is not None
    }
    config = replace(export_case.base, onnx_export=export)
    assert export_case.rebuild(config.to_dict()).onnx_export == export
    assert export_case.rebuild(export_case.base.to_dict()).onnx_export is None


@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    key=st.text(min_size=1, max_size=12).filter(
        lambda key: key not in SUPPORTED_EXPORT_KEYS
    )
)
def test_unknown_export_options_are_rejected(export_case, key):
    """Only the documented options are accepted; misspellings never pass silently."""
    replace(
        export_case.base, onnx_export={}
    )  # The field exists and accepts valid settings.
    with pytest.raises(ValueError, match="onnx_export"):
        replace(export_case.base, onnx_export={key: True})


def _recipient_images(directory, size, count=3):
    """Write unseen, non-uniform image files at the documented input size."""
    directory.mkdir()
    rng = np.random.default_rng(11)
    paths = []
    for index in range(count):
        path = directory / f"{index}.png"
        Image.fromarray(
            rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB"
        ).save(path)
        paths.append(path)
    return paths


def _run_as_recipient(model, images):
    """Run the exported file in a fresh process that cannot import any training library."""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", RECIPIENT_SCRIPT, str(model), *map(str, images)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _validation_batch(config, images):
    """Preprocess files exactly as the trainers' validation pipeline does."""
    transform = ImageTransforms(config.transforms).validation()
    return torch.stack(
        [transform(Image.open(image).convert("RGB")) for image in images]
    )


def test_recipient_can_classify_image_files_with_only_the_exported_classifier(
    classifier_case, tmp_path
):
    """End to end: train, export, then classify files elsewhere with onnxruntime alone."""
    config = replace(classifier_case.base, onnx_export={})
    result = classifier_case.run(config)
    images = _recipient_images(tmp_path / "recipient", classifier_case.image_size)
    received = _run_as_recipient(result.onnx_model, images)
    reference = YOLO(str(result.best_checkpoint), task="classify").model.float().eval()
    with torch.inference_mode():
        expected, _ = reference(_validation_batch(config, images))
    probabilities = np.array(received["output"], dtype=np.float32)
    np.testing.assert_allclose(probabilities, expected.numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    assert received["labels"] == [
        reference.names[int(index)] for index in expected.argmax(dim=1)
    ]


def test_recipient_can_embed_image_files_with_only_the_exported_encoder(
    supcon_case, tmp_path
):
    """End to end: train, export, then embed files elsewhere with onnxruntime alone."""
    config = replace(supcon_case.base, onnx_export={})
    result = supcon_case.run(config)
    images = _recipient_images(tmp_path / "recipient", supcon_case.image_size)
    received = _run_as_recipient(result.onnx_model, images)
    model = YoloSupConModel.from_checkpoint(result.best_checkpoint)
    try:
        with torch.inference_mode():
            model(_validation_batch(config, images))
            expected = model.embeddings
    finally:
        model.finalize()
    embeddings = np.array(received["output"], dtype=np.float32)
    assert embeddings.shape == tuple(expected.shape) and np.isfinite(embeddings).all()
    np.testing.assert_allclose(embeddings, expected.numpy(), rtol=1e-5, atol=1e-5)
