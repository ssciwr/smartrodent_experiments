"""Behavioral contract for direct SpeciesNet batch inference.

Unit tests replace only the external SpeciesNet model. The opt-in integration
test uses the installed SDK and real weights. All source images are temporary.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import os
import shutil

import pytest
from hypothesis import given, strategies as st
from PIL import Image

from smartrodent import yolo_dataset_creation
from smartrodent.yolo_dataset_creation import _SpeciesNetDatasetMixin
from speciesnet_test_support import SpeciesNetDetectorStub, SpeciesNetStub, detection


def configured_mixin(*, model=None, model_name=None):
    """Explicitly initialize the mixin without involving dataset creators."""
    mixin = _SpeciesNetDatasetMixin()
    mixin.initialize_speciesnet(model=model, model_name=model_name)
    return mixin


def infer(paths, predictions, *, threshold=0.5):
    """Exercise the public interface with an external model substitute."""
    return configured_mixin(model=SpeciesNetStub(predictions)).infer_batch(
        paths,
        batch_size=2,
        confidence_threshold=threshold,
        allowed_classes=("animal",),
        iou_threshold=0.45,
    )


def test_batch_results_preserve_full_paths_and_extract_independent_crops(images):
    """Same-named source images retain distinct metadata and correct crop pixels."""
    predictions = [
        {"filepath": str(path), "detections": [detection(0.9)]} for path in images
    ]

    result = infer(images, predictions)

    assert set(result) == {str(path) for path in images}
    for path, color in zip(images, [(255, 0, 0), (0, 0, 255)], strict=True):
        records = result[str(path)]["detections"]
        assert len(records) == 1
        record = records[0]
        assert record["bbox"] == [0.25, 0.2, 0.5, 0.6]
        assert record["confidence"] == 0.9
        assert record["label"] == "animal"
        assert record["detection_index"] == 0
        assert isinstance(record["crop"], Image.Image)
        assert record["crop"].size == (10, 6)
        assert all(
            record["crop"].getpixel((x, y)) == color
            for x in range(10)
            for y in range(6)
        )


def test_confidence_boundary_and_allowed_labels_are_respected(images):
    """The confidence boundary is inclusive and non-animal boxes are excluded."""
    predictions = [
        {
            "filepath": str(images[0]),
            "detections": [
                detection(0.49),
                detection(0.5),
                detection(0.99, label="human"),
            ],
        }
    ]

    result = infer(images[:1], predictions)

    records = result[str(images[0])]["detections"]
    assert len(records) == 1
    assert records[0]["confidence"] == 0.5
    assert records[0]["detection_index"] == 1


def test_nms_keeps_strongest_overlap_and_separate_detection(images):
    """Overlapping boxes are suppressed without losing separate animals."""
    predictions = [
        {
            "filepath": str(images[0]),
            "detections": [
                detection(0.7),
                detection(0.95),
                detection(0.8, bbox=(0.0, 0.0, 0.1, 0.1)),
            ],
        }
    ]

    result = infer(images[:1], predictions)

    records = result[str(images[0])]["detections"]
    assert {record["detection_index"] for record in records} == {1, 2}
    assert sorted(record["confidence"] for record in records) == [0.8, 0.95]


@pytest.mark.parametrize("detections", [[], [detection(0.1)]])
def test_images_without_accepted_detections_remain_in_results(images, detections):
    """An empty result is explicit rather than silently dropping its source image."""
    predictions = [{"filepath": str(images[0]), "detections": detections}]

    result = infer(images[:1], predictions)

    assert result == {str(images[0]): {"detections": []}}


@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_confidence_filter_matches_inclusive_threshold(confidence, threshold):
    """Every valid confidence/threshold pair obeys the same acceptance rule."""
    # Each generated example owns its files; no function-scoped fixture state leaks
    # between Hypothesis examples.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "source.png"
        Image.new("RGB", (20, 10), "red").save(path)
        predictions = [{"filepath": str(path), "detections": [detection(confidence)]}]

        result = infer([path], predictions, threshold=threshold)

        assert len(result[str(path)]["detections"]) == int(confidence >= threshold)


@pytest.mark.parametrize("model_name", [None, "custom-model"])
def test_model_is_loaded_lazily_once_and_reused(monkeypatch, images, model_name):
    """Successive batches reuse one detector loaded with the configured identifier."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    constructor = Mock(return_value=model)
    # Patch the external SDK symbols where the library imports them.
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    monkeypatch.setattr(yolo_dataset_creation, "DEFAULT_MODEL", "test-default")

    mixin = configured_mixin(model_name=model_name)
    constructor.assert_not_called()
    assert mixin.speciesnet_model is None
    assert mixin.infer_batch([]) == {}
    constructor.assert_not_called()

    for path in images:
        result = mixin.infer_batch([path])
        assert result[str(path)]["detections"][0]["confidence"] == 0.9
        assert mixin.speciesnet_model is model

    constructor.assert_called_once_with(
        "test-default" if model_name is None else model_name,
        components="detector",
    )


def test_empty_batch_returns_empty_mapping_without_loading_model():
    """An empty batch needs neither an SDK installation nor an inference call."""
    assert configured_mixin().infer_batch([]) == {}


def test_inference_requires_explicit_initialization(images):
    """Missing lifecycle setup is reported explicitly instead of silently configured."""
    with pytest.raises(RuntimeError, match="initialize_speciesnet"):
        _SpeciesNetDatasetMixin().infer_batch(images)


def test_injected_model_is_stored_and_reused(images):
    """An injected detector remains available across successive batches."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    mixin = configured_mixin(model=model)

    for path in images:
        assert mixin.infer_batch([path])[str(path)]["detections"]
        assert mixin.speciesnet_model is model


def test_initialization_rejects_conflicting_model_settings():
    """A supplied model and a model identifier cannot silently override each other."""
    with pytest.raises(ValueError):
        configured_mixin(model=SpeciesNetStub([]), model_name="other-model")


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"iou_threshold": -0.1},
        {"iou_threshold": 1.1},
    ],
)
def test_invalid_inference_settings_raise_explicit_error(images, settings):
    """Invalid thresholds and batch sizes are rejected before inference."""
    with pytest.raises(ValueError):
        configured_mixin(model=SpeciesNetStub([])).infer_batch(images, **settings)


def test_missing_model_result_is_not_silently_dropped(images):
    """A missing prediction is an inference failure, not a negative detection."""
    with pytest.raises(ValueError, match="prediction"):
        infer(images, [])


def test_source_images_are_processed_in_bounded_batches(images):
    """A size-one model can process multiple inputs through the batch interface."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images],
        max_batch_size=1,
    )

    result = configured_mixin(model=model).infer_batch(images, batch_size=1)

    assert set(result) == {str(path) for path in images}
    assert all(len(metadata["detections"]) == 1 for metadata in result.values())


@pytest.mark.parametrize("fail", [False, True])
def test_external_detector_cutoff_is_restored_after_inference(images, fail):
    """The caller's detector setting survives successful and failed inference."""
    model = SpeciesNetStub(
        [{"filepath": str(images[0]), "detections": [detection(0.9)]}],
        fail=fail,
    )
    previous_threshold = SpeciesNetDetectorStub.DETECTION_THRESHOLD
    mixin = configured_mixin(model=model)

    if fail:
        with pytest.raises(RuntimeError, match="Model failed"):
            mixin.infer_batch(images[:1])
    else:
        mixin.infer_batch(images[:1])

    assert SpeciesNetDetectorStub.DETECTION_THRESHOLD == previous_threshold


@pytest.mark.speciesnet_integration
def test_real_speciesnet_api_returns_detection_metadata_and_crops(tmp_path):
    """Infer real boxes/crops on a bundled photograph, without an SDK substitute.

    Run with RUN_SPECIESNET_INTEGRATION=1. SPECIESNET_MODEL may point to a local
    weights directory to avoid downloads; otherwise the SDK uses its default
    model and may download weights into its normal cache.
    """
    if os.environ.get("RUN_SPECIESNET_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPECIESNET_INTEGRATION=1 to run real model inference")

    from ultralytics.utils import ASSETS

    # This installed sample has people and a bus, so use those detector labels
    # rather than asserting animal detection on an artificial test image.
    paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in paths:
        shutil.copy2(ASSETS / "bus.jpg", path)

    mixin = configured_mixin(model_name=os.environ.get("SPECIESNET_MODEL"))
    result = mixin.infer_batch(
        paths,
        batch_size=2,
        confidence_threshold=0.5,
        allowed_classes=("human", "vehicle"),
        iou_threshold=0.45,
    )

    assert set(result) == {str(path) for path in paths}
    for path in paths:
        records = result[str(path)]["detections"]
        assert records, "Real inference must produce at least one accepted box"
        with Image.open(path) as source:
            source = source.convert("RGB")
            for record in records:
                assert 0.5 <= record["confidence"] <= 1.0
                assert record["label"] in {"human", "vehicle"}
                assert len(record["bbox"]) == 4
                assert record["detection_index"] >= 0
                crop = record["crop"]
                assert crop.mode == "RGB"
                assert 0 < crop.width <= source.width
                assert 0 < crop.height <= source.height
                x, y, width, height = record["bbox"]
                expected = source.crop(
                    (
                        x * source.width,
                        y * source.height,
                        (x + width) * source.width,
                        (y + height) * source.height,
                    )
                )
                assert crop.size == expected.size
                assert crop.tobytes() == expected.tobytes()
