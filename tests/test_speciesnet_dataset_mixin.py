"""Behavioral contract for direct SpeciesNet batch inference.

Only the external SpeciesNet model is replaced. Images and crop operations use
real PIL images, and all input files live in pytest temporary directories.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

import pytest
from hypothesis import given, strategies as st
from PIL import Image

from smartrodent.yolo_dataset_creation import _SpeciesNetDatasetMixin


class SpeciesNetStub:
    """Provide native SpeciesNet predictions without model downloads or a GPU."""

    def __init__(self, predictions):
        """Store the externally supplied model response."""
        self.predictions = predictions
        self.detector = SimpleNamespace(DETECTION_THRESHOLD=0.1)

    def predict(self, *, filepaths, batch_size, **kwargs):
        """Return predictions for the requested input batch."""
        requested = {str(path) for path in filepaths}
        return {
            "predictions": [
                {
                    **deepcopy(item),
                    "detections": [
                        deepcopy(box)
                        for box in item["detections"]
                        if box["conf"] >= self.detector.DETECTION_THRESHOLD
                    ],
                }
                for item in self.predictions
                if item["filepath"] in requested
            ]
        }


@pytest.fixture
def images(tmp_path):
    """Create two differently colored images with identical base filenames."""
    paths = []
    for species, color in [("mouse", "red"), ("shrew", "blue")]:
        directory = tmp_path / species
        directory.mkdir()
        path = directory / "observation.png"
        image = Image.new("RGB", (20, 10), "black")
        image.paste(color, (5, 2, 15, 8))
        image.save(path)
        paths.append(path)
    return paths


def detection(confidence, *, label="animal", bbox=(0.25, 0.2, 0.5, 0.6)):
    """Build a native SpeciesNet detection with a normalized xywh box."""
    return {"conf": confidence, "label": label, "bbox": list(bbox)}


def infer(paths, predictions, *, threshold=0.5):
    """Exercise the proposed public interface with an external model substitute."""
    return _SpeciesNetDatasetMixin().infer_batch(
        paths,
        model=SpeciesNetStub(predictions),
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


def test_default_model_is_constructed_for_detection_only(monkeypatch, images):
    """Live inference can construct its detector without an injected model."""

    class DetectorOnlyModel(SpeciesNetStub):
        """Stand in for the external SDK constructor."""

        def __init__(self, model_name, *, components):
            """Accept only the expected default detector configuration."""
            if model_name != "test-default" or components != "detector":
                raise ValueError("Unexpected model configuration")
            super().__init__(
                [{"filepath": str(images[0]), "detections": [detection(0.9)]}]
            )

    monkeypatch.setitem(
        sys.modules,
        "speciesnet",
        SimpleNamespace(DEFAULT_MODEL="test-default", SpeciesNet=DetectorOnlyModel),
    )

    result = _SpeciesNetDatasetMixin().infer_batch(images[:1])

    assert result[str(images[0])]["detections"][0]["confidence"] == 0.9


def test_empty_batch_returns_empty_mapping_without_loading_model():
    """An empty batch needs neither an SDK installation nor an inference call."""
    assert _SpeciesNetDatasetMixin().infer_batch([]) == {}


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
        _SpeciesNetDatasetMixin().infer_batch(
            images, model=SpeciesNetStub([]), **settings
        )


def test_missing_model_result_is_not_silently_dropped(images):
    """A missing prediction is an inference failure, not a negative detection."""
    with pytest.raises(ValueError, match="prediction"):
        infer(images, [])
