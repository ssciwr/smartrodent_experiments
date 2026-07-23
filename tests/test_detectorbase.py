from pathlib import Path

import pytest

from conftest import DummyDetector, make_result
from smartrodent.base import DetectorBase


class DetectOnlyDetector(DetectorBase):
    def detect(self, path, out, *args, **kwargs):
        del path, out, args, kwargs
        return []


def test_write_detections_json_is_abstract():
    with pytest.raises(TypeError, match="write_detections_json"):
        DetectOnlyDetector("test", False, 1, "boxed", 0.5)


def test_resolve_local_model_absolute_and_missing_relative(detector, tmp_path):
    absolute = tmp_path / "weights.pt"

    assert detector.resolve_local_model(absolute) == str(absolute)
    assert detector.resolve_local_model("missing.pt") == "missing.pt"


def test_resolve_local_model_finds_bundled_file(detector, tmp_path, monkeypatch):
    fake_base = tmp_path / "base.py"
    bundled = tmp_path / "weights.pt"
    fake_base.write_text("")
    bundled.write_text("weights")

    monkeypatch.setattr("smartrodent.base.__file__", str(fake_base))

    assert detector.resolve_local_model("weights.pt") == str(bundled)


def test_restore_result_paths_updates_lists_only(detector, sample_paths):
    results = [make_result("image0.jpg"), make_result("image1.jpg")]

    restored = detector.restore_result_paths(results, sample_paths[:2])

    assert restored is results
    assert [result.path for result in results] == [
        str(path.resolve()) for path in sample_paths[:2]
    ]

    unchanged = [make_result("single.jpg")]
    assert detector.restore_result_paths(unchanged, sample_paths[0]) is unchanged
    assert unchanged[0].path == "single.jpg"


def test_path_batches_single_path_list_chunks_and_invalid_batchsize(sample_paths):
    detector = DummyDetector(batchsize=2)

    assert list(detector.path_batches(sample_paths[0])) == [sample_paths[0]]
    assert list(detector.path_batches("image.jpg")) == ["image.jpg"]
    assert list(detector.path_batches(sample_paths)) == [
        sample_paths[:2],
        sample_paths[2:],
    ]

    bad_detector = DummyDetector(batchsize=0)
    with pytest.raises(ValueError, match="batchsize must be positive"):
        list(bad_detector.path_batches(sample_paths))
