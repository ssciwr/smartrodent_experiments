import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes, Results

from smartrodent.base import DetectorBase


class DummyDetector(DetectorBase):
    def __init__(self, batchsize=2):
        super().__init__("dummy", False, batchsize, "boxed", 0.5)

    def detect(self, _path, _out, *_args, **_kwargs):
        del _path, _out, _args, _kwargs
        return []

    def write_detections_json(self, _results, _json_path):
        del _results, _json_path


class RecordingModel:
    def __init__(self, result_names=None):
        self.predict_calls = []
        self.set_classes_calls = []
        self.get_text_pe_calls = []
        self.result_names = result_names or {0: "mouse", 1: "rat"}

    def get_text_pe(self, classes):
        self.get_text_pe_calls.append(list(classes))
        return "text-pe"

    def set_classes(self, classes, text_pe):
        self.set_classes_calls.append((list(classes), text_pe))

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        source = kwargs["source"]
        sources = source if isinstance(source, list) else [source]
        return [
            make_result(
                f"model_{idx}.jpg",
                boxes=[[0, 0, 5, 5, 0.8, idx % 2]],
                names=self.result_names,
            )
            for idx, _ in enumerate(sources)
        ]


@pytest.fixture
def detector():
    return DummyDetector()


@pytest.fixture
def sample_paths(tmp_path):
    paths = [tmp_path / f"image_{idx}.jpg" for idx in range(3)]
    for path in paths:
        path.write_text("not real image data")
    return paths


@pytest.fixture
def recording_model():
    return RecordingModel()


def make_result(path="image.jpg", boxes=None, names=None):
    result = Results(
        orig_img=np.zeros((10, 10, 3), dtype=np.uint8),
        path=str(path),
        names=names or {0: "mouse", 1: "rat"},
    )
    if boxes is not None:
        if hasattr(boxes, "clone"):
            data = boxes.clone().detach().to(dtype=torch.float32)
        else:
            data = torch.tensor(boxes, dtype=torch.float32)
        result.boxes = Boxes(data, (10, 10))
    return result


@pytest.fixture
def yolo_result():
    return make_result(
        "camera.jpg",
        boxes=[
            [0, 0, 5, 5, 0.4567, 0],
            [1, 1, 6, 6, 0.9876, 1],
        ],
    )


@pytest.fixture
def empty_yolo_result():
    return make_result("empty.jpg", boxes=torch.empty((0, 6)))
