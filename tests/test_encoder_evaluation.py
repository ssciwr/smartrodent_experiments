"""Extract representations by calling encoders directly, in dataset order."""

import pytest
import torch
from classifier_task_test_support import (
    CLASS_NAMES,
    IMAGE_SIZE,
    train_supcon,
)
from torch import nn

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

import copy

import numpy as np
from PIL import Image
from torch.nn import functional as F

from smartrodent.eval import EvaluationImages, FeatureExtractor


class EvalEncoder(nn.Module):
    """A BatchNorm and Dropout encoder that exposes no pooling module to hook."""

    feature_dim = 4

    def __init__(self, fail: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)
        self.fail = fail

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image."""
        if self.fail:
            raise RuntimeError("encoder failed")
        return self.drop(self.bn(self.conv(images))).mean(dim=(2, 3))


@pytest.fixture
def evaluation_images(tmp_path):
    """Build six images across two species with different brightness."""
    paths = []
    for name in ("rat", "mouse"):
        directory = tmp_path / name
        directory.mkdir()
        for index in range(4):
            path = directory / f"{index}.png"
            Image.new("RGB", (10, 12), (index * 50, 40, 90)).save(path)
            paths.append(path)
    return EvaluationImages(
        paths, {"mouse": 0, "rat": 1}, samples_per_class=3, image_size=8
    )


def _modes(*modules: nn.Module) -> dict[nn.Module, bool]:
    """Record the training flag of every submodule."""
    return {m: m.training for root in modules for m in root.modules()}


def test_features_come_from_the_encoder_called_directly_in_dataset_order(
    evaluation_images,
):
    """Pooled rows are eval-mode encoder outputs; projected rows are unit length."""
    encoder = EvalEncoder().train()
    projector = nn.Sequential(nn.Dropout(0.9), nn.Linear(4, 3)).train()
    reference = copy.deepcopy(encoder).eval()
    reference_projector = copy.deepcopy(projector).eval()
    images = torch.stack([image for _, image in evaluation_images])

    first = FeatureExtractor(encoder, projector).extract(
        evaluation_images, batch_size=2
    )
    second = FeatureExtractor(encoder, projector).extract(
        evaluation_images, batch_size=4
    )

    with torch.inference_mode():
        pooled = reference(images)
        projected = F.normalize(reference_projector(pooled), dim=1)
    np.testing.assert_allclose(first.pooled, pooled.numpy(), atol=1e-6)
    np.testing.assert_allclose(first.projected, projected.numpy(), atol=1e-6)
    np.testing.assert_allclose(first.pooled, second.pooled, atol=1e-6)
    np.testing.assert_array_equal(first.labels, [0, 0, 0, 1, 1, 1])
    assert first.images == evaluation_images.images
    assert first.label_names == {"mouse": 0, "rat": 1}


def test_extraction_restores_mixed_training_modes_and_keeps_state(evaluation_images):
    """Frozen BatchNorm inside a training encoder is put back exactly as it was."""
    encoder = EvalEncoder().train()
    encoder.bn.eval()
    modes = _modes(encoder)
    state = {name: value.clone() for name, value in encoder.state_dict().items()}

    FeatureExtractor(encoder).extract(evaluation_images)

    assert _modes(encoder) == modes
    for name, value in encoder.state_dict().items():
        assert torch.equal(value, state[name]), name


def test_baseline_extraction_has_no_projected_features(evaluation_images):
    """Without a projector only the encoder features exist."""
    features = FeatureExtractor(EvalEncoder()).extract(evaluation_images)
    assert features.pooled.shape == (6, 4) and features.projected is None


def test_failed_extraction_restores_training_mode_and_allows_retry(evaluation_images):
    """A failing encoder leaves the modes intact and a later extraction works."""
    encoder = EvalEncoder(fail=True).train()
    extractor = FeatureExtractor(encoder)

    with pytest.raises(RuntimeError, match="encoder failed"):
        extractor.extract(evaluation_images)

    assert encoder.training and encoder.bn.training
    encoder.fail = False
    assert extractor.extract(evaluation_images).pooled.shape == (6, 4)


def test_feature_extractor_rejects_unusable_modules(tmp_path):
    """An encoder without parameters or a projector on another device fails early."""
    from smartrodent.eval import EvaluationImages, FeatureExtractor

    train, _ = (tmp_path / "data" / "train", None)
    train_supcon(tmp_path)
    images = EvaluationImages(
        sorted(train.rglob("*.png")),
        {name: index for index, name in enumerate(CLASS_NAMES)},
        samples_per_class=2,
        image_size=IMAGE_SIZE,
    )
    with pytest.raises(ValueError, match="parameters"):
        FeatureExtractor(nn.Identity()).extract(images)
    projector = nn.Linear(4, 3, device="meta")
    with pytest.raises(ValueError, match="same device"):
        FeatureExtractor(TinyEncoderLike(), projector).extract(images)


class TinyEncoderLike(nn.Module):
    """An encoder whose ``linear`` module the donor does not have."""

    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
