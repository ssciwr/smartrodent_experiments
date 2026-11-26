"""Behavioral checks for balanced image representation evaluation."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from threadpoolctl import threadpool_limits
from torch import nn

from smartrodent.eval import (
    EvaluationFeatures,
    EvaluationImages,
    FeatureExtractor,
    TsneComparison,
)


class SmallNetwork(nn.Module):
    """Provide a parameterized encoder, pooling layer, and classifier on CPU."""

    def __init__(self, skip_pool=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(4, 2)
        self.skip_pool = skip_pool

    def forward(self, images):
        """Return classifier outputs, optionally bypassing the selected pool."""
        features = self.bn(self.conv(images))
        if self.skip_pool:
            pooled = features.mean(dim=(2, 3))
        else:
            pooled = self.pool(features).flatten(1)
        return self.linear(self.drop(pooled))


@pytest.fixture
def evaluation_images(tmp_path):
    """Build RGB and grayscale images in temporary species directories."""
    paths = []
    for name in ("rat", "mouse"):
        directory = tmp_path / name
        directory.mkdir()
        for index in range(4):
            path = directory / f"{index}.png"
            Image.new("L", (10, 12), index * 50).save(path)
            paths.append(path)
    return EvaluationImages(
        paths, {"mouse": 0, "rat": 1}, samples_per_class=3, image_size=8
    )


@given(
    counts=st.lists(st.integers(1, 12), min_size=2, max_size=5),
    seed=st.integers(0, 1000),
)
@settings(max_examples=30, deadline=None)
def test_balanced_selection_is_reproducible_without_duplicates(counts, seed):
    """Every species contributes equal distinct images, regardless of input order."""
    names = {f"class_{i}": i for i in range(len(counts))}
    paths = [
        Path(f"/unused/class_{i}/{j}.png")
        for i, count in enumerate(counts)
        for j in range(count)
    ]
    per_class = min(counts)
    first = EvaluationImages(paths, names, samples_per_class=per_class, seed=seed)
    second = EvaluationImages(
        list(reversed(paths)), names, samples_per_class=per_class, seed=seed
    )
    assert first.images == second.images
    assert len(first) == len(set(first.images)) == per_class * len(names)
    assert Counter(p.parent.name for p in first.images) == {
        name: per_class for name in names
    }


def test_evaluation_preprocessing_is_deterministic_rgb(evaluation_images):
    """Repeated reads yield scaled, fixed-size RGB tensors with stable labels."""
    label, image = evaluation_images[0]
    other_label, other_image = evaluation_images[0]
    assert label == other_label == 0
    assert image.shape == (3, 8, 8) and image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert torch.equal(image, other_image)


@pytest.mark.parametrize(
    "paths,names,kwargs,message",
    [
        ([], {"mouse": 0}, {"samples_per_class": 0}, "positive"),
        ([], {"mouse": 0}, {"image_size": 0}, "positive"),
        ([], {}, {}, "distinct class IDs"),
        ([], {"mouse": 0, "rat": 0}, {}, "distinct class IDs"),
        ([Path("/unused/unknown/0.png")], {"mouse": 0}, {}, "Unknown species"),
        ([], {"mouse": 0}, {}, "distinct images"),
    ],
)
def test_invalid_image_selection_fails_explicitly(paths, names, kwargs, message):
    """Invalid selections fail before loading images or running a network."""
    with pytest.raises(ValueError, match=message):
        EvaluationImages(paths, names, **kwargs)


def test_extraction_restores_modes_and_preserves_weights(evaluation_images):
    """Inference preserves mixed module modes, weights, BatchNorm, and gradients."""
    network = SmallNetwork().train()
    network.bn.eval()
    projector = nn.Sequential(nn.Dropout(0.9), nn.Linear(4, 3)).train()
    modes = {m: m.training for root in (network, projector) for m in root.modules()}
    weights = {name: value.clone() for name, value in network.state_dict().items()}
    network.conv.weight.grad = torch.ones_like(network.conv.weight)
    extractor = FeatureExtractor(network, network.pool, projector)
    first = extractor.extract(evaluation_images, batch_size=2)
    second = extractor.extract(evaluation_images, batch_size=3)
    assert first.pooled.shape == (6, 4) and first.projected.shape == (6, 3)
    assert first.pooled.dtype == first.projected.dtype == np.float32
    np.testing.assert_allclose(first.pooled, second.pooled, atol=1e-6)
    np.testing.assert_allclose(first.projected, second.projected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(first.projected, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(first.labels, [0, 0, 0, 1, 1, 1])
    assert first.images == evaluation_images.images
    assert all(module.training == mode for module, mode in modes.items())
    assert all(
        torch.equal(value, weights[name])
        for name, value in network.state_dict().items()
    )
    assert torch.equal(network.conv.weight.grad, torch.ones_like(network.conv.weight))


def test_baseline_extraction_has_no_projector(evaluation_images):
    """A baseline network produces pooling features without projected features."""
    network = SmallNetwork().eval()
    features = FeatureExtractor(network, network.pool).extract(evaluation_images)
    assert features.pooled.shape == (6, 4) and features.projected is None
    assert not network.training


def test_failed_extraction_restores_modes_and_allows_retry(evaluation_images):
    """A failed forward restores training state and a later extraction succeeds."""
    network = SmallNetwork(skip_pool=True).train()
    projector = nn.Linear(4, 3).train()
    extractor = FeatureExtractor(network, network.pool, projector)
    with pytest.raises(RuntimeError, match="exactly once"):
        extractor.extract(evaluation_images)
    assert network.training and projector.training
    network.skip_pool = False
    assert extractor.extract(evaluation_images).pooled.shape == (6, 4)


def test_extractor_rejects_unrelated_pool():
    """A pool from another model cannot silently yield missing features."""
    with pytest.raises(ValueError, match="belong"):
        FeatureExtractor(SmallNetwork(), nn.AdaptiveAvgPool2d(1))


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"num_workers": -1}])
def test_extractor_rejects_invalid_loader_settings(evaluation_images, kwargs):
    """Loader configuration errors are reported before mode changes."""
    network = SmallNetwork()
    with pytest.raises(ValueError, match="nonempty dataset"):
        FeatureExtractor(network, network.pool).extract(evaluation_images, **kwargs)


def test_extractor_rejects_parameterless_network(evaluation_images):
    """Device and dtype inference requires a parameterized model."""
    network = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    with pytest.raises(ValueError, match="parameters"):
        FeatureExtractor(network, network[0]).extract(evaluation_images)


def test_extractor_rejects_projector_on_another_device(evaluation_images):
    """A device mismatch is detected without moving the user's model."""
    network = SmallNetwork()
    projector = nn.Linear(4, 3, device="meta")
    with pytest.raises(ValueError, match="same device"):
        FeatureExtractor(network, network.pool, projector).extract(evaluation_images)


@pytest.mark.parametrize("dimensions", [4, 12])
def test_tsne_projection_is_finite_and_reproducible(dimensions):
    """Both direct and PCA-reduced inputs yield reproducible 2D coordinates."""
    features = (
        np.random.default_rng(42).normal(size=(12, dimensions)).astype(np.float32)
    )
    plotter = TsneComparison(perplexity=3, pca_components=5, max_iter=300)
    with threadpool_limits(limits=2):
        first = plotter.project(features)
        second = plotter.project(features)
    assert first.shape == (12, 2) and np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    "features,message",
    [
        (np.ones(4), "at least three"),
        (np.ones((4, 1)), "at least three"),  # one feature dimension
        (np.full((4, 4), np.nan), "finite"),
        (np.ones((4, 4)), "identical"),
        (np.arange(16).reshape(4, 4), "perplexity"),
    ],
)
def test_tsne_rejects_invalid_features(features, message):
    """Invalid and collapsed representations produce actionable errors."""
    with pytest.raises(ValueError, match=message):
        TsneComparison().project(features)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"perplexity": 0},
        {"pca_components": 1},
        {"max_iter": 100},
    ],
)
def test_tsne_rejects_invalid_configuration(kwargs):
    """Invalid projection settings fail at configuration time."""
    with pytest.raises(ValueError):
        TsneComparison(**kwargs)


@pytest.fixture
def comparison_features():
    """Provide baseline and trained spaces for the same labeled image paths."""
    generator = np.random.default_rng(42)
    labels = np.repeat([0, 1], 6)
    paths = tuple(Path(f"/{label}/{index}.png") for index, label in enumerate(labels))
    names = {"mouse": 0, "rat": 1}
    baseline = EvaluationFeatures(
        generator.normal(size=(12, 6)), None, labels, paths, names
    )
    trained = replace(
        baseline,
        pooled=generator.normal(size=(12, 6)),
        projected=generator.normal(size=(12, 3)),
    )
    return baseline, trained


def test_plot_uses_matching_species_colors_and_three_spaces(comparison_features):
    """The comparison has three panels with consistent species colors and legend."""
    baseline, trained = comparison_features
    with threadpool_limits(limits=2):
        figure, coordinates = TsneComparison(perplexity=3, max_iter=300).plot(
            baseline, trained
        )
    try:
        assert len(figure.axes) == len(coordinates) == 3
        assert all(points.shape == (12, 2) for points in coordinates.values())
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "mouse",
            "rat",
        ]
        for axis in figure.axes[1:]:
            for first, other in zip(figure.axes[0].collections, axis.collections):
                np.testing.assert_array_equal(
                    first.get_facecolors(), other.get_facecolors()
                )
    finally:
        plt.close(figure)


@pytest.mark.parametrize("change", ["paths", "labels", "names", "projected", "rows"])
def test_plot_rejects_misaligned_comparisons(comparison_features, change):
    """Comparisons cannot silently mix different sources or missing features."""
    baseline, trained = comparison_features
    if change == "paths":
        trained = replace(trained, images=trained.images[::-1])
    elif change == "labels":
        trained = replace(trained, labels=1 - trained.labels)
    elif change == "names":
        trained = replace(trained, label_names={"different": 0, "rat": 1})
    elif change == "projected":
        trained = replace(trained, projected=None)
    elif change == "rows":
        trained = replace(trained, pooled=trained.pooled[:-1])
    else:
        raise AssertionError(change)
    with pytest.raises(ValueError):
        TsneComparison().plot(baseline, trained)
