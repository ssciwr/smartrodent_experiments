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

from smartrodent.eval import (
    EvaluationFeatures,
    EvaluationImages,
    TsneComparison,
)


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
