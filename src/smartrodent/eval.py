"""Evaluate pretrained and trained representations on a fixed image subset."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from pydantic import Field, validate_call
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, PILToTensor, Resize, ToDtype

from .training_validation import Count, NonnegativeInt, PositiveNumber, Seed


class EvaluationImages(Dataset):
    """Select a reproducible balanced subset and load it without augmentation.

    Args:
        images: Source image paths from the training dataset, without resampling.
        label_names: Mapping from species directory names to integer class IDs.
        samples_per_class: Number of distinct images to select from each class.
        image_size: Height and width used by deterministic preprocessing.
        seed: Seed used only for subset selection.

    Raises:
        ValueError: Selection sizes, labels, or available class counts are invalid.
    """

    @validate_call
    def __init__(
        self,
        images: Sequence[Path],
        label_names: Mapping[str, int],
        samples_per_class: Count = 100,
        image_size: Count = 448,
        seed: Seed = 42,
    ):
        if not label_names or len(set(label_names.values())) != len(label_names):
            raise ValueError("label_names must contain distinct class IDs")

        self.label_names = dict(label_names)
        grouped = {name: [] for name in self.label_names}
        for image in images:
            image = Path(image)
            if image.parent.name not in grouped:
                raise ValueError(f"Unknown species directory: {image.parent.name}")
            grouped[image.parent.name].append(image)

        generator = np.random.default_rng(seed)
        selected = []
        for name in sorted(grouped):
            # Sorting makes the selection independent of filesystem enumeration.
            candidates = sorted(set(grouped[name]))
            if len(candidates) < samples_per_class:
                raise ValueError(
                    f"{name!r} has {len(candidates)} distinct images; "
                    f"requested {samples_per_class}"
                )
            indices = generator.choice(
                len(candidates), size=samples_per_class, replace=False
            )
            selected.extend(candidates[int(index)] for index in indices)

        self.images = tuple(selected)
        self.transform = Compose(
            [
                PILToTensor(),
                ToDtype(torch.float32, scale=True),
                Resize((image_size, image_size), antialias=True),
            ]
        )

    def __len__(self) -> int:
        """Return the number of selected images."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        """Return a species ID and deterministically preprocessed RGB image."""
        path = self.images[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return self.label_names[path.parent.name], tensor


@dataclass(frozen=True)
class EvaluationFeatures:
    """Hold extracted features on CPU with their source identities.

    Attributes:
        pooled: Feature matrix before the classifier and projector.
        projected: Normalized projector outputs, or ``None`` for a baseline.
        labels: One integer species ID per feature row.
        images: Source paths in feature-row order.
        label_names: Mapping from species names to their integer IDs.
    """

    pooled: np.ndarray
    projected: np.ndarray | None
    labels: np.ndarray
    images: tuple[Path, ...]
    label_names: dict[str, int]


class FeatureExtractor:
    """Extract encoder features and optional projector outputs without training.

    Args:
        encoder: Encoder returning ``(B, feature_dim)`` features for RGB in [0, 1].
        projector: Optional projection module, on the same device as the encoder.

    Raises:
        ValueError: The encoder has no parameters, or the devices differ.
    """

    def __init__(self, encoder: nn.Module, projector: nn.Module | None = None):
        self.encoder = encoder
        self.projector = projector

    @torch.inference_mode()
    @validate_call(config={"arbitrary_types_allowed": True})
    def extract(
        self,
        dataset: EvaluationImages,
        batch_size: Count = 8,
        num_workers: NonnegativeInt = 0,
    ) -> EvaluationFeatures:
        """Extract CPU arrays in dataset order and restore module training flags.

        Parameters, gradients, and device placement are left intact. BatchNorm
        statistics and dropout remain fixed during extraction.

        Args:
            dataset: Fixed images shared by the models being compared.
            batch_size: Number of images processed in one inference pass.
            num_workers: CPU image-loading worker count.

        Returns:
            Encoder and optional projected features, with labels and source paths.

        Raises:
            ValueError: Loader settings or devices are incompatible.
        """
        parameter = next(self.encoder.parameters(), None)
        if parameter is None:
            raise ValueError(
                "encoder must have parameters to determine its device and dtype"
            )
        if self.projector is not None and any(
            p.device != parameter.device for p in self.projector.parameters()
        ):
            raise ValueError("projector and encoder must be on the same device")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=parameter.device.type == "cuda",
        )
        roots = [self.encoder]
        if self.projector is not None:
            roots.append(self.projector)
        states = {
            module: module.training for root in roots for module in root.modules()
        }
        pooled_rows, projected_rows, label_rows = [], [], []
        try:
            for root in roots:
                root.eval()
            for labels, images in loader:
                images = images.to(
                    device=parameter.device, dtype=parameter.dtype, non_blocking=True
                )
                pooled = self.encoder(images)
                pooled_rows.append(pooled.float().cpu().numpy())
                if self.projector is not None:
                    projected = nn.functional.normalize(
                        self.projector(pooled).float(), dim=1
                    )
                    projected_rows.append(projected.cpu().numpy())
                label_rows.append(labels.numpy())
        finally:
            # Restore each flag directly to preserve mixed states such as frozen
            # BatchNorm modules inside an otherwise training-mode encoder.
            for module, training in states.items():
                module.training = training

        projected = (
            np.concatenate(projected_rows) if self.projector is not None else None
        )
        return EvaluationFeatures(
            pooled=np.concatenate(pooled_rows),
            projected=projected,
            labels=np.concatenate(label_rows),
            images=dataset.images,
            label_names=dict(dataset.label_names),
        )


class TsneComparison:
    """Plot pretrained pooled, trained pooled, and trained projected features.

    Each panel is a separate t-SNE fit. Colors identify the same species across
    panels; coordinates and distances between panels are not directly comparable.

    Args:
        perplexity: t-SNE neighborhood setting, strictly below the sample count.
        pca_components: Maximum feature dimension before t-SNE.
        seed: Seed for PCA and t-SNE.
        max_iter: Number of t-SNE optimization iterations, at least 250.
    """

    @validate_call
    def __init__(
        self,
        perplexity: PositiveNumber = 30,
        pca_components: Annotated[int, Field(strict=True, ge=2)] | None = 50,
        seed: Seed = 42,
        max_iter: Annotated[int, Field(strict=True, ge=250)] = 1000,
    ):
        self.perplexity = perplexity
        self.pca_components = pca_components
        self.seed = seed
        self.max_iter = max_iter

    def project(self, features: np.ndarray) -> np.ndarray:
        """Reduce L2-normalized features through optional PCA and then t-SNE.

        Args:
            features: Finite matrix with at least three rows and two columns.

        Returns:
            Two-dimensional coordinates, one row per image.

        Raises:
            ValueError: Features are invalid or perplexity exceeds the sample count.
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] < 3 or features.shape[1] < 2:
            raise ValueError(
                "features must have at least three samples and two dimensions"
            )
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        if not np.any(np.ptp(features, axis=0) > 0):
            raise ValueError(
                "All feature rows are identical; there is no variation to visualize"
            )
        if self.perplexity >= len(features):
            raise ValueError(
                "perplexity must be smaller than the number of selected images"
            )
        # Compare angular structure consistently in both pooled and projected
        # spaces. Euclidean distances of unit vectors reflect cosine similarity.
        reduced = normalize(features)
        if self.pca_components and reduced.shape[1] > self.pca_components:
            components = min(self.pca_components, len(reduced) - 1)
            reduced = PCA(
                n_components=components, svd_solver="randomized", random_state=self.seed
            ).fit_transform(reduced)
        return TSNE(
            n_components=2,
            perplexity=self.perplexity,
            init="pca",
            learning_rate="auto",
            random_state=self.seed,
            max_iter=self.max_iter,
        ).fit_transform(reduced)

    def plot(
        self,
        baseline: EvaluationFeatures,
        trained: EvaluationFeatures,
        figsize: tuple[int, int] = (18, 6),
        markersize: int = 12,
    ) -> tuple[plt.Figure, dict[str, np.ndarray]]:
        """Plot the three feature spaces for identical images and species labels.

        Args:
            baseline: Pooled features from the original pretrained checkpoint.
            trained: Pooled and projected features from the trained network.

        Returns:
            Matplotlib figure and a dictionary of per-panel t-SNE coordinates.

        Raises:
            ValueError: Images or class labels differ, or projected features are missing.
        """
        if (
            baseline.images != trained.images
            or baseline.label_names != trained.label_names
            or not np.array_equal(baseline.labels, trained.labels)
        ):
            raise ValueError(
                "Compare the same images, in the same order, with identical labels"
            )
        if trained.projected is None:
            raise ValueError("trained features must include projector outputs")
        panels = {
            "Pretrained encoder · pooled": baseline.pooled,
            "Trained encoder · pooled": trained.pooled,
            "Trained projector": trained.projected,
        }
        if any(len(features) != len(trained.labels) for features in panels.values()):
            raise ValueError("Each feature matrix must have one row per label")
        coordinates = {
            title: self.project(features) for title, features in panels.items()
        }
        figure, axes = plt.subplots(1, 3, figsize=figsize)
        # Grab the discrete colors from both tables (20 + 20 = 40 total)
        tab20_colors = plt.get_cmap("tab20").colors
        tab20b_colors = plt.get_cmap("tab20b").colors
        tab20c_colors = plt.get_cmap("tab20c").colors

        # Combine them and slice out exactly how many classes you have (25)
        colors = ListedColormap(
            np.vstack((tab20_colors, tab20b_colors, tab20c_colors))[
                : len(trained.label_names)
            ]
        )

        names = sorted(trained.label_names, key=trained.label_names.get)
        for axis, (title, points) in zip(axes, coordinates.items()):
            for index, name in enumerate(names):
                mask = trained.labels == trained.label_names[name]
                axis.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    color=colors(index),
                    label=name,
                    s=markersize,
                    alpha=0.7,
                )
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, legend_labels, loc="lower center", ncol=min(3, len(names))
        )
        figure.suptitle("Same images · independent t-SNE projections")
        figure.tight_layout(rect=(0, 0.12, 1, 0.94))
        return figure, coordinates
