"""Evaluate pretrained and trained representations on a fixed image subset."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, PILToTensor, Resize, ToDtype


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

    def __init__(
        self,
        images: Sequence[Path],
        label_names: Mapping[str, int],
        samples_per_class: int = 100,
        image_size: int = 448,
        seed: int = 42,
    ):
        if samples_per_class < 1 or image_size < 1:
            raise ValueError("samples_per_class and image_size must be positive")
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
    """Extract pooling features and optional projector outputs without training.

    Args:
        network: Underlying PyTorch image network, rather than the YOLO API wrapper.
        pool: Pooling module belonging to that network.
        projector: Optional projection module, on the same device as the network.

    Raises:
        ValueError: The supplied pooling module does not belong to the network.
    """

    def __init__(
        self, network: nn.Module, pool: nn.Module, projector: nn.Module | None = None
    ):
        if not any(module is pool for module in network.modules()):
            raise ValueError("pool must belong to network")
        self.network = network
        self.pool = pool
        self.projector = projector

    @torch.inference_mode()
    def extract(
        self, dataset: EvaluationImages, batch_size: int = 8, num_workers: int = 0
    ) -> EvaluationFeatures:
        """Extract CPU arrays in subset order and restore module training flags.

        A temporary hook works even after the training wrapper was finalized.
        Parameters, gradients, and device placement are left intact. BatchNorm
        statistics and dropout remain fixed during extraction.

        Args:
            dataset: Fixed images shared by the models being compared.
            batch_size: Number of images processed in one inference pass.
            num_workers: CPU image-loading worker count.

        Returns:
            Pooled and optional projected features, with labels and source paths.

        Raises:
            ValueError: Loader settings, dataset length, or devices are incompatible.
            RuntimeError: Pooling does not run exactly once in a forward pass.
        """
        if batch_size < 1 or num_workers < 0 or len(dataset) == 0:
            raise ValueError(
                "Use a nonempty dataset, positive batch_size, and num_workers >= 0"
            )
        parameter = next(self.network.parameters(), None)
        if parameter is None:
            raise ValueError(
                "network must have parameters to determine its device and dtype"
            )
        if self.projector is not None and any(
            p.device != parameter.device for p in self.projector.parameters()
        ):
            raise ValueError("projector and network must be on the same device")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=parameter.device.type == "cuda",
        )
        roots = [self.network]
        if self.projector is not None:
            roots.append(self.projector)
        states = {
            module: module.training for root in roots for module in root.modules()
        }
        captured = []

        def capture_pool(module, inputs, output):
            """Capture only this forward pass's spatially pooled tensor."""
            captured.append(output.flatten(1))

        handle = self.pool.register_forward_hook(capture_pool)
        pooled_rows, projected_rows, label_rows = [], [], []
        try:
            for root in roots:
                root.eval()
            for labels, images in loader:
                captured.clear()
                images = images.to(
                    device=parameter.device, dtype=parameter.dtype, non_blocking=True
                )
                self.network(images)
                if len(captured) != 1:
                    raise RuntimeError(
                        "Expected the selected pooling module to run exactly once"
                    )
                pooled = captured.pop()
                pooled_rows.append(pooled.float().cpu().numpy())
                if self.projector is not None:
                    projected = nn.functional.normalize(
                        self.projector(pooled).float(), dim=1
                    )
                    projected_rows.append(projected.cpu().numpy())
                label_rows.append(labels.numpy())
        finally:
            handle.remove()
            captured.clear()
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

    def __init__(
        self,
        perplexity: float = 30,
        pca_components: int | None = 50,
        seed: int = 42,
        max_iter: int = 1000,
    ):
        if perplexity <= 0 or (pca_components and pca_components < 2) or max_iter < 250:
            raise ValueError(
                "Use perplexity > 0, pca_components >= 2, and max_iter >= 250"
            )
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
