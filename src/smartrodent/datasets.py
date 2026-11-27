"""Image datasets and multiview batching for contrastive learning."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import default_collate


class ImageDataset(torch.utils.data.Dataset):
    """Load labeled RGB images from one directory per class.

    Only a class directory's own files are indexed, not nested folders. A
    configured class may be empty; callers decide whether they accept that.

    Args:
        dir: Root directory containing the class directories.
        labels: Class names, numbered in alphabetical order, or an existing
            name-to-ID mapping (e.g. the training set's, shared by validation).
        transform: Optional transformation applied to each RGB PIL image; without
            one, items hold the PIL image itself.
        img_extensions: Image filename suffixes to include.

    Raises:
        FileNotFoundError: A configured class has no directory in ``dir``.
    """

    def __init__(
        self,
        dir: Path,
        labels: Iterable[str] | Mapping[str, int],
        transform: Callable | None = None,
        img_extensions: Iterable[str] = (".jpg", ".png"),
    ):
        self.data_dir = dir
        if isinstance(labels, Mapping):
            self.label_names = dict(labels)
        else:
            self.label_names = {label: i for i, label in enumerate(sorted(labels))}
        self.img_extensions = tuple(img_extensions)
        self.transform = transform

        self.images: list[Path] = []
        self.labels: list[int] = []
        self.class_counts: dict[str, int] = {}
        for label_name, label_id in self.label_names.items():
            directory = self.data_dir / label_name
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"no directory named {label_name} in {self.data_dir}"
                )
            # Sorted because iterdir() order depends on the filesystem; it would
            # otherwise change the sampled batches between machines.
            images = sorted(
                image
                for image in directory.iterdir()
                if image.is_file() and image.suffix in self.img_extensions
            )
            self.images.extend(images)
            self.labels.extend([label_id] * len(images))
            self.class_counts[label_name] = len(images)

    def __len__(self):
        """Return the number of indexed images."""
        return len(self.images)

    def _load(self, index: int) -> Image.Image:
        """Read one image as RGB; ``convert`` also reads the pixels before closing."""
        # Grayscale or RGBA files (e.g. some PNGs) would otherwise reach the
        # transforms with the wrong number of channels.
        with Image.open(self.images[index]) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        """Return the integer class label and the (transformed) image at an index."""
        image = self._load(index)
        if self.transform is None:
            return self.labels[index], image
        else:
            return self.labels[index], self.transform(image)


class MultiViewImageDataset(ImageDataset):
    """Produce two independently augmented views of each image with its transform.

    Args:
        dir: Root directory containing the class directories.
        labels: Class names or an existing name-to-ID mapping, as for
            ``ImageDataset``.
        transform: Random augmentation, applied separately for each view.
        img_extensions: Image filename suffixes to include.
    """

    def __init__(
        self,
        dir: Path,
        labels: Iterable[str] | Mapping[str, int],
        transform: Callable,
        img_extensions: Iterable[str] = (".jpg", ".png"),
    ):
        super().__init__(dir, labels, transform, img_extensions)
        # Repeated only to narrow the type for checkers: here it is never None.
        self.transform: Callable = transform

    def __getitem__(self, index):
        """Return a label and two views from separate calls of the transform."""
        image = self._load(index)
        return self.labels[index], self.transform(image), self.transform(image)


def collate_views(samples: tuple):
    """Combine two views per source image into one label-aligned batch.

    Args:
        samples: Samples containing an integer label and two image tensors.

    Returns:
        Repeated labels and concatenated images, with all first views preceding
        all second views.
    """
    labels, view1, view2 = default_collate(samples)
    return labels.repeat(2), torch.cat([view1, view2], dim=0)
