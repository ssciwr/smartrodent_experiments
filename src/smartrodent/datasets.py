"""Image datasets and multiview batching for contrastive learning."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import default_collate
from torchvision.transforms.v2 import Compose


class ImageDataset(torch.utils.data.Dataset):
    """Load labeled images from one directory per species.

    Args:
        dir: Root directory containing the species directories.
        labels: Species names to assign alphabetical IDs, or an existing mapping.
        transform: Optional transformation applied to each image.
        img_extensions: Image filename suffixes to include.
        size: Retained notebook argument; resizing is supplied by the transform.
    """

    def __init__(
        self,
        dir: Path,
        labels: Iterable[str] | Mapping[str, int],
        transform: torch.nn.Module | Compose | None = None,
        img_extensions: Iterable[str] = (".jpg", ".png"),
        size: int = 640,
    ):
        self.data_dir = dir
        if isinstance(labels, Mapping):
            self.label_names = dict(labels)
        else:
            self.label_names = {label: i for i, label in enumerate(sorted(labels))}
        self.labels = []
        self.class_counts = {}
        self.img_extensions = img_extensions

        self.images = []
        for label_name, label_id in self.label_names.items():
            directory = self.data_dir / label_name
            if not directory.exists():
                raise FileExistsError(
                    f"no directory named {label_name} in {self.data_dir}"
                )
            images = [
                image
                for image in directory.iterdir()
                if image.suffix in self.img_extensions
            ]
            self.images.extend(images)
            self.labels.extend([label_id] * len(images))
            self.class_counts[label_name] = len(images)

        self.transform = transform

    def __len__(self):
        """Return the number of indexed images."""
        return len(self.images)

    def __getitem__(self, index):
        """Return the integer class label and transformed image at an index."""
        img_path = self.images[index]
        label = self.labels[index]

        with Image.open(img_path) as img:
            if self.transform:
                image = self.transform(img)
            else:
                image = torch.tensor(img)
        return label, image


class MultiViewImageDataset(ImageDataset):
    """Produce two independently augmented views using the image transform."""

    def __getitem__(self, index):
        """Return a label and two views, or the original single-view fallback."""
        img_path = self.images[index]
        label = self.labels[index]

        if self.transform:
            with Image.open(img_path) as img:
                view1 = self.transform(img)
                view2 = self.transform(img)
            return label, view1, view2
        else:
            with Image.open(img_path) as img:
                image = torch.tensor(img)
            return label, image


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
