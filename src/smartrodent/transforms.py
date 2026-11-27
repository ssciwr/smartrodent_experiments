"""Build shared preprocessing and configured torchvision augmentations."""

from collections.abc import Sequence
from dataclasses import field
from typing import Any

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.transforms import v2

from smartrodent.components import ComponentSpec
from smartrodent.training_validation import Count, MappingInput


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TransformSettings:
    """Configure square RGB preprocessing and training augmentations.

    Args:
        image_size: Size of the square passed to the model.
        resize: Native torchvision resize parameters.
        augmentations: Training transforms in the order they should run.
    """

    image_size: Count
    resize: dict[str, Any] = field(default_factory=dict)
    augmentations: Sequence[MappingInput[ComponentSpec]] = field(default_factory=list)


class ImageTransforms:
    """Prepare preprocessing shared across training and validation.

    Args:
        settings: Validated resize and augmentation settings.
    """

    def __init__(self, settings: TransformSettings):
        """Build the two transform pipelines once per stage."""
        preprocessing = [
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((settings.image_size, settings.image_size), **settings.resize),
        ]
        augmentations = [spec.build(v2) for spec in settings.augmentations]
        self._training = v2.Compose([*preprocessing, *augmentations])
        self._validation = v2.Compose(preprocessing)

    def training(self) -> v2.Compose:
        """Return the preprocessing followed by the configured augmentations."""
        return self._training

    def validation(self) -> v2.Compose:
        """Return the deterministic preprocessing used for evaluation."""
        return self._validation
