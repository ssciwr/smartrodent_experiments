"""Build shared preprocessing and configured torchvision augmentations."""

from collections.abc import Mapping, Sequence
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


def _build_augmentation(spec: ComponentSpec) -> Any:
    """Build one augmentation, building the specs listed in its ``transforms`` first.

    A ``transforms`` parameter is the argument name torchvision's wrappers share
    (``RandomApply``, ``RandomChoice``, ``RandomOrder``, ``Compose``). Treating it as
    a list of augmentation specs lets YAML configure them. The alternative, a
    dedicated ``RandomApply`` entry in the spec format, would cover only one wrapper.

    Args:
        spec: Augmentation name and parameters.

    Returns:
        The built transform.

    Raises:
        ValueError: ``transforms`` is not a list of specs, or a (nested) transform is
            unknown or rejects its parameters.
    """
    if "transforms" not in spec.parameters:
        return spec.build(v2)
    nested = spec.parameters["transforms"]
    if not isinstance(nested, list) or not all(
        isinstance(item, Mapping) for item in nested
    ):
        raise ValueError(
            f"Parameter 'transforms' of '{spec.name}' must be a list of "
            "augmentation specs with a name and optional parameters"
        )
    try:
        nested_specs = [ComponentSpec(**item) for item in nested]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid spec in parameter 'transforms' of '{spec.name}': {error}"
        ) from error
    built = [_build_augmentation(nested_spec) for nested_spec in nested_specs]
    return ComponentSpec(
        name=spec.name, parameters={**spec.parameters, "transforms": built}
    ).build(v2)


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
        augmentations = [_build_augmentation(spec) for spec in settings.augmentations]
        self._training = v2.Compose([*preprocessing, *augmentations])
        self._validation = v2.Compose(preprocessing)

    def training(self) -> v2.Compose:
        """Return the preprocessing followed by the configured augmentations."""
        return self._training

    def validation(self) -> v2.Compose:
        """Return the deterministic preprocessing used for evaluation."""
        return self._validation
