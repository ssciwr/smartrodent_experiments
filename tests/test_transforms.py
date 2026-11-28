"""Training augmentations built from specs, including specs nested in specs."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from study_test_support import load_tagged_yaml
from torchvision.transforms import v2

from smartrodent.search_space import SearchSpace
from smartrodent.transforms import ImageTransforms, TransformSettings

IMAGE_SIZE = 8


def _transforms(*augmentations: dict) -> ImageTransforms:
    """Build the image transforms with the given augmentation specs."""
    return ImageTransforms(
        TransformSettings(image_size=IMAGE_SIZE, augmentations=list(augmentations))
    )


def _image(seed: int, width: int = 12, height: int = 10) -> Image.Image:
    """Return a reproducible random RGB image."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _flip(p: float) -> dict:
    """Return a horizontal flip spec with the given probability."""
    return {"name": "RandomHorizontalFlip", "parameters": {"p": p}}


def _random_apply(p: float, *nested: dict) -> dict:
    """Return a RandomApply spec wrapping the nested specs."""
    return {
        "name": "RandomApply",
        "parameters": {"p": p, "transforms": list(nested)},
    }


def test_flat_augmentations_still_build_and_validation_stays_deterministic():
    """Specs without nesting behave as before; evaluation never augments."""
    transforms = _transforms(_flip(1.0))
    plain = _transforms()
    image = _image(0)
    assert torch.equal(
        transforms.training()(image), torch.flip(plain.training()(image), [-1])
    )
    assert torch.equal(transforms.validation()(image), plain.validation()(image))


def test_nested_spec_builds_the_wrapped_transform_with_its_parameters():
    """A `transforms` list of specs becomes the wrapper's list of built transforms."""
    nested = {"name": "ColorJitter", "parameters": {"brightness": 0.2}}
    transforms = _transforms(_random_apply(0.2, nested))
    wrapper = transforms.training().transforms[-1]
    assert isinstance(wrapper, v2.RandomApply)
    assert wrapper.p == 0.2
    (inner,) = wrapper.transforms
    assert isinstance(inner, v2.ColorJitter)
    assert inner.brightness == (0.8, 1.2)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    width=st.integers(1, 20),
    height=st.integers(1, 20),
)
def test_applying_the_wrapper_with_probability_zero_leaves_any_image_unchanged(
    seed, width, height
):
    """p=0 never runs the nested transform, however extreme it is."""
    nested = {"name": "ColorJitter", "parameters": {"brightness": 1.0, "hue": 0.5}}
    image = _image(seed, width, height)
    wrapped = _transforms(_random_apply(0.0, nested, _flip(1.0)))
    assert torch.equal(wrapped.training()(image), _transforms().training()(image))


def test_wrappers_nest_more_than_one_level():
    """A nested spec may itself contain nested specs."""
    image = _image(2)
    twice = _transforms(_random_apply(1.0, _random_apply(1.0, _flip(1.0))))
    plain = _transforms()
    assert torch.equal(
        twice.training()(image), torch.flip(plain.training()(image), [-1])
    )


def test_unknown_nested_transform_is_reported_by_name():
    """A bad nested spec fails when the transforms are built, naming the culprit."""
    spec = _random_apply(0.5, {"name": "NoSuchTransform"})
    with pytest.raises(ValueError, match="NoSuchTransform"):
        _transforms(spec)


@pytest.mark.parametrize(
    "nested",
    [
        "ColorJitter",  # not a list
        [{"parameters": {"p": 0.5}}],  # an item without a name
        ["ColorJitter"],  # an item that is not a spec
    ],
)
def test_malformed_transforms_parameter_is_rejected(nested):
    """`transforms` must be a list of specs with names."""
    spec = {"name": "RandomApply", "parameters": {"p": 0.5, "transforms": nested}}
    with pytest.raises(ValueError, match="transforms"):
        _transforms(spec)


_NESTED_SPACE = """
augmentations:
  - name: RandomApply
    parameters:
      p: 0.2
      transforms:
        - name: ColorJitter
          parameters:
            brightness: !suggest-float {low: 0.0, high: 0.5}
"""
_BRIGHTNESS = "augmentations[0].parameters.transforms[0].parameters.brightness"


def test_search_tag_inside_a_nested_spec_is_named_by_its_full_path():
    """The search space finds tags below the `transforms` list."""
    space = SearchSpace.from_document(load_tagged_yaml(_NESTED_SPACE))
    assert set(space.distributions) == {_BRIGHTNESS}


@settings(max_examples=25, deadline=None)
@given(brightness=st.floats(0.0, 0.5))
def test_any_sampled_value_of_a_nested_tag_builds_a_working_pipeline(brightness):
    """Every value in the searched range resolves into a buildable pipeline."""
    space = SearchSpace.from_document(load_tagged_yaml(_NESTED_SPACE))
    resolved = space.resolve({_BRIGHTNESS: brightness})
    transforms = ImageTransforms(
        TransformSettings(
            image_size=IMAGE_SIZE, augmentations=resolved["augmentations"]
        )
    )
    assert transforms.training()(_image(3)).shape == (3, IMAGE_SIZE, IMAGE_SIZE)
