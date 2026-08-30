from PIL import Image
import pytest

from smartrodent.inference import (
    _crop_normalized_bbox,
    _validate_max_top_k,
    load_image,
)


def test_load_image_returns_rgb_image_detached_from_file(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (12, 8), color=128).save(image_path)

    image = load_image(str(image_path))
    image_path.unlink()

    assert image.mode == "RGB"
    assert image.size == (12, 8)
    assert image.getpixel((0, 0)) == (128, 128, 128)


@pytest.mark.parametrize("max_top_k", [0, -1, 1.5, True, None])
def test_max_top_k_must_be_a_positive_integer(max_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        _validate_max_top_k(max_top_k)


def test_crop_normalized_bbox_uses_image_coordinates():
    image = Image.new("RGB", (100, 50))

    crop = _crop_normalized_bbox(image, [0.1, 0.2, 0.5, 0.4])

    assert crop is not None
    assert crop.size == (50, 20)


def test_crop_normalized_bbox_clips_to_image_bounds():
    image = Image.new("RGB", (100, 50))

    crop = _crop_normalized_bbox(image, [-0.1, -0.2, 0.3, 0.5])

    assert crop is not None
    assert crop.size == (20, 15)


def test_crop_normalized_bbox_rejects_empty_crop():
    image = Image.new("RGB", (100, 50))

    assert _crop_normalized_bbox(image, [0.5, 0.5, 0.0, 0.0]) is None
