from pathlib import Path

from PIL import Image
import pytest

from smartrodent.inference import (
    YoloClassifier,
    _crop_normalized_bbox,
    _load_component_config,
    _load_yaml_mapping,
    _validate_max_top_k,
    load_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CONFIGS = [
    "inference_yolo26_speciesnames.yaml",
    "inference_yolo26_genusnames_224.yaml",
    "inference_yolo26_genusnames_640.yaml",
    "inference_yolo26_commonnames.yaml",
]


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


@pytest.mark.parametrize("config_name", INFERENCE_CONFIGS)
def test_inference_configs_reference_available_classifier_models(config_name):
    config_path = PROJECT_ROOT / "configs" / config_name

    config, _ = _load_yaml_mapping(config_path)
    detector_model, detector_kwargs = _load_component_config(
        config_path, component="detector", backend="speciesnet"
    )
    classifier_model, classifier_kwargs = _load_component_config(
        config_path, component="classifier", backend="yolo26"
    )

    assert detector_model == "kaggle:google/speciesnet/pyTorch/v4.0.3b/1"
    assert detector_kwargs["conf"] == 0.01
    assert Path(classifier_model).is_file()
    assert classifier_kwargs["max_top_k"] == 5
    assert config["inference"]["classes"]


def test_yolo_classifier_from_config_loads_real_model():
    config_path = PROJECT_ROOT / "configs" / "inference_yolo26_commonnames.yaml"

    classifier = YoloClassifier.from_config(config_path)

    assert classifier.max_top_k == 5
    assert "monitor lizard" in classifier.model.names.values()
    assert "max_top_k" not in classifier.kwargs
