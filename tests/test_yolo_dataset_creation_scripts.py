"""Tests for configuration-driven YOLO dataset creation entry points."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str) -> ModuleType:
    """Load a script as a module without executing its CLI entry point."""
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(path: Path, contents: dict) -> Path:
    """Write a YAML configuration and return its path."""
    path.write_text(yaml.safe_dump(contents))
    return path


@pytest.mark.parametrize(
    ("script_name", "creator_name", "extra_config"),
    [
        (
            "create_yolo_detector_trainingdataset.py",
            "YoloDetectorDatasetCreatorFromSpeciesnet",
            {"path_to_labels": "labels"},
        ),
        (
            "create_yolo_classifier_trainingdataset.py",
            "YoloClassifierDatasetCreatorFromSpeciesnet",
            {"image_directory": "crops"},
        ),
    ],
)
def test_creation_scripts_resolve_paths_and_forward_configuration(
    tmp_path, monkeypatch, script_name, creator_name, extra_config
):
    module = load_script(script_name)
    config = {
        "path_to_image_data": "images",
        "dataset_output_path": "output",
        "class_names": ["mouse"],
        "train_val_test_split": [0.7, 0.2, 0.1],
        "rng_seed": 7,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.3,
        "labels_to_filter": ["animal"],
        **extra_config,
    }
    config_path = write_config(tmp_path / "config.yaml", config)
    calls = []

    class FakeCreator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self):
            return tmp_path / "output"

    monkeypatch.setattr(module, creator_name, FakeCreator)

    assert module.build_dataset(config_path) == tmp_path / "output"
    assert calls[0]["path_to_image_data"] == tmp_path / "images"
    assert calls[0]["dataset_output_path"] == tmp_path / "output"
    if "path_to_labels" in calls[0]:
        assert calls[0]["path_to_labels"] == tmp_path / "labels"
    assert calls[0]["train_val_test_split"] == (0.7, 0.2, 0.1)


def test_detector_main_accepts_configuration_argument(tmp_path, monkeypatch):
    module = load_script("create_yolo_detector_trainingdataset.py")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    calls = []
    monkeypatch.setattr(module, "build_dataset", lambda path: calls.append(path))

    module.main(["--config", str(config_path)])

    assert calls == [config_path]


def test_remapping_script_builds_classification_layout(tmp_path):
    module = load_script("create_yolo_classifier_trainingdataset_remapped.py")
    input_root = tmp_path / "input"
    for split in ("train", "val", "test"):
        class_dir = input_root / split / "Mus musculus"
        class_dir.mkdir(parents=True)
        (class_dir / f"{split}.jpg").write_bytes(split.encode())
    config_path = write_config(
        tmp_path / "remap.yaml",
        {
            "input_data_path": "input",
            "output_path": "output",
            "image_formats": [".jpg"],
            "ignore_unmapped_species": False,
            "mappings": {"common": {"Mus musculus": "mouse"}},
        },
    )

    result = module.remap_dataset(config_path, mapping="common")

    assert result == tmp_path / "output"
    for split in ("train", "val", "test"):
        link = result / split / "mouse" / f"{split}.jpg"
        assert link.is_symlink()
        assert link.resolve() == input_root / split / "Mus musculus" / f"{split}.jpg"
    metadata = yaml.safe_load((result / "data.yaml").read_text())
    assert metadata["names"] == {0: "mouse"}


def test_remapping_script_rejects_unknown_species(tmp_path):
    module = load_script("create_yolo_classifier_trainingdataset_remapped.py")
    input_root = tmp_path / "input" / "train" / "unknown"
    input_root.mkdir(parents=True)
    (input_root / "image.jpg").write_bytes(b"image")
    for split in ("val", "test"):
        (tmp_path / "input" / split).mkdir(parents=True)
    config_path = write_config(
        tmp_path / "remap.yaml",
        {
            "input_data_path": "input",
            "output_path": "output",
            "image_formats": [".jpg"],
            "ignore_unmapped_species": False,
            "mappings": {"common": {}},
        },
    )

    with pytest.raises(KeyError, match="unknown"):
        module.remap_dataset(config_path, mapping="common")


@pytest.mark.parametrize(
    "config_name",
    [
        "create_yolo_detector_dataset.yaml",
        "create_yolo_classifier_dataset.yaml",
        "coco_style_trainingdataset_creation_config.yaml",
    ],
)
def test_repository_configs_use_portable_paths(config_name):
    config_path = PROJECT_ROOT / "configs" / config_name
    config = yaml.safe_load(config_path.read_text())

    serialized = yaml.safe_dump(config)
    assert "/home/" not in serialized
    assert "datasets/" in serialized
