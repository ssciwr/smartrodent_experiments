"""Verify script dispatch using real creators and an external SpeciesNet stub."""

from pathlib import Path
import json
import runpy

import pandas as pd
import pytest
import yaml

from smartrodent import yolo_dataset_creation
from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)
from speciesnet_test_support import SpeciesNetStub, detection


@pytest.fixture(
    params=[
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ]
)
def script_config(request, images, tmp_path, monkeypatch):
    """Configure each real creator without downloading weights or using project data."""
    for path in images:
        pd.DataFrame([{"image_path": str(path), "dataset_split": "train"}]).to_csv(
            path.parent / "records.csv", index=False
        )
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    monkeypatch.setattr(
        yolo_dataset_creation, "SpeciesNet", lambda *args, **kwargs: model
    )
    output = tmp_path / "stage8_output"
    config_path = tmp_path / "creation.yaml"
    creator_type = request.param
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    creator_type.__name__: {
                        "path_to_image_data": str(images[0].parent.parent),
                        "dataset_output_path": str(output),
                        "class_names": ["mouse"],
                        "model_name": "test-speciesnet",
                    }
                }
            }
        )
    )
    return config_path, output, creator_type


def assert_created_dataset(output, creator_type):
    """Check selected classes and format-specific outputs through their files."""
    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert len(metadata) == 1
    assert metadata[0]["species"] == "mouse"
    assert metadata[0]["status"] == "written"
    image = output / metadata[0]["outputs"][0]["image_path"]
    assert image.is_file()
    if creator_type is YoloDetectorDatasetCreatorFromSpeciesnet:
        assert image.is_symlink()
        label = output / metadata[0]["outputs"][0]["label_path"]
        assert label.read_text().split() == ["0", "0.5", "0.5", "0.5", "0.6"]
    elif creator_type is YoloClassifierDatasetCreatorFromSpeciesnet:
        assert not image.is_symlink()
        assert image.parent == output / "train" / "mouse"
    else:
        pytest.fail("Unexpected creator type")


def test_main_selects_creator_from_class_named_config(script_config):
    """The config's key selects the real detector or classifier conversion workflow."""
    from scripts.create_yolo_dataset import main

    config_path, output, creator_type = script_config

    main(config_path)

    assert_created_dataset(output, creator_type)


def test_cli_runs_creation_from_positional_config(script_config, monkeypatch):
    """The executable entry point consumes one positional YAML path."""
    config_path, output, creator_type = script_config
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_yolo_dataset.py"
    monkeypatch.setattr("sys.argv", [str(script), str(config_path)])

    runpy.run_path(str(script), run_name="__main__")

    assert_created_dataset(output, creator_type)


@pytest.mark.parametrize(
    "configuration, error",
    [
        (None, TypeError),
        ([], TypeError),
        ({"data": None}, TypeError),
        ({"data": []}, TypeError),
        ({"data": {}}, ValueError),
        ({"data": {"UnknownCreator": {}}}, ValueError),
        (
            {
                "data": {
                    "YoloDetectorDatasetCreatorFromSpeciesnet": {},
                    "YoloClassifierDatasetCreatorFromSpeciesnet": {},
                }
            },
            ValueError,
        ),
        ({"data": {"YoloDetectorDatasetCreatorFromSpeciesnet": None}}, TypeError),
    ],
)
def test_main_rejects_invalid_or_ambiguous_creator_configs(
    tmp_path, configuration, error
):
    """Unsupported or ambiguous keys cannot silently choose a format."""
    from scripts.create_yolo_dataset import main

    config_path = tmp_path / "creation.yaml"
    config_path.write_text(yaml.safe_dump(configuration))

    with pytest.raises(error):
        main(config_path)
