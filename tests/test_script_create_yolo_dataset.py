"""Verify dataset-list creation using real creators and an external SpeciesNet stub."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from speciesnet_test_support import SpeciesNetStub, detection

from smartrodent import yolo_dataset_creation
from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)


def _creation_document(creator_type, input_root, outputs):
    """Configure one creator shared by every listed input/output pair."""
    return {
        "creator": {
            "name": creator_type.__name__,
            "parameters": {"class_names": ["mouse"], "model_name": "test-speciesnet"},
        },
        "datasets": [
            {"path_to_image_data": str(input_root), "dataset_output_path": str(output)}
            for output in outputs
        ],
    }


@pytest.fixture(
    params=[
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ]
)
def script_config(request, images, tmp_path, monkeypatch):
    """Configure each real creator for two datasets, without weights or project data."""
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
    outputs = [tmp_path / "stage8_first", tmp_path / "stage8_second"]
    config_path = tmp_path / "creation.yaml"
    creator_type = request.param
    config_path.write_text(
        yaml.safe_dump(
            _creation_document(creator_type, images[0].parent.parent, outputs)
        )
    )
    return config_path, outputs, creator_type


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


def test_main_creates_every_listed_dataset_with_the_shared_creator(script_config):
    """Each list entry is one dataset; creator settings are shared by all of them."""
    from scripts.create_yolo_dataset import main

    config_path, outputs, creator_type = script_config

    main(config_path)

    for output in outputs:
        assert_created_dataset(output, creator_type)


def test_cli_runs_creation_from_positional_config(script_config, monkeypatch):
    """The executable entry point consumes one positional YAML path."""
    config_path, outputs, creator_type = script_config
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_yolo_dataset.py"
    monkeypatch.setattr("sys.argv", [str(script), str(config_path)])

    runpy.run_path(str(script), run_name="__main__")

    for output in outputs:
        assert_created_dataset(output, creator_type)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["creator"].update(name="UnknownCreator"),
        # a path belongs to one list entry, never to the shared creator settings
        lambda d: d["creator"]["parameters"].update(dataset_output_path="x"),
        lambda d: d.update(datasets=[]),
        lambda d: d["datasets"][1].pop("dataset_output_path"),
        lambda d: d.update(data={}),
    ],
    ids=["unknown", "shared_path", "empty", "incomplete", "unknown_key"],
)
def test_main_rejects_invalid_creation_configs_before_creating_anything(
    script_config, change
):
    """Configuration mistakes cannot silently choose a creator or drop a dataset."""
    from scripts.create_yolo_dataset import main

    config_path, outputs, _ = script_config
    document = yaml.safe_load(config_path.read_text())
    change(document)
    config_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError):
        main(config_path)
    assert not any(output.exists() for output in outputs)
