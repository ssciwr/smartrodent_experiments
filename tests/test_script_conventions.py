"""Conventions shared by every pipeline script: positional config and config copies."""

import importlib
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml


@pytest.mark.parametrize(
    ("module_name", "parser_name"),
    [
        ("scripts.download_lila_data", "_parse_args"),
        ("scripts.filter_data_vlm", "parse_args"),
        ("scripts.materialize_filtered_data", "parse_args"),
        ("scripts.sample_training_dataset", "_parse_args"),
        ("scripts.split_dataset", "_parse_args"),
        ("scripts.create_yolo_dataset", "_parse_args"),
        ("scripts.train_yolo", "_parse_args"),
        ("scripts.tune_yolo", "_parse_args"),
    ],
)
# The config is positional and required: it parses, and omitting it fails. Unknown
# options such as -c are argparse's own error handling and need no test here.
@pytest.mark.parametrize("arguments", [["settings.yaml"], []])
def test_scripts_require_positional_config(
    module_name, parser_name, arguments, monkeypatch
):
    parser = getattr(importlib.import_module(module_name), parser_name)
    monkeypatch.setattr("sys.argv", [module_name, *arguments])
    if arguments == ["settings.yaml"]:
        assert parser().config == Path("settings.yaml")
    else:
        with pytest.raises(SystemExit) as error:
            parser()
        assert error.value.code == 2


@pytest.mark.parametrize(
    ("module_name", "worker_name", "method", "input_key", "output_key"),
    [
        (
            "scripts.filter_data_vlm",
            "VLMFilter",
            "filter_data",
            "input_root",
            "output_root",
        ),
        (
            "scripts.split_dataset",
            "DatasetSplitter",
            "split",
            "dataset_splitter_input",
            "dataset_splitter_output",
        ),
        (
            "scripts.sample_training_dataset",
            "TrainingDatasetSampler",
            "sample",
            "training_dataset_sampler_input",
            "training_dataset_sampler_output",
        ),
    ],
)
def test_processing_scripts_copy_supplied_yaml(
    tmp_path, monkeypatch, module_name, worker_name, method, input_key, output_key
):
    module = importlib.import_module(module_name)
    input_root = tmp_path / "input"
    species_dir = input_root / "mouse"
    species_dir.mkdir(parents=True)
    records = pd.DataFrame({"photo_id": [1], "image_path": ["image.jpg"]})
    records.to_csv(species_dir / "records.csv", index=False)
    output_root = tmp_path / "output"
    config_path = tmp_path / "settings.yaml"
    config = {
        "paths": {input_key: str(input_root), output_key: str(output_root)},
    }
    if module_name == "scripts.split_dataset":
        config["paths"]["records_glob"] = "*/records.csv"
    config_path.write_text(
        "# supplied configuration\n" + yaml.safe_dump(config), encoding="utf-8"
    )

    def process(records_by_species):
        assert (output_root / config_path.name).read_bytes() == config_path.read_bytes()
        return records_by_species

    worker = Mock()
    getattr(worker, method).side_effect = process
    factory = Mock()
    factory.from_config.return_value = worker
    monkeypatch.setattr(module, worker_name, factory)
    module.main(config_path)
    pd.testing.assert_frame_equal(
        pd.read_csv(output_root / "mouse" / "records.csv"), records
    )

    # Ordinary copies replace the config on reruns rather than adding new policy.
    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write("# updated run\n")
    module.main(config_path)
    assert (output_root / config_path.name).read_bytes() == config_path.read_bytes()


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "settings", "log_name"),
    [
        (
            "scripts.download_inaturalist_data",
            "InaturalistDataset",
            {"inaturalist": {"species": ["Mus musculus"], "years": [2024]}},
            "inaturalist_dataset.log",
        ),
        (
            "scripts.download_lila_data",
            "LilaDataset",
            {
                "lila": {
                    "datasets": {"Alpha Traps": "cdla"},
                    "classes": ["Rattus"],
                    "allowed_licenses": ["cdla"],
                    "index_url": "https://lila.test/index.csv",
                    "taxonomy_url": "https://lila.test/taxonomy.csv",
                }
            },
            "lila_dataset.log",
        ),
    ],
)
def test_download_entry_points_log_to_a_file_in_the_output(
    tmp_path, monkeypatch, package_logger, module_name, dataset_name, settings, log_name
):
    """The downloaders' messages reach a log file next to the downloaded data."""
    module = importlib.import_module(module_name)
    output_root = tmp_path / "downloaded"
    (section,) = settings
    config_path = tmp_path / "download.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"data": {section: {"output_path": str(output_root), **settings[section]}}}
        )
    )

    def download(dataset):
        dataset.logger.info("probe message")

    monkeypatch.setattr(getattr(module, dataset_name), "download", download)

    module.main(config_path)

    assert "probe message" in (output_root / log_name).read_text()


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "section"),
    [
        ("scripts.download_inaturalist_data", "InaturalistDataset", "inaturalist"),
    ],
)
def test_download_entry_points_copy_config_before_download(
    tmp_path, monkeypatch, package_logger, module_name, dataset_name, section
):
    module = importlib.import_module(module_name)
    output_root = tmp_path / "downloaded"
    config_path = tmp_path / "download.yaml"
    config_path.write_text(
        "# original download settings\n"
        + yaml.safe_dump(
            {
                "data": {
                    section: {
                        "output_path": str(output_root),
                        "species": ["Mus musculus"],
                        "years": [2024],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    downloads = []

    def download(dataset):
        assert (output_root / config_path.name).read_bytes() == config_path.read_bytes()
        downloads.append(dataset.output_path)

    monkeypatch.setattr(getattr(module, dataset_name), "download", download)
    monkeypatch.setattr("sys.argv", [module_name, str(config_path)])
    runpy.run_path(module.__file__, run_name="__main__")
    assert downloads == [output_root.resolve()]


@pytest.mark.parametrize(
    ("module_name", "operation", "project_attribute"),
    [
        ("scripts.train_yolo", "train", "train_project"),
        ("scripts.tune_yolo", "tune", "tune_project"),
    ],
)
def test_yolo_scripts_copy_original_config_to_project_before_running(
    tmp_path, monkeypatch, module_name, operation, project_attribute
):
    """Config copied byte-for-byte into the project; only training exports; no CUDA env."""
    import smartrodent

    # Device choice belongs to the config: an unset variable must stay unset.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config_path = tmp_path / "yolo_config.yaml"
    config_path.write_text("# original configuration\ntask: detect\n", encoding="utf-8")
    project = tmp_path / "outputs" / "nested"
    trainer = Mock()
    setattr(trainer, project_attribute, project)

    def run():
        assert (project / config_path.name).read_bytes() == config_path.read_bytes()
        return project

    getattr(trainer, operation).side_effect = run
    factory = Mock()
    factory.from_config.return_value = trainer
    monkeypatch.setattr(smartrodent, "UltralyticsTrainer", factory)
    monkeypatch.setattr(sys, "argv", [module_name, str(config_path)])
    # Other tests import the script as a regular module; runpy warns when re-running
    # an already-imported module as __main__, so drop it for this test only.
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    runpy.run_module(module_name, run_name="__main__")

    factory.from_config.assert_called_once_with(config_path)
    getattr(trainer, operation).assert_called_once()
    assert trainer.export.call_count == (1 if operation == "train" else 0)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
