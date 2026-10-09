"""Config copies at script output roots, without downloads or model training."""

import builtins
import importlib
from io import StringIO
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml


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
        "taskname": "animal",
        "labels": ["kept", "rejected"],
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
    ("module_name", "dataset_name", "section"),
    [
        ("scripts.download_inaturalist_data", "InaturalistDataset", "inaturalist"),
        ("scripts.download_gbif_data", "GbifDataset", "gbif"),
    ],
)
def test_download_entry_points_copy_config_before_download(
    tmp_path, monkeypatch, module_name, dataset_name, section
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
    if section == "inaturalist":
        monkeypatch.setattr("sys.argv", [module_name, str(config_path)])
        runpy.run_path(module.__file__, run_name="__main__")
    else:
        module.main(config_path)
    assert downloads == [output_root.resolve()]


@pytest.mark.parametrize(
    ("module_name", "trainer_name", "config_name", "operation", "uses_cli"),
    [
        (
            "scripts.train_yolo_detector",
            "YoloDetectionTrainer",
            "train_yolo_detector_config.yaml",
            "train",
            True,
        ),
        (
            "scripts.train_yolo_classfier",
            "YoloClassificationTrainer",
            "train_yolo_classifier_config.yaml",
            "train",
            True,
        ),
        (
            "scripts.tune_yolo_classifier",
            "YoloClassificationTrainer",
            "train_yolo_classifier_config.yaml",
            "tune",
            False,
        ),
    ],
)
def test_training_scripts_copy_original_config_to_project_base(
    tmp_path,
    monkeypatch,
    module_name,
    trainer_name,
    config_name,
    operation,
    uses_cli,
):
    import smartrodent

    monkeypatch.chdir(tmp_path)
    config_dir = (
        tmp_path
        if uses_cli
        else tmp_path / "projects" / "smartrodent_experiments" / "configs"
    )
    config_dir.mkdir(parents=True)
    config_path = config_dir / config_name
    output_root = tmp_path / "training_outputs"
    config_path.write_text(
        "# original training configuration\n"
        + yaml.safe_dump(
            {
                "train_kwargs": {"project": str(output_root)},
                "tune_kwargs": {"project": str(output_root)},
            }
        ),
        encoding="utf-8",
    )
    # Isolate the scripts' existing fixed /tmp scratch file during testing.
    original_open = builtins.open

    def isolated_open(file, *args, **kwargs):
        if str(file) == "/tmp/train_yolo_classifier_config.yaml":
            return StringIO()
        else:
            return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", isolated_open)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    trainer = Mock()
    trainer.train_kwargs = {"project": str(output_root)}
    trainer.model = SimpleNamespace(ckpt={"train_args": {}})

    def run():
        assert (output_root / config_name).read_bytes() == config_path.read_bytes()

    getattr(trainer, operation).side_effect = run
    factory = Mock()
    factory.from_config.return_value = trainer
    monkeypatch.setattr(smartrodent, trainer_name, factory)
    if uses_cli:
        monkeypatch.setattr(sys, "argv", [module_name, str(config_path)])
    runpy.run_module(module_name, run_name="__main__")
    getattr(trainer, operation).assert_called_once()
    trainer.export.assert_called_once()
