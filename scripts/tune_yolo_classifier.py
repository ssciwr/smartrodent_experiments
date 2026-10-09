"""Tune a YOLO classifier from a positional YAML configuration file."""

import argparse
import os
import shutil
from pathlib import Path

import yaml

from smartrodent import YoloClassificationTrainer
from smartrodent import config_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config_path: Path) -> None:
    """Tune and export every classifier run expanded from a configuration.

    Args:
        config_path: Path to the classifier-tuning YAML configuration.
    """
    with config_path.open("r") as config_file:
        configuration = yaml.load(config_file, config_utils.get_loader())

    configs = config_utils.ConfigHandler(configuration).run_configs
    for config in configs:
        temporary_config_path = Path("/tmp/train_yolo_classifier_config.yaml")
        with open(temporary_config_path, "w") as temporary_config:
            yaml.dump(config, temporary_config)

        trainer = YoloClassificationTrainer.from_config(temporary_config_path)

        output_dir = Path(config["tune_kwargs"]["project"])
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_dir / config_path.name)
        trainer.tune()

        output = trainer.export()
        print("model exported to: ", output)


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(description="Tune a configured YOLO classifier.")
    parser.add_argument(
        "config", type=Path, help="Path to a classifier-tuning YAML configuration."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
