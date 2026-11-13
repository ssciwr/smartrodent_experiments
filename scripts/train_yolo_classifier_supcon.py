"""Train a supervised-contrastive YOLO classifier from YAML configuration."""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

from smartrodent import YoloClassificationTrainer
from smartrodent import config_utils
from smartrodent.yolo_supcon import (
    SupConTrainerConfiguration,
    build_supcon_trainer,
)


def main(config_path: Path) -> None:
    """Train and export every SupCon classifier run in a configuration.

    Args:
        config_path: Path to the SupCon classifier-training YAML configuration.

    Raises:
        ValueError: The configuration does not enable supervised-contrastive
            training.
    """
    with config_path.open("r", encoding="utf-8") as config_file:
        configuration = yaml.load(config_file, config_utils.get_loader())

    run_configurations = config_utils.ConfigHandler(configuration).run_configs
    for run_configuration in run_configurations:
        supcon_configuration = SupConTrainerConfiguration.from_mapping(
            run_configuration.get("supcon")
        )
        trainer_type = build_supcon_trainer(supcon_configuration)
        if trainer_type is None:
            raise ValueError(
                "The SupCon training entry point requires supcon.enabled=true"
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
        ) as run_config_file:
            yaml.dump(run_configuration, run_config_file)
            run_config_file.flush()
            trainer = YoloClassificationTrainer.from_config(
                Path(run_config_file.name),
                trainer=trainer_type,
            )

        output_directory = Path(run_configuration["train_kwargs"]["project"])
        output_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_directory / config_path.name)

        trainer.train()
        exported_model = trainer.export()
        print("model exported to: ", exported_model)


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(
        description="Train a configured supervised-contrastive YOLO classifier."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the SupCon classifier-training YAML configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
