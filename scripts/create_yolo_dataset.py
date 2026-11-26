"""Create the YOLO dataset selected by a class-named YAML configuration."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from smartrodent.config_utils import ConfigHandler, get_loader, load_section
from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)


def main(config_path: Path) -> None:
    """Select a supported creator, load its configuration, and create the dataset.

    Args:
        config_path: YAML containing exactly one creator class name under ``data``.

    Raises:
        ValueError: The creator key is missing, unsupported or ambiguous, or settings
            are invalid.
        OSError: Configuration/source files cannot be read or outputs cannot be written.
    """
    configuration = yaml.load(
        config_path.expanduser().read_text(encoding="utf-8"), get_loader()
    )
    # Keep supported classes explicit: a YAML key must not trigger arbitrary imports.
    creators = {
        creator.__name__: creator
        for creator in (
            YoloDetectorDatasetCreatorFromSpeciesnet,
            YoloClassifierDatasetCreatorFromSpeciesnet,
        )
    }
    for config in ConfigHandler(configuration).run_configs:
        with TemporaryDirectory() as directory:
            run_config_path = Path(directory) / "config.yaml"
            run_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            names = list(load_section(run_config_path, "data"))
            if len(names) != 1 or names[0] not in creators:
                raise ValueError(
                    "Configure exactly one supported dataset creator under 'data' "
                    f"({sorted(creators)}), not {names}"
                )
            creators[names[0]].from_config(run_config_path).create()


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(description="Create a configured YOLO dataset.")
    parser.add_argument(
        "config", type=Path, help="Path to a dataset-creation YAML config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
