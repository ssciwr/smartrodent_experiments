"""Create the YOLO dataset selected by a class-named YAML configuration."""

import argparse
from pathlib import Path

import yaml

from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)


def main(config_path: Path) -> None:
    """Select a supported creator, load its configuration, and create the dataset.

    Args:
        config_path: YAML containing exactly one creator class name under ``data``.

    Raises:
        TypeError: Configuration sections are not mappings or values are invalid.
        ValueError: The creator key is unsupported or ambiguous, or settings are invalid.
        OSError: Configuration/source files cannot be read or outputs cannot be written.
    """
    configuration = yaml.safe_load(config_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise TypeError("The configuration must be a mapping")
    data = configuration.get("data")
    if not isinstance(data, dict):
        raise TypeError("The 'data' configuration must be a mapping")
    if len(data) != 1:
        raise ValueError("Configure exactly one dataset creator under 'data'")

    # Keep supported classes explicit: a YAML key must not trigger arbitrary imports.
    creators = {
        "YoloDetectorDatasetCreatorFromSpeciesnet": YoloDetectorDatasetCreatorFromSpeciesnet,
        "YoloClassifierDatasetCreatorFromSpeciesnet": YoloClassifierDatasetCreatorFromSpeciesnet,
    }
    name = next(iter(data))
    if name not in creators:
        raise ValueError(f"Unsupported dataset creator: {name!r}")
    creators[name].from_config(config_path).create()


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(description="Create a configured YOLO dataset.")
    parser.add_argument(
        "config", type=Path, help="Path to a dataset-creation YAML config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
