"""Create the YOLO datasets listed in a dataset-creation YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.yolo_dataset_creation import DatasetCreationPlan


def main(config_path: Path) -> None:
    """Create every listed dataset with the configured creator.

    Args:
        config_path: YAML with a ``creator`` (name and shared parameters) and a
            ``datasets`` list of input/output directories.

    Raises:
        ValueError: The configuration or any dataset's records are invalid; nothing
            is created then.
        OSError: Source files cannot be read or outputs cannot be written.
    """
    DatasetCreationPlan.from_yaml(config_path).create()


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(description="Create configured YOLO datasets.")
    parser.add_argument(
        "config", type=Path, help="Path to a dataset-creation YAML config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
