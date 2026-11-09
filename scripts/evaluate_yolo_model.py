"""Evaluate a trained YOLO classifier on a test split from a positional YAML config."""

import argparse
import shutil
from pathlib import Path

from smartrodent.testing import YoloTester


def main(config_path: Path) -> None:
    """Copy the config next to the results, then compute and save test metrics.

    Args:
        config_path: Path to the model-testing YAML configuration.
    """
    tester = YoloTester.from_config(config_path)
    shutil.copy2(config_path, tester.output_path / config_path.name)
    tester.test()
    print("metrics written to: ", tester.output_path / tester.metrics_file_name)


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO classifier on a test split."
    )
    parser.add_argument(
        "config", type=Path, help="Path to a model-testing YAML configuration."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
