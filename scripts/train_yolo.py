"""Train and export an Ultralytics detector or classifier from a positional YAML configuration."""

import argparse
import shutil
from pathlib import Path

from smartrodent import UltralyticsTrainer


def main(config_path: Path) -> None:
    """Train a model and export it using the supplied configuration.

    Args:
        config_path: Path to the Ultralytics training YAML configuration.
    """
    trainer = UltralyticsTrainer.from_config(config_path)

    trainer.train_project.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, trainer.train_project / config_path.name)
    trainer.train()

    print("model exported to:", trainer.export())


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(
        description="Train a configured Ultralytics detector or classifier."
    )
    parser.add_argument(
        "config", type=Path, help="Path to an Ultralytics training YAML configuration."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
