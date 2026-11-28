"""Tune an Ultralytics detector or classifier from a positional YAML configuration."""

import argparse
import shutil
from pathlib import Path

from smartrodent import UltralyticsTrainer


def main(config_path: Path) -> None:
    """Run Ultralytics' hyperparameter tuner using the supplied configuration.

    Nothing is exported: the tuner does not load its best weights into the model, so
    the best weights and ``best_hyperparameters.yaml`` stay in the tuner's output
    directory, ``<tune project>/<name>`` (``tune`` when no name is configured),
    numbered on reruns.

    Args:
        config_path: Path to the Ultralytics training YAML configuration.
    """
    trainer = UltralyticsTrainer.from_config(config_path)

    trainer.tune_project.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, trainer.tune_project / config_path.name)

    print("tuning project:", trainer.tune(), "(results in <name or tune>[N]/ inside)")


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(
        description="Tune a configured Ultralytics detector or classifier."
    )
    parser.add_argument(
        "config", type=Path, help="Path to an Ultralytics training YAML configuration."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
