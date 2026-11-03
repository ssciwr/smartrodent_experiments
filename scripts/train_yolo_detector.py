"""Train a YOLO detector from a positional YAML configuration file."""

import argparse
import os
import shutil
from pathlib import Path

from smartrodent import YoloDetectionTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config_path: Path) -> None:
    """Train and export a detector using the supplied configuration.

    Args:
        config_path: Path to the detector-training YAML configuration.
    """
    trainer = YoloDetectionTrainer.from_config(config_path)

    print(trainer.model.ckpt["train_args"])

    output_dir = Path(trainer.train_kwargs["project"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)
    trainer.train()

    output = trainer.export()
    print("model exported to: ", output)


def _parse_args() -> argparse.Namespace:
    """Parse the required positional configuration-file path."""
    parser = argparse.ArgumentParser(description="Train a configured YOLO detector.")
    parser.add_argument(
        "config", type=Path, help="Path to a detector-training YAML configuration."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
