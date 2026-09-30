"""Create a YOLO detection dataset from SpeciesNet predictions."""

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from smartrodent.dataprocessing import YoloDetectorDatasetCreatorFromSpeciesnet


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "create_yolo_detector_dataset.yaml"
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize detector dataset configuration.

    Args:
        config_path: YAML file containing creator arguments. Relative paths inside
            the file are resolved from the configuration directory.

    Returns:
        Arguments ready for ``YoloDetectorDatasetCreatorFromSpeciesnet``.
    """
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    for key in (
        "path_to_image_data",
        "path_to_labels",
        "dataset_output_path",
        "background_image_dir",
    ):
        value = config.get(key)
        if value is not None:
            path = Path(value)
            config[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
    if "train_val_test_split" in config:
        config["train_val_test_split"] = tuple(config["train_val_test_split"])
    return config


def build_dataset(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Build a detection dataset using one YAML configuration."""
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(**load_config(config_path))
    return creator()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the detector dataset creation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    output = build_dataset(arguments.config)
    if output is not None:
        print(f"YOLO detection dataset written to {output}")


if __name__ == "__main__":
    main()
