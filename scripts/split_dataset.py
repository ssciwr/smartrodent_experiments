"""Split per-species downloader metadata into train, validation, and test rows."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from smartrodent.dataset_splitting import DatasetSplitter


def main(config_path: Path) -> None:
    """Split configured per-species records and write annotated CSV files.

    Args:
        config_path: YAML file containing splitter paths and settings.
    """
    config = yaml.safe_load(config_path.read_text())
    input_root = Path(config["paths"]["dataset_splitter_input"])
    output_root = Path(config["paths"]["dataset_splitter_output"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / config_path.name)
    records_by_species = {
        records_path.parent.name: pd.read_csv(records_path)
        for records_path in input_root.glob("*/records.csv")
    }
    split_records = DatasetSplitter.from_config(config_path).split(records_by_species)

    for species, records in split_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(output_path, index=False)


def _parse_args() -> argparse.Namespace:
    """Parse the splitting command's configuration path."""
    parser = argparse.ArgumentParser(description="Split per-species metadata records.")
    parser.add_argument(
        "config", type=Path, help="Path to a dataset-splitting YAML config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
