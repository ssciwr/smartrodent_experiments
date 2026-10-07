"""Oversample training rows in already split per-species metadata."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from smartrodent.dataset_splitting import TrainingDatasetSampler


def compute_oversampling(
    species_name: str, frame: pd.DataFrame, all_frames: dict[str, pd.DataFrame]
) -> float:

    full_size = sum(len(df) for df in all_frames.values())
    all_sizes = {k: len(df) / full_size for k, df in all_frames.items()}
    max_size = max(all_sizes.values())
    current_size = all_sizes[species_name]
    target_size = 0.1 * max_size

    if current_size >= target_size:
        print("sampling to 1.0")
        return 1.0
    else:
        print("sampling to: ", target_size / current_size)
        return target_size / current_size


def main(config_path: Path) -> None:
    """Oversample training records and save metadata, config, and script source.

    Copy this script, including its custom oversampling rule, into the output
    root before sampling. Reruns replace the config and source snapshots.

    Args:
        config_path: YAML file containing sampler paths and settings.
    """
    config = yaml.safe_load(config_path.read_text())
    input_root = Path(config["paths"]["training_dataset_sampler_input"])
    output_root = Path(config["paths"]["training_dataset_sampler_output"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / config_path.name)
    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, output_root / source_path.name)
    records_by_species = {
        records_path.parent.name: pd.read_csv(records_path)
        for records_path in input_root.glob("*/records.csv")
    }
    sampled_records = TrainingDatasetSampler.from_config(config_path).sample(
        records_by_species
    )

    for species, records in sampled_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(output_path, index=False)


def _parse_args() -> argparse.Namespace:
    """Parse the training-sampling command's configuration path."""
    parser = argparse.ArgumentParser(
        description="Oversample training rows in per-species metadata records."
    )
    parser.add_argument(
        "config", type=Path, help="Path to a dataset-sampling YAML config."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
