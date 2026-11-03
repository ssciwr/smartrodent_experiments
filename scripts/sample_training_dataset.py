"""Oversample training rows in already split per-species metadata."""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from smartrodent.dataset_splitting import TrainingDatasetSampler


def compute_oversampling(
    species_name: str, frame: pd.DataFrame, all_frames: dict[str, pd.DataFrame]
) -> float:
    """Scale nonempty species toward ten percent of the largest population.

    Counts use full split frames, while the sampler applies the multiplier only
    to training rows. Empty species receive one: sampling cannot invent source
    rows, and this also makes an entirely empty population safe.

    Args:
        species_name: Species identifier supplied by the sampler's rule interface.
        frame: Full split metadata for this species.
        all_frames: Complete species population, including this species' frame.

    Returns:
        A multiplier of at least one; one leaves existing rows unchanged.
    """
    current_size = len(frame)
    if current_size == 0:
        print("sampling to 1.0")
        return 1.0

    # Population normalization cancels in the target/current ratio. Using counts
    # directly avoids division by the total population, including an empty one.
    target_size = 0.1 * max(len(records) for records in all_frames.values())
    if current_size >= target_size:
        print("sampling to 1.0")
        return 1.0
    else:
        multiplier = target_size / current_size
        print("sampling to: ", multiplier)
        return multiplier


def main(config_path: Path) -> None:
    """Oversample training records and save metadata, config, and script source.

    Copy this script, including its custom oversampling rule, into the output
    root before sampling. Reruns replace the config and source snapshots. Saved
    image references use the shared dataset root, the parent of the input stage.

    Args:
        config_path: YAML file containing sampler paths and settings.
    """
    config = yaml.safe_load(config_path.read_text())
    input_root = Path(config["paths"]["training_dataset_sampler_input"])
    dataset_root = input_root.resolve().parent
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
        if "image_path" in records.columns:
            # na_action="ignore" preserves missing paths without warnings;
            # metadata sampling does not validate image availability.
            records["image_path"] = records["image_path"].map(
                lambda path: os.path.relpath(
                    (dataset_root / path).resolve(), dataset_root
                ),
                na_action="ignore",
            )
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
