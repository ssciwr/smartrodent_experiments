"""Cap the number of image records per species by random downsampling."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


def main(config_path: Path) -> None:
    """Keep at most max_images_per_class random rows of each species' records.csv.

    Writes <output_directory>/<species>/records.csv and a copy of the config.
    Species with fewer rows keep all of them. Images are not copied.
    """
    config = yaml.safe_load(config_path.read_text())
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_directory"])
    max_images = config["max_images_per_class"]

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)

    for records_path in input_dir.glob("*/records.csv"):
        records = pd.read_csv(records_path)
        sampled = records.sample(
            n=min(max_images, len(records)), random_state=config.get("rng_seed", 42)
        ).sort_index()
        species_dir = output_dir / records_path.parent.name
        species_dir.mkdir(exist_ok=True)
        sampled.to_csv(species_dir / "records.csv", index=False)
        print(f"{records_path.parent.name}: kept {len(sampled)} of {len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the downsampling YAML config.")
    main(parser.parse_args().config)
