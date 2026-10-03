"""Migrate image path prefixes in per-species records files."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm


def main(config_path: Path) -> None:
    """Rewrite configured image path prefixes in species records.

    Args:
        config_path: YAML file defining ``image_root``, ``old_root``, and
            ``new_root``.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    image_root = Path(config["image_root"]).resolve()
    old_root = config["old_root"]
    new_root = config["new_root"]

    for species_path in tqdm(sorted(image_root.iterdir())):
        records_path = species_path / "records.csv"
        if not species_path.is_dir() or not records_path.is_file():
            continue

        print(species_path.name)
        records = pd.read_csv(records_path)
        backup_path = species_path / "records_original.csv"
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        records.to_csv(backup_path, index=False)

        mask = records["image_path"].str.startswith(old_root, na=False)
        records.loc[mask, "image_path"] = records.loc[
            mask, "image_path"
        ].str.replace(old_root, new_root, regex=False)

        missing = [
            path
            for path in records.loc[mask, "image_path"]
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} rewritten image paths do not exist"
            )

        records.to_csv(records_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse the migration configuration path."""
    parser = argparse.ArgumentParser(
        description="Migrate image paths stored in per-species records files."
    )
    parser.add_argument("config", type=Path, help="Path to the migration YAML file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
