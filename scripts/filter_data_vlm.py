"""Annotate per-species metadata records using a configured VLM filter."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from smartrodent import VLMFilter


def main(config_path: Path) -> None:
    """Filter input records and write annotated copies by species.

    Args:
        config_path: YAML file containing VLM and input/output path settings.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_root = Path(config["paths"]["input_root"]).resolve()
    output_root = Path(config["paths"]["output_root"]).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    selected_species = config.get("species")
    records_by_species = {
        records_path.parent.name: pd.read_csv(records_path)
        for records_path in sorted(input_root.glob("*/records.csv"))
        if selected_species is None or records_path.parent.name in selected_species
    }

    filtered_records = VLMFilter.from_config(config_path).filter_data(
        records_by_species
    )
    for species, records in filtered_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse the VLM filtering configuration path."""
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "filter_data_vlm_config_animal.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Annotate image metadata using a VLM backend."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to a YAML config file (default: {default_config})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
