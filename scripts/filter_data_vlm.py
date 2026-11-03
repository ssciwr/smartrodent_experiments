"""Annotate per-species metadata records using a configured VLM filter."""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from smartrodent import VLMFilter


def main(config_path: Path) -> None:
    """Filter records using shared-root image paths and write portable CSVs.

    Image paths are relative to the parent of input_root. Absolute paths are
    used only while the backend reads originals, never in the output records.

    Args:
        config_path: YAML file containing VLM and input/output path settings.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_root = Path(config["paths"]["input_root"]).resolve()
    output_root = Path(config["paths"]["output_root"]).resolve()
    dataset_root = input_root.parent
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / config_path.name)
    selected_species = config.get("species")
    records_by_species = {
        records_path.parent.name: pd.read_csv(records_path)
        for records_path in sorted(input_root.glob("*/records.csv"))
        if selected_species is None or records_path.parent.name in selected_species
    }

    # Get the records for species that are not selected, so we can write them out with the task labels set to False
    other_records = {
        records_path.parent.name: pd.read_csv(records_path)
        for records_path in sorted(input_root.glob("*/records.csv"))
        if selected_species is not None
        and records_path.parent.name not in selected_species
    }

    # Resolve only at the inference boundary; the CSV anchor stays the same for
    # every stage, including species that bypass this filtering task.
    for records in records_by_species.values():
        # na_action="ignore" keeps missing paths intact so VLMFilter can report
        # invalid records, rather than path conversion raising a type error first.
        records["image_path"] = records["image_path"].map(
            lambda path: str((dataset_root / path).resolve()), na_action="ignore"
        )
    filtered_records = VLMFilter.from_config(config_path).filter_data(
        records_by_species
    )
    for species, records in filtered_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # na_action="ignore" preserves missing values; conversion itself neither
        # validates them nor emits warnings.
        records["image_path"] = records["image_path"].map(
            lambda path: os.path.relpath(path, dataset_root), na_action="ignore"
        )
        records.to_csv(output_path, index=False)

    # Get the task name and labels from the config, and create a list of task-specific label names
    task = config["taskname"]
    labels = config["labels"]
    task_labels = [f"{label}_{task}" for label in labels]

    # Write out the other species records with the task labels set to False
    for species, records in other_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for label in task_labels:
            records[label] = False
        # Bypassed species are not validated by VLMFilter. na_action="ignore"
        # deliberately preserves missing paths without warnings or dropping rows.
        records["image_path"] = records["image_path"].map(
            lambda path: os.path.relpath((dataset_root / path).resolve(), dataset_root),
            na_action="ignore",
        )
        records.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse the VLM filtering configuration path."""
    parser = argparse.ArgumentParser(
        description="Annotate image metadata using a VLM backend."
    )
    parser.add_argument("config", type=Path, help="Path to a VLM filtering YAML file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
