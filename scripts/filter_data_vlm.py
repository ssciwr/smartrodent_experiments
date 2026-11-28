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
        config_path: YAML file with ``paths`` (input/output roots), an optional
            ``species`` selection, and the ``filter`` section read by ``VLMFilter``.
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
    # Species outside the selection bypass the filter but are still written out.
    records_by_species, other_records = {}, {}
    for records_path in sorted(input_root.glob("*/records.csv")):
        species = records_path.parent.name
        if selected_species is None or species in selected_species:
            records_by_species[species] = pd.read_csv(records_path)
        else:
            other_records[species] = pd.read_csv(records_path)

    # Resolve only at the inference boundary; the CSV anchor stays the same for
    # every stage, including species that bypass this filtering task.
    for records in records_by_species.values():
        # na_action="ignore" keeps missing paths intact so VLMFilter can report
        # invalid records, rather than path conversion raising a type error first.
        records["image_path"] = records["image_path"].map(
            lambda path: str((dataset_root / path).resolve()), na_action="ignore"
        )
    vlm_filter = VLMFilter.from_config(config_path)
    filtered_records = vlm_filter.filter_data(records_by_species)
    for species, records in filtered_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # na_action="ignore" preserves missing values; conversion itself neither
        # validates them nor emits warnings.
        records["image_path"] = records["image_path"].map(
            lambda path: os.path.relpath(path, dataset_root), na_action="ignore"
        )
        records.to_csv(output_path, index=False)

    # Bypassed species are not filtered: no label applies and no answer exists.
    for species, records in other_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for column in vlm_filter.label_columns.values():
            records[column] = False
        for column in vlm_filter.answer_columns.values():
            records[column] = pd.NA
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
