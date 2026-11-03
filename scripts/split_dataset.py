"""Split configured per-species metadata into train, validation, and test rows."""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from smartrodent.dataset_splitting import DatasetSplitter


def _discover_species_records(input_root: Path, records_glob: str) -> dict[str, Path]:
    """Find one metadata file per species, named by its immediate parent directory.

    Patterns are relative to the input root. Validate all matches before any
    output writes so broad globs cannot silently replace a species' metadata.
    """
    if not isinstance(records_glob, str) or not records_glob.strip():
        raise ValueError("records_glob must be a nonempty string")
    pattern_path = Path(records_glob)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("records_glob must be relative to the input root")
    if not input_root.exists():
        raise FileNotFoundError(
            f"Splitter input directory does not exist: {input_root}"
        )
    if not input_root.is_dir():
        raise NotADirectoryError(f"Splitter input is not a directory: {input_root}")

    records_paths = sorted(input_root.glob(records_glob))
    if not records_paths:
        raise FileNotFoundError(
            f"No records match {records_glob!r} in input directory {input_root}"
        )

    paths_by_species = {}
    for records_path in records_paths:
        if not records_path.is_file():
            raise ValueError(
                f"Records glob matched a path that is not a file: {records_path}"
            )
        # Directory names preserve species labels, including spaces, across layouts.
        species = records_path.parent.name
        if species in paths_by_species:
            raise ValueError(
                f"Multiple records files for species {species!r}: "
                f"{paths_by_species[species]} and {records_path}"
            )
        paths_by_species[species] = records_path
    return paths_by_species


def main(config_path: Path) -> None:
    """Split configured per-species records and write annotated CSV files.

    Each input CSV's parent directory names its species. Outputs remain
    ``<output>/<species>/records.csv``, independent of the input layout.
    Config directory paths use the working directory. Image references are saved
    relative to the shared dataset root, the parent of the configured input stage.

    Args:
        config_path: YAML file containing splitter paths, the required
            ``paths.records_glob``, and splitter settings.

    Raises:
        KeyError: A required configuration key is missing.
        FileNotFoundError: The configuration, input root, or matching records
            are missing.
        NotADirectoryError: The input root is not a directory.
        ValueError: The glob is invalid, matches a non-file, selects multiple
            CSVs for a species, or the splitter rejects the metadata.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_root = Path(config["paths"]["dataset_splitter_input"])
    dataset_root = input_root.resolve().parent
    output_root = Path(config["paths"]["dataset_splitter_output"])
    paths_by_species = _discover_species_records(
        input_root, config["paths"]["records_glob"]
    )
    records_by_species = {
        species: pd.read_csv(records_path)
        for species, records_path in paths_by_species.items()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / config_path.name)
    split_records = DatasetSplitter.from_config(config_path).split(records_by_species)

    for species, records in split_records.items():
        output_path = output_root / species / "records.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if "image_path" in records.columns:
            # na_action="ignore" preserves missing paths without warnings;
            # metadata splitting does not validate image availability.
            records["image_path"] = records["image_path"].map(
                lambda path: os.path.relpath(
                    (dataset_root / path).resolve(), dataset_root
                ),
                na_action="ignore",
            )
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
