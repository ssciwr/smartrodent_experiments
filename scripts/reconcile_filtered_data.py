"""Record human corrections to materialized kept/rejected images."""

import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def copy_config(config_path: Path, corrected_root: Path) -> None:
    """Copy the reconciliation configuration into the materialized output root.

    Skip copying when the source and destination resolve to the same path.

    Args:
        config_path: Path to the reconciliation YAML file.
        corrected_root: Existing root directory containing corrected partitions.

    Raises:
        FileNotFoundError: The source file or destination directory is missing.
        PermissionError: Reading the source or writing the destination is denied.
    """
    if config_path.resolve() != (corrected_root / config_path.name).resolve():
        shutil.copy2(config_path, corrected_root / config_path.name)


def _validate_associations(
    corrected: pd.DataFrame,
    original_records: pd.DataFrame,
    materialized_images: list[str],
) -> None:
    """Reject materialized images or corrected associations absent from the source."""
    original_images = set(original_records["image_path"])
    if set(materialized_images).difference(original_images):
        raise ValueError(
            "Association between a materialized image and its original record is missing"
        )

    association_columns = ["id", "photo.id", "image_path"]
    corrected_associations = set(
        corrected.loc[:, association_columns].itertuples(name=None, index=False)
    )
    original_associations = set(
        original_records.loc[:, association_columns].itertuples(name=None, index=False)
    )
    if corrected_associations.difference(original_associations):
        raise ValueError("Association between id, photo.id, and image_path has changed")


def main(config_path: Path) -> None:
    """Record human image decisions in parameterized corrected-record files.

    For each configured input subdirectory and output column, select source
    records whose images remain in the corresponding materialized image
    directory. Validate every partition before writing any corrected records:
    partitions must be nonempty, materialized images must have source records,
    record associations must remain unchanged, and a species image may not occur
    in more than one configured partition.

    Args:
        config_path: Path to the reconciliation YAML file. Config directory paths
            use the working directory. Image paths use the shared dataset root,
            the parent of input_directory; corrected CSVs retain that convention.

    Raises:
        FileNotFoundError: A required configuration, CSV, image target, or
            partition directory does not exist.
        NotADirectoryError: A directory path refers to a non-directory entry.
        KeyError: A required configuration key or record column is missing.
        ValueError: Configuration lists have different lengths, a corrected
            partition is empty, partitions overlap, or a corrected association
            is absent from the original records.
        yaml.YAMLError: The configuration file contains invalid YAML.
        PermissionError: Reading an input or writing an output is denied.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original_dir = Path(config["input_directory"]).resolve()
    corrected_dir = Path(config["output_directory"]).resolve()
    dataset_root = original_dir.parent
    relevant_subdirs = config["relevant_subdirs"]
    output_columns = config["output_columns"]
    if len(relevant_subdirs) != len(output_columns):
        raise ValueError(
            "relevant_subdirs and output_columns must contain the same number of items"
        )

    copy_config(config_path, corrected_dir)
    corrected_paths_by_species: dict[str, set[str]] = {}
    pending_outputs: list[tuple[pd.DataFrame, Path]] = []

    # Validate all configured partitions before writing any corrected records.
    for subdir, output_column in zip(relevant_subdirs, output_columns, strict=True):
        for original_species_dir in tqdm(sorted((original_dir / subdir).iterdir())):
            original_records_path = original_species_dir / "records.csv"
            if not original_records_path.is_file():
                raise ValueError(
                    f"Error, records not found for original path {original_species_dir}"
                )
            original_records = pd.read_csv(original_records_path)
            original_records["image_path"] = original_records["image_path"].apply(
                lambda path: str((dataset_root / path).resolve(strict=True))
            )

            corrected_img_dir = (
                corrected_dir / subdir / original_species_dir.name / "imgs"
            )
            materialized_images = [
                str(path.resolve(strict=True))
                for path in corrected_img_dir.iterdir()
                if path.suffix in config["allowed_suffixes"]
            ]
            corrected = original_records.loc[
                original_records["image_path"].isin(materialized_images), :
            ].copy()

            _validate_associations(corrected, original_records, materialized_images)
            if corrected.empty:
                raise ValueError(f"Nothing {subdir}")

            species_paths = corrected_paths_by_species.setdefault(
                original_species_dir.name, set()
            )
            corrected_paths = set(corrected["image_path"])
            if species_paths.intersection(corrected_paths):
                raise ValueError(
                    "Error, some images are assigned to more than one partition"
                )
            species_paths.update(corrected_paths)

            corrected[output_column] = True
            corrected[f"not_{output_column}"] = False
            pending_outputs.append(
                (corrected, corrected_img_dir.parent / "records_corrected.csv")
            )

    for corrected, output_path in pending_outputs:
        # Absolute identities are used only for association validation, not storage.
        corrected["image_path"] = corrected["image_path"].map(
            lambda path: os.path.relpath(path, dataset_root)
        )
        corrected.to_csv(output_path)


def parse_args() -> Namespace:
    """Parse the positional reconciliation YAML configuration path.

    Returns:
        Parsed command-line arguments with config as a Path instance.

    Raises:
        SystemExit: Help is requested or command-line arguments are invalid.
    """
    parser = ArgumentParser(description="Record human corrections to filtered images.")
    parser.add_argument(
        "config", type=Path, help="Path to the reconciliation YAML file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().config)
