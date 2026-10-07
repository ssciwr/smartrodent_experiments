"""Record human corrections to materialized kept/rejected images."""

import shutil
from argparse import ArgumentParser, Namespace
from datetime import datetime
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


def check_images_consistency(
    kept_corrected: pd.DataFrame, rejected_corrected: pd.DataFrame
):
    """Validate that corrected partitions are nonempty and do not overlap.

    Args:
        kept_corrected: Kept records containing an image_path column.
        rejected_corrected: Rejected records containing an image_path column.

    Raises:
        ValueError: Either partition is empty or an image path appears in both.
        KeyError: A nonempty partition is missing the image_path column.
    """
    # check that the rejected and kept stuff is consistent
    if len(kept_corrected) == 0:
        raise ValueError("Nothing kept")

    if len(rejected_corrected) == 0:
        raise ValueError("Nothing rejected")

    if (
        len(
            set(kept_corrected["image_path"]).intersection(
                set(rejected_corrected["image_path"])
            )
        )
        > 0
    ):
        raise ValueError(
            "Error, some images are assigned both to corrected and rejected"
        )


def check_association_consistency(
    corrected: pd.DataFrame, original_records: pd.DataFrame
):
    """Validate that corrected image associations occur in the original records.

    Check each distinct (id, photo.id, image_path) tuple and image path against
    the original records. This does not require every original row to remain
    in the corrected records or preserve duplicate counts.

    Args:
        corrected: Corrected records with id, photo.id, and image_path columns.
        original_records: Original records with the same association columns.

    Raises:
        ValueError: A corrected association or image path is absent from the
            original records.
        KeyError: Either dataframe is missing a required association column.
    """
    corrected_tuples = set(
        corrected.loc[:, ["id", "photo.id", "image_path"]].itertuples(
            name=None, index=False
        )
    )

    original_tuples = set(
        original_records.loc[:, ["id", "photo.id", "image_path"]].itertuples(
            name=None, index=False
        )
    )

    if len(corrected_tuples.difference(original_tuples)) > 0:
        raise ValueError("Association between id, photo.id, and image_path has changed")

    if (
        len(
            set(corrected["image_path"]).difference(set(original_records["image_path"]))
        )
        > 0
    ):
        raise ValueError(
            "Some images that are in the original original_records are not in the corrected original_records or vice versa"
        )


def main(config_path: Path) -> None:
    """Record human image decisions in timestamped species CSV files.

    Read input_directory and allowed_suffixes from the reconciliation YAML.
    Recover the original input_dir from materialize_filtered_data.yaml in that
    directory. For each original species directory, read original_records.csv
    and scan images directly under the materialized kept and rejected species
    directories. Resolve image paths and assign kept_human and rejected_human
    flags according to the current partitions.

    Copy the reconciliation YAML into the materialized root, write
    original_records_corrected_<timestamp>.csv into each original species
    directory, and save normalized original records as
    original_records_original.csv. Both CSV outputs include the dataframe index.

    Args:
        config_path: Path to the reconciliation YAML file. Relative directory
            and image paths are interpreted from the working directory.

    Raises:
        FileNotFoundError: A required configuration, CSV, image target, or
            partition directory does not exist.
        NotADirectoryError: A directory path refers to a non-directory entry.
        KeyError: A required configuration key or record column is missing.
        ValueError: A corrected partition is empty, partitions overlap, or a
            corrected association is absent from the original records.
        yaml.YAMLError: A configuration file contains invalid YAML.
        PermissionError: Reading an input or writing an output is denied.
    """
    # Load the corrected location and recover the original original_records location.
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    corrected_root = Path(config["input_directory"]).resolve()
    saved_config = corrected_root / "materialize_filtered_data.yaml"
    materialization = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
    original_root = Path(materialization["input_dir"]).resolve()
    output_dir = Path(config["output_directory"]).resolve()

    timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
    copy_config(config_path, corrected_root)

    # Process only original original_records.csv files, one species at a time.
    for species_dir in tqdm(sorted(original_root.iterdir())):
        # find original original_records.csv file
        original_original_records_path = species_dir / "original_records.csv"
        original_records = pd.read_csv(original_original_records_path)

        # record all image files in kept and rejected directories
        kept_dir = corrected_root / "kept" / species_dir.name / "imgs"
        rejected_dir = corrected_root / "rejected" / species_dir.name / "imgs"

        # get image names in the current species dir

        kept_images = [
            str(p.resolve(strict=True))
            for p in kept_dir.iterdir()
            if p.suffix in config["allowed_suffixes"]
        ]

        rejected_images = [
            str(p.resolve(strict=True))
            for p in rejected_dir.iterdir()
            if p.suffix in config["allowed_suffixes"]
        ]

        # process paths to make them adhere to a universal format equivalent to
        # the system's
        original_records["image_path"] = original_records["image_path"].apply(
            lambda p: str(Path(p).resolve(strict=True))
        )

        # select the kept images from the original_records
        kept_corrected = original_records.loc[
            original_records["image_path"].isin(kept_images), :
        ]

        kept_corrected["kept_human"] = True
        kept_corrected["rejected_human"] = False

        rejected_corrected = original_records.loc[
            original_records["image_path"].isin(rejected_images), :
        ]
        rejected_corrected["kept_human"] = False
        rejected_corrected["rejected_human"] = True

        check_images_consistency(kept_corrected, rejected_corrected)

        # concatenate and check that the associations between observation, photo and image hasn't changed
        corrected_records = pd.concat([kept_corrected, rejected_corrected])
        check_association_consistency(corrected_records, original_records)

        (output_dir / species_dir.name).mkdir(parents=True, exist_ok=True)

        # ... then save records
        corrected_records.to_csv(
            output_dir
            / species_dir.name
            / f"original_records_corrected_{timestamp}.csv"
        )
        original_records.to_csv(
            output_dir / species_dir.name / "original_records_original.csv"
        )


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
