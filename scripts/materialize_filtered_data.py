from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import yaml


def main(config_path: Path) -> None:
    """Partition species records and symlink images under a configured output.

    The original YAML is copied into output_directory for traceability. Only
    records.csv in each species directory is processed; other CSVs are ignored.
    Each output partition stores records.csv under the species directory and
    image symlinks in its imgs subdirectory.

    Args:
        config_path: YAML file containing input_dir and output_directory, with
            optional deciding_column_name_start (default: rejected). Records
            must contain an image_path column. Relative directory and image
            paths are interpreted from the working directory.

    Raises:
        NotADirectoryError: The input directory does not exist or is not a directory.
        ValueError: Image paths are missing.
        FileNotFoundError: The config, species records, or an image does not exist.
        FileExistsError: An output image or config conflicts with an existing entry.
    """
    config_contents = config_path.read_bytes()
    config = yaml.safe_load(config_contents)
    input_dir = Path(config["input_dir"]).resolve()
    output_root = Path(config["output_directory"]).resolve()
    deciding_col_start: str = config.get("deciding_column_name_start", "rejected")
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(config_contents, output_root / config_path.name)

    for path in input_dir.iterdir():
        if not path.is_dir():
            continue

        records = pd.read_csv(path / "records.csv")
        if "image_path" not in records.columns:
            raise ValueError(f"{path / 'records.csv'}: missing image_path column")
        if records["image_path"].isna().any():
            raise ValueError(
                f"{path / 'records.csv'}: image_path values must not be null"
            )

        deciding_cols = [
            column
            for column in records.columns
            if column.startswith(deciding_col_start)
        ]
        # CSVs with missing flags may load booleans as text. Only an explicit
        # True rejects a row; missing flags do not imply rejection.
        if deciding_cols:
            rejected = (
                records[deciding_cols]
                .apply(
                    lambda column: (
                        column.astype("string").str.strip().str.lower().eq("true")
                    )
                )
                .any(axis=1)
            )
        else:
            rejected = pd.Series(False, index=records.index)

        for partition, mask in (("kept", ~rejected), ("rejected", rejected)):
            output_dir = output_root / partition / path.name
            # Separate image links from metadata, even for empty partitions.
            image_dir = output_dir / "imgs"
            image_dir.mkdir(parents=True, exist_ok=True)
            filtered = records.loc[mask]
            output_csv = output_dir / (path / "records.csv").name
            for image_path in filtered["image_path"]:
                print(image_path)
                source = Path(image_path).resolve(strict=True)
                destination = image_dir / Path(image_path).name
                _symlink_image(source, destination)
            filtered.to_csv(output_csv, index=False)


def _save_config_snapshot(contents: bytes, destination: Path) -> None:
    """Save the exact config bytes without replacing a different run's snapshot."""
    if destination.exists() or destination.is_symlink():
        # Identical snapshots allow reruns, including using the saved config.
        # Never follow an output symlink when writing traceability metadata.
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != contents
        ):
            raise FileExistsError(f"Conflicting output config: {destination}")
        else:
            return
    else:
        destination.write_bytes(contents)


def _symlink_image(source: Path, destination: Path) -> None:
    """Create a link without overwriting files or links to other images."""
    # Duplicate rows and repeated runs may request the same link again.
    if destination.is_symlink() and destination.resolve() == source:
        return
    elif destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Conflicting output image: {destination}")
    else:
        destination.symlink_to(source)


def parse_args() -> Namespace:
    """Parse the materialization YAML configuration path."""
    parser = ArgumentParser(
        description="Materialize filtered data from csv files stored in a directory."
    )
    parser.add_argument(
        "config", type=Path, help="Path to the materialization YAML file."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
