"""Command-line entry point for downloading the configured iNaturalist dataset."""

import argparse
from pathlib import Path

from smartrodent.inaturalist import InaturalistDataset
from smartrodent.logging_setup import log_to_console_and_file


def main(config_path: Path) -> None:
    """Build the downloader, log next to its dataset, and download.

    Args:
        config_path: YAML configuration containing the ``data.inaturalist`` section.
    """
    dataset = InaturalistDataset.from_config(config_path)
    log_to_console_and_file(dataset.output_path / "inaturalist_dataset.log")
    dataset.download()


def _parse_args() -> argparse.Namespace:
    """Parse the downloader's command-line arguments."""
    parser = argparse.ArgumentParser(description="Download an iNaturalist dataset.")
    parser.add_argument("config", type=Path, help="Path to an iNaturalist YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args().config)
