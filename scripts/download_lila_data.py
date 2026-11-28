"""Command-line entry point for downloading the configured LILA datasets."""

import argparse
from pathlib import Path

from smartrodent.lila import LilaDataset
from smartrodent.logging_setup import log_to_console_and_file


def main(config_path: Path) -> None:
    """Build the downloader, log next to its dataset, and download.

    Args:
        config_path: YAML configuration containing the ``data.lila`` section.
    """
    dataset = LilaDataset.from_config(config_path)
    log_to_console_and_file(dataset.output_path / "lila_dataset.log")
    dataset.download()


def _parse_args() -> argparse.Namespace:
    """Parse the LILA downloader's command-line arguments."""
    parser = argparse.ArgumentParser(description="Download LILA camera-trap images.")
    parser.add_argument("config", type=Path, help="Path to a LILA YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
