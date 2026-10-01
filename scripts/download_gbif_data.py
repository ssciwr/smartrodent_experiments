"""Command-line entry point for downloading the configured GBIF dataset."""

import argparse
from pathlib import Path

from smartrodent.gbif import GbifDataset


def main(config_path: Path) -> None:
    """Build and run a GBIF dataset downloader.

    Args:
        config_path: YAML configuration containing the ``data.gbif`` section.
    """
    dataset_client = GbifDataset.from_config(config_path)
    dataset_client.download()


def _parse_args() -> argparse.Namespace:
    """Parse the GBIF downloader's command-line arguments."""
    parser = argparse.ArgumentParser(description="Download a GBIF dataset.")
    parser.add_argument("config", type=Path, help="Path to a GBIF YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
