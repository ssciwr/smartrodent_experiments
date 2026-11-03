"""Command-line entry point for downloading the configured LILA datasets."""

import argparse
from pathlib import Path

from smartrodent.lila import LilaDataset


def main(config_path: Path) -> None:
    """Build and run a LILA dataset downloader.

    Args:
        config_path: YAML configuration containing the ``data.lila`` section.
    """
    dataset_client = LilaDataset.from_config(config_path)
    dataset_client.download()


def _parse_args() -> argparse.Namespace:
    """Parse the LILA downloader's command-line arguments."""
    parser = argparse.ArgumentParser(description="Download LILA camera-trap images.")
    parser.add_argument("config", type=Path, help="Path to a LILA YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config)
