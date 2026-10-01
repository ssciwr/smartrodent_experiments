"""Command-line entry point for downloading the configured GBIF dataset."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from smartrodent.gbif import GbifDataset


def main(config_path: Path, env_file: Path | None = None) -> None:
    """Build and run a GBIF dataset downloader.

    Args:
        config_path: YAML configuration containing the ``data.gbif`` section.
        env_file: Explicit credentials file. Existing environment values win;
            without this option, only the existing environment is used.

    Raises:
        FileNotFoundError: If the explicitly selected credentials file is absent.
    """
    if env_file is not None:
        path = env_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GBIF credentials file does not exist: {path}")
        load_dotenv(dotenv_path=path, override=False)
    dataset_client = GbifDataset.from_config(config_path)
    dataset_client.download()


def _parse_args() -> argparse.Namespace:
    """Parse the GBIF downloader's command-line arguments."""
    parser = argparse.ArgumentParser(description="Download a GBIF dataset.")
    parser.add_argument("config", type=Path, help="Path to a GBIF YAML config.")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Explicit .env credentials file; does not override the environment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config, env_file=args.env_file)
