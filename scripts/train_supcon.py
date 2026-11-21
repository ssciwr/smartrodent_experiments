"""Run standalone multiview SupCon training from a YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.supcon_training import run_training


def _parse_args() -> argparse.Namespace:
    """Parse the positional experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to a standalone SupCon YAML configuration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    for final_checkpoint in run_training(_parse_args().config):
        print("Final checkpoint:", final_checkpoint)
