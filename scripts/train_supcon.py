"""Run standalone multiview SupCon training, or an Optuna study, from a YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.hyperparameter_search import SupConStudy, is_study_configuration
from smartrodent.supcon_training import run_training


def _parse_args() -> argparse.Namespace:
    """Parse the positional experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to a standalone SupCon YAML configuration"
    )
    return parser.parse_args()


def main() -> None:
    """Run a study for configurations with a tuning section, else the configured runs."""
    config = _parse_args().config
    if is_study_configuration(config):
        result = SupConStudy.from_yaml(config).run()
        for trial in result.results:
            print("Final checkpoint:", trial.last_checkpoint)
        print("Selected checkpoint:", result.selected.best_checkpoint)
    else:
        for final_checkpoint in run_training(config):
            print("Final checkpoint:", final_checkpoint)


if __name__ == "__main__":
    main()
