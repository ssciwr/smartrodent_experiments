"""Run standalone multiview SupCon training, or an Optuna study, from a YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.stage_configurations import SupConConfiguration
from smartrodent.stages import SUPCON
from smartrodent.supcon_training import SupConExperiment
from smartrodent.training_studies import StageStudy, is_study_configuration


def _parse_args() -> argparse.Namespace:
    """Parse the positional experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="Path to a standalone SupCon YAML configuration"
    )
    return parser.parse_args()


def main() -> None:
    """Run a study for configurations with a tuning section, else the single run."""
    config = _parse_args().config
    if is_study_configuration(config):
        result = StageStudy.from_yaml(SUPCON, config).run()
        for trial in result.results:
            print("Final checkpoint:", trial.last_checkpoint)
        print("Selected checkpoint:", result.selected.best_checkpoint)
    else:
        experiment = SupConExperiment(SupConConfiguration.from_yaml(config))
        print("Final checkpoint:", experiment.run())


if __name__ == "__main__":
    main()
