"""Run the SupCon stage, select its best encoder, and train the classifier stage on it."""

import argparse
from pathlib import Path

from smartrodent.logging_setup import log_training_run
from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment


def main() -> None:
    """Load one unified configuration and report the completed CE checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    configuration = TwoStageConfiguration.from_yaml(args.configuration)
    log_training_run(configuration.output_directory, configuration.logging)
    result = TwoStageExperiment(configuration).run()
    for classifier in result.classifier_results:
        print(f"Final classifier checkpoint: {classifier.last_checkpoint}")
    print(
        f"Selected classifier checkpoint: {result.selected_classifier.best_checkpoint}"
    )


if __name__ == "__main__":
    main()
