"""Train a native Ultralytics classifier run or Optuna study from one YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
)
from smartrodent.training_studies import ClassifierStudy, is_study_configuration


def main() -> None:
    """Run a study or the single configured run; report checkpoints and the study winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    if is_study_configuration(args.configuration):
        result = ClassifierStudy.from_yaml(args.configuration).run()
        for classifier in result.results:
            print(f"Final checkpoint: {classifier.last_checkpoint}")
        print(f"Selected classifier checkpoint: {result.selected.best_checkpoint}")
    else:
        configuration = ClassifierConfiguration.from_yaml(args.configuration)
        result = ClassifierExperiment(configuration).run_with_result()
        print(f"Final checkpoint: {result.last_checkpoint}")


if __name__ == "__main__":
    main()
