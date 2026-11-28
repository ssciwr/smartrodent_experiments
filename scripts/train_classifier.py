"""Train a Lightning classifier run or Optuna study from one YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.logging_setup import log_training_run
from smartrodent.stage_configurations import ClassifierConfiguration
from smartrodent.stages import CLASSIFIER
from smartrodent.training_studies import StageStudy, is_study_configuration


def main() -> None:
    """Run a study or the single configured run; report checkpoints and the study winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    if is_study_configuration(args.configuration):
        study = StageStudy.from_yaml(CLASSIFIER, args.configuration)
        log_training_run(study.output_directory, study.reference.logging)
        result = study.run()
        for classifier in result.results:
            print(f"Final checkpoint: {classifier.last_checkpoint}")
        print(f"Selected classifier checkpoint: {result.selected.best_checkpoint}")
    else:
        configuration = ClassifierConfiguration.from_yaml(args.configuration)
        log_training_run(configuration.output_directory, configuration.logging)
        result = ClassifierExperiment(configuration).run_with_result()
        print(f"Final checkpoint: {result.last_checkpoint}")


if __name__ == "__main__":
    main()
