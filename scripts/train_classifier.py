"""Train a native Ultralytics head-only classifier sweep or Optuna study from one YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.classifier_sweeps import (
    ClassifierSweepConfiguration,
    ClassifierSweepExperiment,
)
from smartrodent.hyperparameter_search import ClassifierStudy, is_study_configuration


def main() -> None:
    """Run a study or expand configured runs; report every checkpoint and the validation winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    if is_study_configuration(args.configuration):
        result = ClassifierStudy.from_yaml(args.configuration).run()
        classifiers, selected = result.results, result.selected
    else:
        configuration = ClassifierSweepConfiguration.from_yaml(args.configuration)
        sweep = ClassifierSweepExperiment(configuration).run()
        classifiers, selected = sweep.classifier_results, sweep.selected_classifier
    for classifier in classifiers:
        print(f"Final checkpoint: {classifier.last_checkpoint}")
    if selected is not None:
        print(f"Selected classifier checkpoint: {selected.best_checkpoint}")


if __name__ == "__main__":
    main()
