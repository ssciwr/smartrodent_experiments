"""Train a standalone CE classifier sweep from one YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.classifier_sweeps import ClassifierSweepConfiguration, ClassifierSweepExperiment


def main() -> None:
    """Expand configured runs and report every checkpoint and the validation winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    configuration = ClassifierSweepConfiguration.from_yaml(args.configuration)
    result = ClassifierSweepExperiment(configuration).run()
    for classifier in result.classifier_results:
        print(f"Final checkpoint: {classifier.last_checkpoint}")
    if result.selected_classifier is not None:
        print(f"Selected classifier checkpoint: {result.selected_classifier.best_checkpoint}")


if __name__ == "__main__":
    main()
