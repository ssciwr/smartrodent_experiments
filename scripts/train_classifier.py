"""Train a standalone CE classifier from one YAML configuration."""

import argparse
from pathlib import Path

from smartrodent.classifier_training import ClassifierConfiguration, ClassifierExperiment


def main() -> None:
    """Parse a configuration filename and report the final completed checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    configuration = ClassifierConfiguration.from_yaml(args.configuration)
    checkpoint = ClassifierExperiment(configuration).run()
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
