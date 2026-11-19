"""Run a SupCon sweep, select its best encoder, and train a CE classifier sweep."""

import argparse
from pathlib import Path
import sys

# SupCon's trusted trainer still lives under scripts, outside the installed src
# package. Make the repository importable when invoked as python scripts/... .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment


def main() -> None:
    """Load one unified configuration and report the completed CE checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    result = TwoStageExperiment(TwoStageConfiguration.from_yaml(args.configuration)).run()
    for classifier in result.classifier_results:
        print(f"Final classifier checkpoint: {classifier.last_checkpoint}")
    print(f"Selected classifier checkpoint: {result.selected_classifier.best_checkpoint}")


if __name__ == "__main__":
    main()
