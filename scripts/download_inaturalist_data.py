import argparse
from pathlib import Path
from smartrodent.inaturalist import InaturalistDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an iNaturalist dataset.")
    parser.add_argument("config", type=Path, help="Path to an iNaturalist YAML config.")
    args = parser.parse_args()
    InaturalistDataset.from_config(args.config).download()
