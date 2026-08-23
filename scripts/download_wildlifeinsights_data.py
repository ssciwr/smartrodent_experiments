import argparse
import shutil
from pathlib import Path

import pandas as pd
import requests
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wildlife Insights image exports."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML config with a wildlifeinsights section.",
    )
    args = parser.parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)["wildlifeinsights"]

    outpath = Path(config["output_path"]).expanduser().resolve()
    (outpath / "images").mkdir(parents=True, exist_ok=True)

    for input_path in config["input_paths"]:
        path = Path(input_path).expanduser().resolve()
        images = pd.read_csv(path / "images.csv")

        for _, row in images.iterrows():
            # filenames repeat across images, so key by image_id instead
            dest = outpath / "images" / f"{row.image_id}{Path(row.filename).suffix}"
            if not dest.exists():
                dest.write_bytes(requests.get(row.location).content)

        # copy all the csvs in each export dir to the outpath b/c they hold
        # camera location knowledge, provenance etc for dataset building
        for csv_file in path.glob("*.csv"):
            shutil.copy2(csv_file, outpath / csv_file.name)
