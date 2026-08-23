import argparse
import shutil
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

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
    session = requests.Session()

    for input_path in config["input_paths"]:
        path = Path(input_path).expanduser().resolve()
        images = pd.read_csv(path / "images.csv")
        # print("# images: ", len(images))
        # print(images.loc[:, "license"].head())
        images = images[images["license"].isin(config["allowed_licences"])]
        images = (
            images.sample(frac=1, random_state=config["seed"])
            .groupby(["genus", "species"], dropna=False, sort=False)
            .head(config["max_img_num"])
        )
        print("# images: ", len(images))
        for _, row in tqdm(images.iterrows(), desc=str(input_path)):
            genus = row.genus if isinstance(row.genus, str) else "unknown"
            species = row.species if isinstance(row.species, str) else "unknown"
            species_dir = outpath / "images" / f"{genus.capitalize()} {species.lower()}"
            species_dir.mkdir(parents=True, exist_ok=True)

            # filenames repeat across images, so key by image_id instead
            dest = species_dir / f"{row.image_id}{Path(row.filename).suffix}"

            # piece of shit AI didn't do the right thing. again.
            if not dest.exists():
                response = session.get(row.location)
                response.raise_for_status()
                dest.write_bytes(response.content)
