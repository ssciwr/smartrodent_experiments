from pyinaturalist import get_observations
import json
import requests
import os
from pathlib import Path
from typing import Any
import pandas as pd
import shutil
from tqdm import tqdm


def process_config(config: dict[str, Any]):

    start_year = config.get("first_year")
    stop_year = config.get("last_year")

    if start_year is not None and stop_year is not None:
        year_range = list(range(start_year, stop_year + 1))

        if config.get("years") is not None and len(config["years"]) > 0:
            raise ValueError(
                "'years' and 'start_year'/'end_year' are mutually exclusive, only give one."
            )

        config["years"] = year_range
        return config

    elif (
        start_year is None
        and stop_year is not None
        or start_year is not None
        and stop_year is None
    ):
        raise ValueError(
            "One of start_year/stop_year is not given and the other is. Please delete both to use the 'years' list directly, or give both to give a range"
        )

    elif start_year is None and stop_year is None and config.get("years") is None:
        raise ValueError("Error, (start_year,stop_year) or 'years' must be given")
    else:
        return config  # do nothing


def fetch_image_from_inat(photo: dict, img_path: Path, id: int, index: int):
    if photo["license_code"] != "cc-by-nc":
        return
    else:
        photo_url = photo["url"].replace("square", "large")
        photo_path = img_path / f"{id}_{index}.jpg"
        img_bytes = requests.get(photo_url).content

        with open(photo_path, "wb") as f:
            f.write(img_bytes)


def download_inat_data(config_path: str):

    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r") as cfgfile:
        config = json.load(cfgfile)["data"]["inaturalist"]

    print("read config: ", config)
    config = process_config(config)

    print("processed config: ", config)
    output_path = Path(config["output_path"]).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, output_path)

    quality = config["quality_grade"]

    for species in config["species"]:
        all_records = []
        print(f"current species {species}")
        for year in config["years"]:
            print(f"  current year: {year}")
            response = get_observations(
                taxon_name=species,
                quality_grade=quality,
                per_page=200,
                year=year,
                page="all",
                photos=True,
                geo=True,
            )

            all_records.extend(response["results"])

        df = pd.json_normalize(all_records)

        save_path = output_path / f"{species}"
        save_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(save_path / "records.csv", index=False)

        img_path = save_path / "imgs"
        img_path.mkdir(parents=True, exist_ok=True)
        print(img_path)
        for _, row in tqdm(df.iterrows()):
            photos = row["photos"]
            id = row["id"]
            for i, photo in enumerate(photos):
                fetch_image_from_inat(photo, img_path, id, i)


if __name__ == "__main__":
    download_inat_data(
        "/mnt/dataLinux/Development/rodent_experiments/config/central_europe.json"
    )
