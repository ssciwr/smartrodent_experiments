from pyinaturalist import get_observations
import json
import requests
import os
from pathlib import Path
from typing import Any
import pandas as pd


def process_config(config: dict[str, Any]):

    start_year = config.get("first_year")
    stop_year = config.get("last_year")

    if start_year is not None and stop_year is not None:
        year_range = range(start_year, stop_year + 1)

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


def fetch_image_from_inat(response: dict[str, Any]):
    pass


def download_inat_data(config_path: str):

    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r") as cfgfile:
        config = json.load(cfgfile)["data"]["inaturalist"]

    config = process_config(config)
    output_path = config["output_path"]
    quality = config["quality"]

    all_records = []
    for species in config["species"]:
        for year in config["years"]:
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
    df.to_csv(Path(output_path).resolve() / "records.csv", index=False)


if __name__ == "__main__":
    download_inat_data(
        "/mnt/dataLinux/Development/rodent_experiments/config/central_europe.json"
    )
