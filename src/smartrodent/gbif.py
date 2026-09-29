"""Download licensed gbif observations into a species-organized dataset."""

from __future__ import annotations
from collections.abc import Sequence

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from pygbif import occurrences as occ
from pygbif import registry

from tqdm.auto import tqdm
import os
from .base import DatasetLoader


class GbifDataset(DatasetLoader):
    def __init__(
        self,
        output_path: str | Path,
        species: Sequence[str],
        years: Sequence[int] | None = None,
        first_year: int | None = None,
        last_year: int | None = None,
        seed: int = 42,
        max_img_num: int = 2000,
        allowed_licenses: Sequence[str] = ("cc-by-nc",),
        config_path: str | Path | None = None,
        media_type: str = "StillImage",
    ):
        """_summary_

        Args:
            output_path (str | Path): _description_
            species (Sequence[str]): _description_
            years (Sequence[int] | None, optional): _description_. Defaults to None.
            first_year (int | None, optional): _description_. Defaults to None.
            last_year (int | None, optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to 42.
            max_img_num (int, optional): _description_. Defaults to 2000.
            allowed_licenses (Sequence[str], optional): _description_. Defaults to ("cc-by-nc",).
            config_path (str | Path | None, optional): _description_. Defaults to None.
            media_type (str, optional): _description_. Defaults to "StillImage".

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        self.output_path = Path(output_path).expanduser().resolve()
        self.species = list(species)
        self.years = self._resolve_years(years, first_year, last_year)
        self.seed = seed
        self.max_img_num = max_img_num
        self.allowed_licenses = set(allowed_licenses)
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )
        self.media_type = media_type
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(Path(output_path) / "gbif_dataset.log")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        if self.max_img_num < 0:
            raise ValueError("max_img_num must be zero or greater")
        if not self.species:
            raise ValueError("species must contain at least one scientific name")

        # Keep setup local and deterministic; downloading remains explicit.
        self._copy_config()

    @staticmethod
    def _resolve_years(
        years: Sequence[int] | None, first_year: int | None, last_year: int | None
    ) -> list[int]:
        """Validate explicit years or turn an inclusive range into a list.

        Exactly one of the two ways to select years must be given: a non-empty
        `years` sequence, or both `first_year` and `last_year`.

        Args:
            years: Explicit observation years, or None. An empty sequence counts
                as no selection, which lets a config file keep an unused
                `years: []` entry next to a year range.
            first_year: First year of an inclusive range, or None.
            last_year: Last year of an inclusive range, or None.

        Returns:
            The explicit years unchanged, or the expanded inclusive range.

        Raises:
            ValueError: If both selections are given, if neither is given, or if
                `first_year` is greater than `last_year`.
        """
        explicit_years = list(years or [])
        has_range = first_year is not None and last_year is not None

        if has_range and explicit_years:
            raise ValueError("years and first_year/last_year are mutually exclusive")
        if not has_range and not explicit_years:
            raise ValueError("must have either years or first_year and last_year")
        if has_range:
            if first_year > last_year:
                raise ValueError("first_year must be less than or equal to last_year")
            return list(range(first_year, last_year + 1))

        return explicit_years

    def _copy_config(self) -> None:
        """Copy the configuration into the dataset root when one was supplied.

        Does nothing when no configuration was given, or when the file already
        lives in the dataset root and would be copied onto itself.
        """
        if self.config_path is None:
            return

        destination = self.output_path / self.config_path.name
        if self.config_path != destination:
            shutil.copy2(self.config_path, destination)

    def _get_all_records(self, **params) -> list[dict[str, Any]]:
        """Fetch all matching occurrences and flatten them into image records.

        GBIF searches and paginates occurrence records, but an occurrence can
        contain multiple media items. This method emits one dictionary per
        still image and retains the parent occurrence key as ``source_group_id``.

        Returns:
            Image-level records containing the image URL, licence, attribution,
            source information, and source-dataset citation.
        """
        image_records: list[dict[str, Any]] = []
        offset = 0
        page_size = 300

        while True:
            response = occ.search(
                **params,
                limit=page_size,
                offset=offset,
            )
            page = response["results"]

            for occurrence in page:
                dataset_key = occurrence.get("datasetKey")
                dataset = registry.datasets(uuid=dataset_key) if dataset_key else {}
                citation = dataset.get("citation", {}).get("text")
                doi = dataset.get("doi")
                for media in occurrence.get("media", []):
                    if media.get("type") != self.media_type:
                        continue

                    image_url = media.get("identifier")
                    if not image_url:
                        continue

                    image_records.append(
                        {
                            "image_url": image_url,
                            "image_license": media.get("license"),
                            "creator": media.get("creator"),
                            "source_url": media.get("references")
                            or occurrence.get("references"),
                            "source_group_id": occurrence.get("key"),
                            "dataset_key": dataset_key,
                            "citation": citation,
                            "doi": doi,
                        }
                    )

            if response["endOfRecords"] or not page:
                break

            offset += len(page)

        return image_records

    def _get_species_records(self, species: str) -> pd.DataFrame:
        """_summary_

        Args:
            species (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        records: list[dict[str, Any]] = []
        for year in self.years:
            self.logger.info("Fetching %s observations from %s", species, year)
            records_for_year = self._get_all_records(
                scientificName=species,
                mediatype=self.media_type,
                year=year,
            )
            records.extend(records_for_year)

        records_df = pd.json_normalize(records)
        if records_df.empty:
            return records_df
        # make pandas dataframe from json
        # we need to handle nested records in the answer
        # Random sampling prevents the API's ordering from biasing a capped dataset.
        records_df = records_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        # cap data if bigger than limit
        if len(records_df) > self.max_img_num:
            records_df = records_df.head(self.max_img_num)

        return records_df

    def download(self) -> None:

        for sp in self.species:
            records_df = self._get_species_records(sp)
