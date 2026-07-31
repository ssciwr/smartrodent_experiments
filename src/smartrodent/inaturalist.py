"""Download licensed iNaturalist observations into a species-organized dataset."""

from __future__ import annotations
from collections.abc import Sequence

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from pyinaturalist import get_observations
from tqdm.auto import tqdm

from .base import Configurable


class InaturalistDataset(Configurable):
    """Configure and download a reproducible iNaturalist image dataset.

    The constructor performs only local setup: it validates the requested
    years, creates the output directory, and optionally copies the YAML
    configuration there for provenance. Network requests happen only when
    `download` is called.

    Example:
        >>> dataset = InaturalistDataset(
        ...     "datasets/raw", ["Mus musculus"], first_year=2020, last_year=2022
        ... )
        >>> dataset.download()  # doctest: +SKIP

    Args:
        output_path: Directory in which species folders and metadata are saved.
        species: Scientific names to retrieve from iNaturalist.
        years: Explicit observation years. Mutually exclusive with a complete
            `first_year`/`last_year` range.
        first_year: First year in an inclusive year range.
        last_year: Last year in an inclusive year range.
        quality_grade: iNaturalist observation quality grade to request.
        seed: Seed used to shuffle observations before applying the image limit.
        max_img_num: Maximum number of licensed images to save per species.
        allowed_licenses: iNaturalist license codes accepted for downloads.
        config_path: Optional source YAML file to copy into `output_path`.

    Attributes:
        output_path (Path): Absolute path to the dataset root directory.
        species (list[str]): Scientific names to download, in configured order.
        years (list[int]): Resolved observation years, one API request each.
        quality_grade (str): Quality grade passed to the iNaturalist API.
        seed (int): Random seed for the deterministic record shuffle.
        max_img_num (int): Per-species image cap.
        allowed_licenses (set[str]): License codes accepted for downloads.
        config_path (Path | None): Absolute path to the source YAML file, if any.
        logger (logging.Logger): Logger named after this class.

    Raises:
        ValueError: If the year selection is missing, inconsistent, or empty,
            if `species` is empty, or if `max_img_num` is negative.
    """

    def __init__(
        self,
        output_path: str | Path,
        species: Sequence[str],
        years: Sequence[int] | None = None,
        first_year: int | None = None,
        last_year: int | None = None,
        quality_grade: str = "research",
        seed: int = 42,
        max_img_num: int = 2000,
        allowed_licenses: Sequence[str] = ("cc-by-nc",),
        config_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.species = list(species)
        self.years = self._resolve_years(years, first_year, last_year)
        self.quality_grade = quality_grade
        self.seed = seed
        self.max_img_num = max_img_num
        self.allowed_licenses = set(allowed_licenses)
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.max_img_num < 0:
            raise ValueError("max_img_num must be zero or greater")
        if not self.species:
            raise ValueError("species must contain at least one scientific name")

        # Keep setup local and deterministic; downloading remains explicit.
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._copy_config()

    @classmethod
    def from_config(cls, config_path: str | Path) -> InaturalistDataset:
        """Build a dataset downloader from an iNaturalist YAML configuration.

        The loader accepts either a top-level `inaturalist` mapping or a
        configuration that is itself that mapping. `allowed_licences` is
        supported as a backwards-compatible spelling for `allowed_licenses`.
        Every setting other than `output_path` and `species` falls back to the
        constructor default when it is absent. The configuration file is copied
        into the dataset root so a download can always be traced back to it.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A configured dataset downloader ready for `download`.

        Raises:
            FileNotFoundError: If `config_path` does not exist.
            TypeError: If the file is empty or neither it nor its `inaturalist`
                entry is a mapping.
            ValueError: If `output_path` or `species` is missing, or if the
                remaining settings fail constructor validation.
        """
        path = Path(config_path).expanduser().resolve()
        with path.open(encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file)

        if not isinstance(loaded_config, dict):
            raise TypeError(f"Expected a mapping in configuration file {path}")

        config = loaded_config.get("inaturalist", loaded_config)
        if not isinstance(config, dict):
            raise TypeError("The 'inaturalist' configuration must be a mapping")

        try:
            output_path = config["output_path"]
            species = config["species"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required iNaturalist setting: {exc.args[0]}"
            ) from exc

        return cls(
            output_path=output_path,
            species=species,
            years=config.get("years"),
            first_year=config.get("first_year"),
            last_year=config.get("last_year"),
            quality_grade=config.get("quality_grade", "research"),
            seed=config.get("seed", 42),
            max_img_num=config.get("max_img_num", 2000),
            allowed_licenses=config.get(
                "allowed_licenses", config.get("allowed_licences", ("cc-by-nc",))
            ),
            config_path=path,
        )

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

    def _get_species_records(self, species: str) -> pd.DataFrame:
        """Fetch and deterministically shuffle observation records for a species.

        One paginated API request is made per configured year, and the combined
        results are flattened into a table. Shuffling with `seed` keeps the
        selection reproducible while stopping the API's ordering from biasing a
        dataset that is later capped by `max_img_num`.

        Args:
            species: Scientific name to request from iNaturalist.

        Returns:
            The flattened observation records in shuffled order, or an empty
            frame if no observation was found. Columns follow the iNaturalist
            response, with nested fields flattened to dotted names.
        """
        records: list[dict[str, Any]] = []
        for year in self.years:
            self.logger.info("Fetching %s observations from %s", species, year)
            response = get_observations(
                taxon_name=species,
                quality_grade=self.quality_grade,
                per_page=200,
                year=year,
                page="all",
                photos=True,
                geo=True,
            )
            records.extend(response.get("results", []))

        records_df = pd.json_normalize(records)
        if records_df.empty:
            return records_df

        # Random sampling prevents the API's ordering from biasing a capped dataset.
        return records_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def _download_photo(
        self, photo: dict[str, Any], images_path: Path, observation_id: int, index: int
    ) -> bool:
        """Download a single photo if its license allows it.

        The thumbnail URL returned by the API is rewritten to the large variant
        before the request, and the image is saved as
        `{observation_id}_{index}.jpg`.

        Args:
            photo: One entry of an observation's `photos` list.
            images_path: Directory the image is written to.
            observation_id: Identifier of the owning observation.
            index: Position of the photo within its observation.

        Returns:
            True if an image was written, False if the photo was skipped
            because of its license or a missing URL.

        Raises:
            requests.HTTPError: If the image request returns an error status.
        """
        if photo.get("license_code") not in self.allowed_licenses:
            return False

        photo_url = photo.get("url")
        if not photo_url:
            return False

        response = requests.get(photo_url.replace("square", "large"), timeout=30)
        response.raise_for_status()
        image_path = images_path / f"{observation_id}_{index}.jpg"
        image_path.write_bytes(response.content)
        return True

    def _download_species_images(
        self, records_df: pd.DataFrame, images_path: Path
    ) -> int:
        """Download up to `max_img_num` allowed photos from a species' records.

        Records are processed in the shuffled order they arrive in, and records
        without a usable id or photo list are skipped. The cap is checked both
        between records and between the photos of one record, so an observation
        with many photos cannot overshoot it.

        Args:
            records_df: Observation records, as returned by
                `_get_species_records`.
            images_path: Directory the images are written to.

        Returns:
            The number of images actually written.

        Raises:
            requests.HTTPError: If an image request returns an error status.
        """
        downloaded_images = 0
        for _, record in tqdm(records_df.iterrows(), total=len(records_df)):
            if downloaded_images >= self.max_img_num:
                break

            photos = record.get("photos", [])
            observation_id = record.get("id")
            if not isinstance(photos, list) or pd.isna(observation_id):
                continue

            # A record without an id turns the whole column into floats, so the
            # id is normalised back to an integer for the file name.
            observation_id = int(observation_id)
            for index, photo in enumerate(photos):
                if downloaded_images >= self.max_img_num:
                    break
                if isinstance(photo, dict) and self._download_photo(
                    photo, images_path, observation_id, index
                ):
                    downloaded_images += 1

        return downloaded_images

    def download(self) -> None:
        """Fetch records and download allowed photos for every configured species.

        Each species receives a `records.csv` file and an `imgs` directory below
        the dataset root. Records are written before any image is downloaded, so
        they are saved even when no photo has an allowed license. Species are
        processed one after another and existing files are overwritten, which
        makes a re-run of an interrupted download resume at species granularity.

        Raises:
            requests.HTTPError: If an image request returns an error status.
                Species processed before the failure keep their downloaded data.
        """
        for species in self.species:
            species_path = self.output_path / species
            images_path = species_path / "imgs"
            species_path.mkdir(parents=True, exist_ok=True)
            images_path.mkdir(exist_ok=True)

            records_df = self._get_species_records(species)
            records_df.to_csv(species_path / "records.csv", index=False)
            downloaded = self._download_species_images(records_df, images_path)
            self.logger.info("Downloaded %s images for %s", downloaded, species)
