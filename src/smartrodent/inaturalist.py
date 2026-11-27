"""Download licensed iNaturalist observations into a species-organized dataset."""

from __future__ import annotations

import ast
import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from pydantic import Field, validate_call
from pyinaturalist.session import ClientSession
from pyinaturalist.v2 import get_observations as get_observations_v2
from tqdm.auto import tqdm

from .base import DatasetLoader
from .config_loading import load_section
from .http_session import retrying_session
from .training_validation import (
    Nonblank,
    NonnegativeInt,
    NonnegativeNumber,
    PathSetting,
)

type Year = Annotated[int, Field(strict=True)]


class InaturalistDataset(DatasetLoader):
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
        years: Explicit observation years. Mutually exclusive with
            `first_year` and `last_year`.
        first_year: First year in an inclusive year range.
        last_year: Last year in an inclusive year range.
        quality_grade: iNaturalist observation quality grade to request.
        max_img_num: Maximum number of licensed images to save per species.
        max_observations: Maximum number of raw API observations requested per
            species. Duplicate observations count toward this budget.
        allowed_licenses: iNaturalist photo license codes accepted for downloads.
        cache_file: SQLite response-cache path. Defaults next to `output_path`.
        ratelimit_path: Persistent SQLite rate-limit path. Defaults next to
            `output_path` and is retained after downloads.
        backoff_factor: Exponential backoff factor for API and image retries.
        max_retries: Retries after the first attempt of each API or image request.
        config_path: Optional source YAML file to copy into `output_path`.

    Attributes:
        output_path (Path): Absolute path to the dataset root directory.
        species (list[str]): Scientific names to download, in configured order.
        years (list[int]): Resolved years sent together in each API request.
        quality_grade (str): Quality grade passed to the iNaturalist API.
        max_img_num (int): Per-species image cap.
        max_observations (int): Per-species raw observation request budget.
        allowed_licenses (set[str]): Accepted photo license codes.
        cache_file (Path): SQLite API response-cache path.
        ratelimit_path (Path): Persistent SQLite rate-limit path.
        session (ClientSession): Persistent API session owned by this downloader;
            it rate-limits, caches and retries the API requests.
        image_session (requests.Session): Retrying session for image downloads.
        config_path (Path | None): Absolute path to the source YAML file, if any.
        logger (logging.Logger): The module's logger; the entry point decides
            where its messages go.

    Raises:
        ValueError: If a setting is invalid, or the year selection is missing,
            inconsistent, or empty.
    """

    @validate_call
    def __init__(
        self,
        output_path: PathSetting,
        species: Annotated[list[Nonblank], Field(min_length=1)],
        years: list[Year] | None = None,
        first_year: Year | None = None,
        last_year: Year | None = None,
        quality_grade: Nonblank = "research",
        max_img_num: NonnegativeInt = 2000,
        max_observations: NonnegativeInt = 4000,
        allowed_licenses: tuple[Nonblank, ...] = ("cc-by-nc",),
        cache_file: PathSetting | None = None,
        ratelimit_path: PathSetting | None = None,
        backoff_factor: NonnegativeNumber = 0.5,
        max_retries: NonnegativeInt = 5,
        config_path: PathSetting | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.species = list(species)
        self.years = self._resolve_years(years, first_year, last_year)
        self.quality_grade = quality_grade
        self.max_img_num = max_img_num
        self.max_observations = max_observations
        configured_licenses = list(dict.fromkeys(allowed_licenses))
        self.allowed_licenses = set(configured_licenses)
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )

        self.output_path.mkdir(parents=True, exist_ok=True)
        default_prefix = self.output_path.parent / f".{self.output_path.name}"
        self.cache_file = (
            Path(cache_file).expanduser().resolve()
            if cache_file is not None
            else Path(f"{default_prefix}_api_requests.db")
        )
        self.ratelimit_path = (
            Path(ratelimit_path).expanduser().resolve()
            if ratelimit_path is not None
            else Path(f"{default_prefix}_api_ratelimit.db")
        )
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.ratelimit_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = ClientSession(
            cache_file=self.cache_file,
            ratelimit_path=str(self.ratelimit_path),
            backoff_factor=backoff_factor,
            max_retries=max_retries,
        )
        # pyinaturalist 0.21.1 validates uppercase licenses while API v2 accepts
        # lowercase values. Session parameters bypass that incompatible validator.
        self.session.params["photo_license"] = ",".join(configured_licenses)
        # Images come from a CDN, not the API: no rate limit or response cache, and
        # caching image bytes in SQLite would only grow the cache file.
        self.image_session = retrying_session(max_retries, backoff_factor)
        self.logger = logging.getLogger(__name__)

        # Keep setup local and deterministic; downloading remains explicit.
        self._copy_config()

    @classmethod
    def from_config(cls, config_path: str | Path) -> InaturalistDataset:
        """Build a dataset downloader from the ``data.inaturalist`` YAML mapping.

        The configuration file is copied into the dataset root so a download can
        always be traced back to it.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A configured dataset downloader ready for `download`.

        Raises:
            FileNotFoundError: If `config_path` does not exist.
            ValueError: If the section is missing or its settings are invalid.
        """
        path = Path(config_path).expanduser().resolve()
        return cls(**load_section(path, "data", "inaturalist"), config_path=path)

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
        range_parts = (first_year is not None) + (last_year is not None)

        if range_parts and explicit_years:
            raise ValueError("years and first_year/last_year are mutually exclusive")
        if range_parts != 2 and not explicit_years:
            raise ValueError("must have either years or first_year and last_year")
        if range_parts == 2:
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
        """Fetch a bounded random v2 sample and flatten it to photo rows.

        All configured years and allowed photo licenses are sent together. Raw
        observations, including duplicates, count toward `max_observations`.
        Individual photo rows are then deduplicated, license-filtered, and capped.

        Args:
            species: Scientific name to request from iNaturalist.

        Returns:
            One row per eligible photo, preserving the established dotted-column
            schema, or a readable header-only frame when no photo is eligible.
        """
        records: list[dict[str, Any]] = []
        remaining = self.max_observations
        page = 1
        with tqdm(
            total=self.max_observations, desc=f"Fetching {species} records"
        ) as progress:
            while remaining > 0:
                page_size = min(200, remaining)
                self.logger.info(
                    "Fetching up to %s %s observations on page %s",
                    page_size,
                    species,
                    page,
                )
                # The API session retries transient failures itself; anything it
                # gives up on propagates, so saved records stay as they are.
                response = get_observations_v2(
                    taxon_name=species,
                    quality_grade=self.quality_grade,
                    year=self.years,
                    photos=True,
                    order_by="random",
                    page=page,
                    per_page=page_size,
                    fields="all",
                    session=self.session,
                )
                page_records = response.get("results", [])[:remaining]
                records.extend(page_records)
                retrieved = len(page_records)
                remaining -= retrieved
                progress.update(retrieved)

                total_results = response.get("total_results")
                population_exhausted = (
                    isinstance(total_results, int) and len(records) >= total_results
                )
                if retrieved < page_size or population_exhausted:
                    break
                page += 1

        # Keep one row per photo, retaining its original position for filenames.
        photo_records = []
        for record in records:
            observation = {
                key: value for key, value in record.items() if key != "photos"
            }
            for photo_index, photo in enumerate(record.get("photos", [])):
                photo_records.append(
                    {**observation, "photo_index": photo_index, "photo": photo}
                )
        records_df = pd.json_normalize(photo_records)
        required_columns = [
            "id",
            "photo_index",
            "photo.url",
            "photo.license_code",
        ]
        if records_df.empty or "photo.license_code" not in records_df.columns:
            return pd.DataFrame(columns=required_columns)

        records_df = records_df[
            records_df["photo.license_code"].isin(self.allowed_licenses)
        ]
        records_df = records_df.drop_duplicates(subset=["id", "photo_index"]).head(
            self.max_img_num
        )
        return records_df.reset_index(drop=True)

    def _download_photo(self, photo_url: str, image_path: Path) -> None:
        """Download one photo's large variant to ``image_path``.

        The API returns thumbnail URLs; they are rewritten to the large variant.

        Args:
            photo_url: Thumbnail URL from the photo record.
            image_path: Where the image is written.

        Raises:
            requests.HTTPError: If the image request still fails after retries.
            requests.ConnectionError: If the server stays unreachable.
        """
        response = self.image_session.get(
            photo_url.replace("square", "large"), timeout=30
        )
        response.raise_for_status()
        image_path.write_bytes(response.content)

    def _download_species_images(
        self, records_df: pd.DataFrame, images_path: Path
    ) -> tuple[int, pd.DataFrame]:
        """Download photo rows and annotate them in place.

        Args:
            records_df: One row per photo, with `id`, `photo_index`,
                `photo.url`, and `photo.license_code` columns.
            images_path: Directory the images are written to.

        Returns:
            The number of images now available (downloaded or already present,
            at most `max_img_num`) and the same DataFrame, with an `image_path`
            column containing paths for those photos. Skipped photos have no
            path. This method does not save the DataFrame.

        Raises:
            KeyError: If a nonempty frame is missing required photo columns.
            requests.HTTPError: If an image request still fails after retries.
        """
        if not records_df.empty:
            required = {"id", "photo_index", "photo.url"}
            missing = required.difference(records_df.columns)
            if missing:
                raise KeyError(f"Missing photo columns: {sorted(missing)}")
        if "image_path" not in records_df.columns:
            records_df["image_path"] = pd.Series(
                None, index=records_df.index, dtype=object
            )
        path_column = records_df.columns.get_loc("image_path")

        available_images = 0
        for row in tqdm(range(len(records_df))):
            if available_images >= self.max_img_num:
                break
            record = records_df.iloc[row]
            photo_url = record["photo.url"]
            if (
                pd.isna(record["id"])
                or pd.isna(record["photo_index"])
                or not isinstance(photo_url, str)
                or not photo_url
            ):
                continue

            # CSV ids can become floats when another row has a missing id.
            image_path = (
                images_path / f"{int(record['id'])}_{int(record['photo_index'])}.jpg"
            )
            if not image_path.exists():
                self._download_photo(photo_url, image_path)
            available_images += 1
            records_df.iat[row, path_column] = str(image_path)
        return available_images, records_df

    def download(self) -> None:
        """Retrieve records for all species before downloading any images.

        Raises:
            requests.HTTPError: If a record or image request fails. Saved
                records and images remain available for a subsequent run.
        """
        try:
            self.retrieve_records()
            self.download_images()
        finally:
            self.image_session.close()
            cache_path = Path(self.session.cache.db_path)
            self.session.close()
            for path in (
                cache_path,
                Path(f"{cache_path}-wal"),
                Path(f"{cache_path}-shm"),
            ):
                path.unlink(missing_ok=True)

    def retrieve_records(self) -> None:
        """Retrieve and save records for all species without downloading images.

        Existing readable ``records.csv`` files are reused, including empty
        tables with column headers. Delete a cache to retrieve fresh records.

        Raises:
            requests.HTTPError: If an observation request returns an error.
        """
        for species in self.species:
            self.logger.info(f"species: {species}")
            species_path = self.output_path / species
            species_path.mkdir(parents=True, exist_ok=True)

            try:
                pd.read_csv(
                    species_path / "records.csv",
                    converters={"photos": ast.literal_eval},
                )
            except FileNotFoundError:
                self.logger.info("Retrieving species records")
                records_df = self._get_species_records(species)
                records_df.to_csv(species_path / "records.csv", index=False)
            else:
                self.logger.info(
                    "Found existing records.csv for %s, skipping retrieval", species
                )

    def download_images(self) -> None:
        """Download allowed photos from saved records, skipping existing images.

        No species records are retrieved during this phase. Saved image paths
        are relative to the shared dataset root (the parent of ``output_path``).

        Raises:
            FileNotFoundError: If a species has no saved records. Run
                ``retrieve_records()`` first or provide its ``records.csv``.
            requests.HTTPError: If an image request returns an error status.
        """
        for species in self.species:
            species_path = self.output_path / species
            records_path = species_path / "records.csv"
            try:
                # Restore photo lists stored as Python literals in the CSV.
                records_df = pd.read_csv(
                    records_path, converters={"photos": ast.literal_eval}
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Missing records for {species}: {records_path}. "
                    "Run retrieve_records() before download_images()."
                ) from exc

            images_path = species_path / "imgs"
            images_path.mkdir(exist_ok=True)
            self.logger.info("Downloading images for %s", species)
            downloaded, records_df = self._download_species_images(
                records_df, images_path
            )
            # Skipped or undownloaded photos lack a path. na_action="ignore" retains
            # those missing values and rows without warning or filling in an image
            # path that was never established.
            records_df["image_path"] = records_df["image_path"].map(
                lambda path: os.path.relpath(
                    (self.output_path.parent / path).resolve(),
                    self.output_path.parent,
                ),
                na_action="ignore",
            )
            records_df.to_csv(records_path, index=False)
            self.logger.info("%s images available for %s", downloaded, species)
