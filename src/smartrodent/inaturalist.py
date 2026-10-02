"""Download licensed iNaturalist observations into a species-organized dataset."""

from __future__ import annotations

import ast
import logging
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import urllib3
import yaml
from pyinaturalist.session import ClientSession
from pyinaturalist.v2 import get_observations as get_observations_v2
from tqdm.auto import tqdm

from .base import DatasetLoader


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
        years: Explicit observation years. Mutually exclusive with a complete
            `first_year`/`last_year` range.
        first_year: First year in an inclusive year range.
        last_year: Last year in an inclusive year range.
        quality_grade: iNaturalist observation quality grade to request.
        seed: Retained for backwards-compatible dataset configuration.
        max_img_num: Maximum number of licensed images to save per species.
        max_observations: Maximum number of raw API observations requested per
            species. Duplicate observations count toward this budget.
        allowed_licenses: iNaturalist photo license codes accepted for downloads.
        cache_file: SQLite response-cache path. Defaults next to `output_path`.
        ratelimit_path: Persistent SQLite rate-limit path. Defaults next to
            `output_path` and is retained after downloads.
        backoff_factor: Exponential backoff factor for API retries.
        max_retries: Maximum retries after the initial failure for API requests.
        config_path: Optional source YAML file to copy into `output_path`.

    Attributes:
        output_path (Path): Absolute path to the dataset root directory.
        species (list[str]): Scientific names to download, in configured order.
        years (list[int]): Resolved years sent together in each API request.
        quality_grade (str): Quality grade passed to the iNaturalist API.
        seed (int): Backwards-compatible dataset configuration value.
        max_img_num (int): Per-species image cap.
        max_observations (int): Per-species raw observation request budget.
        allowed_licenses (set[str]): Accepted photo license codes.
        cache_file (Path): SQLite API response-cache path.
        ratelimit_path (Path): Persistent SQLite rate-limit path.
        session (ClientSession): Persistent API session owned by this downloader.
        config_path (Path | None): Absolute path to the source YAML file, if any.
        logger (logging.Logger): Logger named after this class.

    Raises:
        ValueError: If the year selection is missing, inconsistent, or empty,
            if `species` is empty, if `max_img_num` is negative, or if
            `max_observations` is not a nonnegative integer.
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
        max_observations: int = 4000,
        allowed_licenses: Sequence[str] = ("cc-by-nc",),
        cache_file: str | Path | None = None,
        ratelimit_path: str | Path | None = None,
        backoff_factor: float = 0.5,
        max_retries: int = 5,
        config_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.species = list(species)
        self.years = self._resolve_years(years, first_year, last_year)
        self.quality_grade = quality_grade
        self.seed = seed
        self.max_img_num = max_img_num
        self.max_observations = max_observations
        configured_licenses = list(dict.fromkeys(allowed_licenses))
        self.allowed_licenses = set(configured_licenses)
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )

        if self.max_img_num < 0:
            raise ValueError("max_img_num must be zero or greater")
        if isinstance(self.max_observations, bool) or not isinstance(
            self.max_observations, int
        ):
            raise ValueError("max_observations must be an integer")
        if self.max_observations < 0:
            raise ValueError("max_observations must be zero or greater")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be a nonnegative integer")
        if max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.max_retries = max_retries
        if not self.species:
            raise ValueError("species must contain at least one scientific name")

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

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            Path(output_path) / "inaturalist_dataset.log"
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Keep setup local and deterministic; downloading remains explicit.
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

        config = loaded_config.get("data", loaded_config).get(
            "inaturalist", loaded_config
        )
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
            max_observations=config.get("max_observations", 4000),
            allowed_licenses=config.get(
                "allowed_licenses", config.get("allowed_licences", ("cc-by-nc",))
            ),
            cache_file=config.get("cache_file"),
            ratelimit_path=config.get("ratelimit_path"),
            backoff_factor=config.get("backoff_factor", 0.5),
            max_retries=config.get("max_retries", 5),
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
        consecutive_errors = 0
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
                try:
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
                    consecutive_errors = 0
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
                except requests.exceptions.RetryError as exc:
                    self.logger.error("Retry error while fetching records: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=30
                    )
                except requests.HTTPError as exc:
                    self.logger.error("Failed to fetch records: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=30
                    )
                except requests.RequestException as exc:
                    self.logger.error("Request error while fetching records: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=30
                    )
                except Exception as exc:
                    self.logger.error(
                        "Unexpected error while fetching records: %s", exc
                    )
                    raise

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

    def _retry_after_error(
        self, error: Exception, consecutive_errors: int, *, delay_seconds: int
    ) -> int:
        """Bound a manual retry loop using the configured retry budget.

        Args:
            error: The latest anticipated request failure.
            consecutive_errors: Number of failures for the current request so far.
            delay_seconds: Delay before retrying, if retries remain.

        Returns:
            Updated consecutive error count.

        Raises:
            The original request exception after ``max_retries`` retries.
        """
        next_error_count = consecutive_errors + 1
        if next_error_count > self.max_retries:
            raise error
        time.sleep(delay_seconds)
        return next_error_count

    def _download_photo(
        self, photo: dict[str, Any], images_path: Path, observation_id: int, index: int
    ) -> bool:
        """Download a single photo with a usable URL.

        The thumbnail URL returned by the API is rewritten to the large variant
        before the request, and the image is saved as
        `{observation_id}_{index}.jpg`.

        Args:
            photo: One entry of an observation's `photos` list.
            images_path: Directory the image is written to.
            observation_id: Identifier of the owning observation.
            index: Position of the photo within its observation.

        Returns:
            True if an image was written, or False for a missing URL or an
            already-existing image.

        Raises:
            requests.HTTPError: If the image request returns an error status.
        """
        photo_url = photo.get("url")
        if not photo_url:
            return False

        image_path = images_path / f"{observation_id}_{index}.jpg"
        if image_path.exists():
            return False
        else:
            response = requests.get(photo_url.replace("square", "large"), timeout=30)
            response.raise_for_status()
            image_path.write_bytes(response.content)
            return True

    def _download_species_images(
        self, records_df: pd.DataFrame, images_path: Path
    ) -> tuple[int, pd.DataFrame]:
        """Download photo rows and annotate them in place.

        Args:
            records_df: One row per photo, with `id`, `photo_index`,
                `photo.url`, and `photo.license_code` columns.
            images_path: Directory the images are written to.

        Returns:
            The number of newly written images and the same DataFrame, with
            an `image_path` column containing paths for saved or existing photos.
            Skipped photos have no path. This method does not save the DataFrame.

        Raises:
            KeyError: If a nonempty frame is missing required photo columns.
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

        downloaded_images = 0
        consecutive_errors = 0
        with tqdm(total=len(records_df)) as progress:
            i = 0
            while i < len(records_df) and downloaded_images < self.max_img_num:
                try:
                    record = records_df.iloc[i]
                    observation_id = record["id"]
                    photo_index = record["photo_index"]
                    photo_url = record["photo.url"]
                    if (
                        pd.isna(observation_id)
                        or pd.isna(photo_index)
                        or not isinstance(photo_url, str)
                        or not photo_url
                    ):
                        consecutive_errors = 0
                        i += 1
                        progress.update(1)
                        continue

                    # CSV ids can become floats when another row has a missing id.
                    observation_id = int(observation_id)
                    photo_index = int(photo_index)
                    photo = {"url": photo_url}
                    image_path = images_path / f"{observation_id}_{photo_index}.jpg"
                    if not image_path.exists():
                        # download the image if it hasn't been saved yet
                        self._download_photo(
                            photo, images_path, observation_id, photo_index
                        )
                        downloaded_images += 1
                    else:
                        # has been downloaded before
                        downloaded_images += 1

                    if image_path.exists():
                        records_df.iat[i, records_df.columns.get_loc("image_path")] = (
                            str(image_path)
                        )
                    progress.update(1)
                    i += 1
                    consecutive_errors = 0
                except requests.exceptions.RetryError as exc:
                    self.logger.error("Retry error while fetching image: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=20
                    )
                except urllib3.exceptions.MaxRetryError as exc:
                    self.logger.error(
                        "Max retries exceeded while fetching image: %s", exc
                    )
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=20
                    )
                except requests.HTTPError as exc:
                    self.logger.error("Failed to fetch image: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=20
                    )
                except requests.RequestException as exc:
                    self.logger.error("Request error while fetching image: %s", exc)
                    consecutive_errors = self._retry_after_error(
                        exc, consecutive_errors, delay_seconds=20
                    )
                except Exception as exc:
                    self.logger.error("Unexpected error while fetching image: %s", exc)
                    raise
        return downloaded_images, records_df

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

        No species records are retrieved during this phase.

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
            records_df.to_csv(records_path, index=False)
            self.logger.info("Downloaded %s images for %s", downloaded, species)
