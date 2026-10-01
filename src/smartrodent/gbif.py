"""Download licensed gbif observations into a species-organized dataset."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import requests
import yaml
from pygbif import occurrences as occ
from pygbif import registry
from pygbif import species as species_api
from tqdm.auto import tqdm

from .base import DatasetLoader
from .gbif_archive import GbifArchiveReader
from .licenses import LicenseManagerBase, create_license_manager


class GbifDataset(DatasetLoader):
    """Fetch and filter image-level GBIF records for configured species."""

    def __init__(
        self,
        output_path: str | Path,
        species: Sequence[str],
        years: Sequence[int] | None = None,
        first_year: int | None = None,
        last_year: int | None = None,
        seed: int = 42,
        max_img_num: int = 2000,
        allowed_licenses: Mapping[str, Sequence[str]] | None = None,
        media_type: str = "StillImage",
        config_path: str | Path | None = None,
        poll_interval: float = 5,
        max_wait_seconds: float = 3600,
    ):
        """Configure GBIF retrieval and image-license filtering.

        Args:
            output_path: Root directory for downloaded data and metadata.
            species: Scientific names to retrieve.
            years: Explicit observation years.
            first_year: First year of an inclusive range.
            last_year: Last year of an inclusive range.
            seed: Seed used when shuffling records.
            max_img_num: Maximum number of image records retained per species.
            allowed_licenses: Mapping from registered license families to their
                allowed canonical license identifiers. Defaults to Creative
                Commons Attribution-NonCommercial.
            media_type: GBIF media type to retrieve.
            config_path: Optional source configuration copied for provenance.
            poll_interval: Seconds between checks of an unfinished download job.
            max_wait_seconds: Maximum elapsed polling time per species.

        Raises:
            TypeError: If ``allowed_licenses`` is not a mapping or contains
                invalid family or license values.
            ValueError: If the license mapping is empty, a family or license is
                unsupported, the year selection is invalid, ``max_img_num`` is
                negative, no species are configured, or a polling setting is
                not finite and positive.
        """
        for name, value in (
            ("poll_interval", poll_interval),
            ("max_wait_seconds", max_wait_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        license_policy = (
            {"creative-commons": ("cc-by-nc",)}
            if allowed_licenses is None
            else allowed_licenses
        )
        if not isinstance(license_policy, Mapping):
            raise TypeError("allowed_licenses must be a mapping")
        if not license_policy:
            raise ValueError("allowed_licenses must contain at least one family")

        self._license_managers: tuple[LicenseManagerBase, ...] = tuple(
            create_license_manager(family, licenses)
            for family, licenses in license_policy.items()
        )
        self.output_path = Path(output_path).expanduser().resolve()
        self.species = list(species)
        self.years = self._resolve_years(years, first_year, last_year)
        self.seed = seed
        self.max_img_num = max_img_num
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

    @classmethod
    def from_config(cls, config_path: str | Path) -> GbifDataset:
        """Build a GBIF dataset downloader from the project YAML configuration.

        The configuration must contain a ``data.gbif`` mapping. ``output_path``
        and ``species`` are required; all other supported settings use the
        constructor defaults when omitted. The configuration's ``maxlen`` key
        maps to the constructor's ``max_img_num`` argument. The source file is
        copied into the dataset root for provenance.

        Args:
            config_path: Path to a YAML file containing ``data.gbif`` settings.

        Returns:
            A configured GBIF dataset downloader.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            TypeError: If the configuration or its required sections are not
                mappings.
            ValueError: If a required GBIF setting is absent or constructor
                validation fails.
        """
        path = Path(config_path).expanduser().resolve()
        with path.open(encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file)

        if not isinstance(loaded_config, dict):
            raise TypeError(f"Expected a mapping in configuration file {path}")

        data_config = loaded_config.get("data")
        if not isinstance(data_config, dict):
            raise TypeError("The 'data' configuration must be a mapping")

        config = data_config.get("gbif")
        if not isinstance(config, dict):
            raise TypeError("The 'data.gbif' configuration must be a mapping")

        try:
            output_path = config["output_path"]
            species = config["species"]
        except KeyError as exc:
            raise ValueError(f"Missing required GBIF setting: {exc.args[0]}") from exc

        return cls(
            output_path=output_path,
            species=species,
            years=config.get("years"),
            first_year=config.get("first_year"),
            last_year=config.get("last_year"),
            seed=config.get("seed", 42),
            max_img_num=config.get("maxlen", 2000),
            allowed_licenses=config.get("allowed_licenses"),
            media_type=config.get("media_type", "StillImage"),
            config_path=path,
            poll_interval=config.get("poll_interval", 5),
            max_wait_seconds=config.get("max_wait_seconds", 3600),
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

    @staticmethod
    def _record_columns() -> tuple[str, ...]:
        """Return the stable schema used for image-level GBIF records."""
        return (
            "source_group_id",
            "scientific_name",
            "accepted_scientific_name",
            "taxon_key",
            "species",
            "species_key",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
            "year",
            "event_date",
            "basis_of_record",
            "country_code",
            "decimal_latitude",
            "decimal_longitude",
            "occurrence_source_url",
            "image_url",
            "image_license",
            "image_creator",
            "image_references",
            "image_type",
            "image_format",
            "image_created",
            "dataset_key",
            "publishing_org_key",
            "dataset_citation",
            "dataset_doi",
        )

    def _resolve_taxon_key(self, scientific_name: str) -> str:
        """Require an unambiguous species match and resolve synonyms to accepted keys."""
        match = species_api.name_backbone(scientificName=scientific_name, strict=True)
        usage = match.get("usage", {})
        diagnostics = match.get("diagnostics", {})
        if match.get("synonym"):
            accepted = match.get("acceptedUsage", {})
        else:
            accepted = match.get("acceptedUsage", usage)
        reliable = (
            diagnostics.get("matchType") == "EXACT"
            and "multiple equal matches" not in diagnostics.get("note", "").lower()
            and usage.get("rank") == "SPECIES"
            and accepted.get("rank") == "SPECIES"
            and accepted.get("key") is not None
            and str(accepted["key"]).strip()
        )
        if not reliable:
            raise ValueError(f"No reliable GBIF species match for {scientific_name}")
        return str(accepted["key"])

    def _wait_for_download(self, key: str) -> None:
        """Poll one SDK job until success, explicit failure, or the elapsed-time limit."""
        deadline = time.monotonic() + self.max_wait_seconds
        while time.monotonic() < deadline:
            metadata = occ.download_meta(key)
            if not isinstance(metadata, Mapping):
                raise TypeError(f"Invalid metadata for GBIF download {key}")
            status = metadata.get("status")
            if status == "SUCCEEDED":
                return
            elif status in {"FAILED", "CANCELLED", "KILLED"}:
                raise RuntimeError(f"GBIF download {key} ended with status {status}")
            elif status in {"PREPARING", "RUNNING"}:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(self.poll_interval, remaining))
            else:
                raise ValueError(f"Invalid status for GBIF download {key}: {status}")
        raise TimeoutError(
            f"GBIF download {key} exceeded {self.max_wait_seconds} seconds"
        )

    def _get_all_records_for_params(self, scientificName: str) -> list[dict[str, Any]]:
        """Download one species across its years and project media into the fixed schema."""
        # Validate only when retrieving: readable CSV caches remain usable offline.
        for variable in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"):
            value = os.environ.get(variable)
            if value is None or not value.strip():
                raise ValueError(
                    f"{variable} must be set and nonblank for GBIF downloads"
                )
        taxon_key = self._resolve_taxon_key(scientificName)
        predicate = {
            "type": "and",
            "predicates": [
                {"type": "equals", "key": "TAXON_KEY", "value": taxon_key},
                {
                    "type": "in",
                    "key": "YEAR",
                    "values": [str(year) for year in self.years],
                },
                {"type": "equals", "key": "MEDIA_TYPE", "value": self.media_type},
            ],
        }
        # The SDK also returns an account-bearing payload; deliberately discard it.
        key, _ = occ.download(predicate, format="DWCA")
        self.logger.info("Submitted GBIF download %s", key)
        self._wait_for_download(key)
        # Archives are temporary inputs, not a second cache or resumable job store.
        with TemporaryDirectory(prefix="gbif-", dir=self.output_path) as directory:
            downloaded = occ.download_get(key, path=directory)
            joined = GbifArchiveReader(downloaded["path"]).read()

        fields = {
            "source_group_id": "_core_id",
            "scientific_name": "scientificName",
            "accepted_scientific_name": "acceptedScientificName",
            "taxon_key": "taxonKey",
            "species": "species",
            "species_key": "speciesKey",
            "genus": "genus",
            "family": "family",
            "order": "order",
            "class": "class",
            "phylum": "phylum",
            "kingdom": "kingdom",
            "year": "year",
            "event_date": "eventDate",
            "basis_of_record": "basisOfRecord",
            "country_code": "countryCode",
            "decimal_latitude": "decimalLatitude",
            "decimal_longitude": "decimalLongitude",
            "occurrence_source_url": "references",
            "image_url": "media_identifier",
            "image_license": "media_license",
            "image_creator": "media_creator",
            "image_references": "media_references",
            "image_type": "media_type",
            "image_format": "media_format",
            "image_created": "media_created",
            "dataset_key": "datasetKey",
            "publishing_org_key": "publishingOrgKey",
        }
        records = joined.reindex(columns=list(fields.values())).rename(
            columns={source: target for target, source in fields.items()}
        )
        records = (
            records.loc[
                (records["image_type"] == self.media_type)
                & records["image_url"].notna()
            ]
            .reindex(columns=self._record_columns())
            .astype(object)
        )
        for dataset_key in records["dataset_key"].dropna().unique():
            dataset = registry.datasets(uuid=dataset_key)
            citation = dataset.get("citation")
            if isinstance(citation, dict):
                citation = citation.get("text")
            selected = records["dataset_key"] == dataset_key
            records.loc[selected, "dataset_citation"] = citation
            records.loc[selected, "dataset_doi"] = dataset.get("doi")
        for column in (
            "source_group_id",
            "taxon_key",
            "species_key",
            "year",
            "decimal_latitude",
            "decimal_longitude",
        ):
            records[column] = pd.to_numeric(records[column], errors="raise")
        # Preserve the previous helper's None values for absent optional metadata.
        records = records.astype(object).where(records.notna(), None)
        return records.to_dict(orient="records")

    def _get_species_records(self, species: str) -> pd.DataFrame:
        """Fetch allowed image records for one species.

        Records from all configured years are retained only when at least one
        configured license manager accepts their image license. The eligible
        records are then deterministically shuffled and capped.

        Args:
            species: Scientific name to retrieve.

        Returns:
            Image-level records with the stable GBIF schema.
        """
        self.logger.info("Fetching %s observations for years %s", species, self.years)
        records = self._get_all_records_for_params(scientificName=species)

        # Each photo has its own license: an occurrence license cannot authorize it.
        records = [
            record
            for record in records
            if any(
                manager.normalize_allowed_license(record.get("image_license"))
                is not None
                for manager in self._license_managers
            )
        ]
        records_df = pd.DataFrame.from_records(records, columns=self._record_columns())
        if records_df.empty:
            return records_df
        # Random sampling prevents the API's ordering from biasing a capped dataset.
        records_df = records_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        # cap data if bigger than limit
        if len(records_df) > self.max_img_num:
            records_df = records_df.head(self.max_img_num)

        return records_df

    def _download_species_images(
        self, records_df: pd.DataFrame, images_path: Path
    ) -> pd.DataFrame:
        """Download every image row and append its outcome to a report.

        The returned frame retains the records frame's columns, order, and
        index. Because GBIF records are already image-level, repeated
        occurrence identifiers remain separate rows in the report.

        Args:
            records_df: Image-level GBIF records to download.
            images_path: Directory in which downloaded images are stored.

        Returns:
            A copy of ``records_df`` with a boolean ``success`` column.
        """
        images_path.mkdir(parents=True, exist_ok=True)
        successes: list[bool] = []
        session = requests.Session()
        try:
            for _, record in tqdm(records_df.iterrows(), total=len(records_df)):
                successes.append(self._download_image(record, images_path, session))
        finally:
            session.close()

        report = records_df.copy()
        report["success"] = pd.Series(
            successes,
            index=report.index,
            dtype=bool,
        )
        return report

    @staticmethod
    def _image_suffix(content_start: bytes) -> str | None:
        """Return a supported file suffix based on an image signature."""
        if content_start.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        elif content_start.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        else:
            return None

    def _existing_image_is_valid(
        self,
        image_path: Path,
        expected_suffix: str,
    ) -> bool:
        """Check whether an existing path contains its expected image type."""
        if not image_path.is_file():
            return False
        else:
            try:
                with image_path.open("rb") as image_file:
                    content_start = image_file.read(8)
            except OSError:
                return False
            else:
                detected_suffix = self._image_suffix(content_start)
                if detected_suffix == expected_suffix:
                    return True
                else:
                    return False

    @staticmethod
    def _normalize_occurrence_id(raw_occurrence_id: Any) -> int | None:
        """Normalize a scalar GBIF occurrence identifier to an integer."""
        if pd.isna(raw_occurrence_id) or isinstance(raw_occurrence_id, bool):
            return None
        else:
            try:
                occurrence_id = int(raw_occurrence_id)
            except (TypeError, ValueError, OverflowError):
                return None
            else:
                is_fractional_float = (
                    isinstance(raw_occurrence_id, float)
                    and not raw_occurrence_id.is_integer()
                )
                if is_fractional_float:
                    return None
                else:
                    return occurrence_id

    @staticmethod
    def _response_size_is_allowed(
        response: requests.Response,
        maximum_size: int,
    ) -> bool:
        """Validate an optional HTTP content length against the byte limit."""
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            return True
        else:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                return False
            else:
                if declared_size < 0:
                    return False
                elif declared_size > maximum_size:
                    return False
                else:
                    return True

    def _stream_image_response(
        self,
        response: requests.Response,
        partial_path: Path,
        maximum_size: int,
    ) -> str | None:
        """Stream a bounded response and return its signature-derived suffix."""
        downloaded_size = 0
        with partial_path.open("wb") as partial_file:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                chunk_size = len(chunk)
                next_size = downloaded_size + chunk_size
                if chunk_size == 0:
                    self.logger.debug(
                        "Ignoring an empty GBIF image response chunk for %s",
                        partial_path,
                    )
                elif next_size > maximum_size:
                    return None
                else:
                    partial_file.write(chunk)
                    downloaded_size = next_size

        with partial_path.open("rb") as partial_file:
            content_start = partial_file.read(8)
        suffix = self._image_suffix(content_start)
        if suffix is None:
            return None
        else:
            return suffix

    def _download_cached_image(
        self,
        cache_url: str,
        image_url: str,
        filename_stem: str,
        images_path: Path,
        session: requests.Session,
    ) -> bool:
        """Request, validate, and atomically store one GBIF cache image."""
        partial_path = images_path / f"{filename_stem}.part"
        response: requests.Response | None = None
        maximum_size = 20 * 1024 * 1024

        try:
            response = session.get(
                cache_url,
                stream=True,
                timeout=30,
                allow_redirects=False,
            )
            if response.status_code != requests.codes.ok:
                return False
            elif not self._response_size_is_allowed(response, maximum_size):
                return False
            else:
                suffix = self._stream_image_response(
                    response,
                    partial_path,
                    maximum_size,
                )
                if suffix is None:
                    return False
                else:
                    destination = images_path / f"{filename_stem}{suffix}"
                    partial_path.replace(destination)
                    return True
        except requests.RequestException as exc:
            self.logger.warning("GBIF image request failed for %s: %s", image_url, exc)
            return False
        except OSError as exc:
            self.logger.warning("Could not save GBIF image %s: %s", image_url, exc)
            return False
        finally:
            if response is not None:
                response.close()
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError as exc:
                    self.logger.warning(
                        "Could not remove partial GBIF image %s: %s",
                        partial_path,
                        exc,
                    )

    def _download_image(
        self,
        record: pd.Series,
        images_path: Path,
        session: requests.Session,
    ) -> bool:
        """Download one image record through the GBIF occurrence cache.

        The cache URL uses the occurrence key and the MD5 digest of the exact
        publisher URL, as required by the GBIF image-cache API. Redirects are
        disabled because a cache redirect could otherwise send the downloader
        to an untrusted publisher-controlled destination.

        Args:
            record: One image-level row from the GBIF records frame.
            images_path: Directory in which the image is stored.
            session: Session shared by downloads for one species.

        Returns:
            True when a valid image already exists or is downloaded; otherwise
            False.
        """
        occurrence_id = self._normalize_occurrence_id(record.get("source_group_id"))
        image_url = record.get("image_url")
        if occurrence_id is None:
            return False
        elif not isinstance(image_url, str) or not image_url.strip():
            return False
        else:
            image_digest = hashlib.md5(
                image_url.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
            filename_stem = f"{occurrence_id}_{image_digest}"
            images_path.mkdir(parents=True, exist_ok=True)
            jpeg_path = images_path / f"{filename_stem}.jpg"
            png_path = images_path / f"{filename_stem}.png"
            jpeg_is_valid = self._existing_image_is_valid(jpeg_path, ".jpg")
            png_is_valid = self._existing_image_is_valid(png_path, ".png")

            if jpeg_is_valid:
                return True
            elif png_is_valid:
                return True
            else:
                cache_url = (
                    "https://api.gbif.org/v1/image/cache/occurrence/"
                    f"{occurrence_id}/media/{image_digest}"
                )
                return self._download_cached_image(
                    cache_url,
                    image_url,
                    filename_stem,
                    images_path,
                    session,
                )

    def download(self) -> None:
        """Retrieve records for all species before downloading any images."""
        self.retrieve_records()
        self.download_images()

    @staticmethod
    def _write_records_atomic(records: pd.DataFrame, destination: Path) -> None:
        """Promote a complete CSV on the same filesystem; never cache partial bytes."""
        partial_path = destination.with_suffix(".csv.part")
        try:
            records.to_csv(partial_path, index=False)
            partial_path.replace(destination)
        finally:
            partial_path.unlink(missing_ok=True)

    def retrieve_records(self) -> None:
        """Retrieve and save records for all species without downloading images.

        Existing readable ``records.csv`` files are reused, including empty
        tables with column headers. Delete a cache to retrieve fresh records.
        """
        for sp in self.species:
            self.logger.info(f"species: {sp}")
            species_path = self.output_path / sp
            species_path.mkdir(parents=True, exist_ok=True)

            try:
                records_df = pd.read_csv(species_path / "records.csv")
            except FileNotFoundError:
                self.logger.info("Retrieving species records")
                records_df = self._get_species_records(sp)
                self._write_records_atomic(records_df, species_path / "records.csv")
            else:
                self.logger.info(
                    "Found existing records.csv for %s, skipping retrieval", sp
                )

    def download_images(self) -> None:
        """Download images from saved records and refresh per-image reports.

        Valid existing images are reused. No species records are retrieved.

        Raises:
            FileNotFoundError: If a species has no saved records. Run
                ``retrieve_records()`` first or provide its ``records.csv``.
        """
        for sp in self.species:
            species_path = self.output_path / sp
            records_path = species_path / "records.csv"
            try:
                records_df = pd.read_csv(records_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Missing records for {sp}: {records_path}. "
                    "Run retrieve_records() before download_images()."
                ) from exc

            images_path = species_path / "imgs"
            images_path.mkdir(exist_ok=True)
            self.logger.info("Downloading images for %s", sp)
            report = self._download_species_images(records_df, images_path)
            report.to_csv(
                species_path / "download_report.csv",
                index=True,
                index_label="index",
            )
            downloaded = int(report["success"].sum())
            self.logger.info("Downloaded %s images for %s", downloaded, sp)
