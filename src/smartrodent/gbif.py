"""Download licensed gbif observations into a species-organized dataset."""

from __future__ import annotations
from collections.abc import Mapping, Sequence

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

        Raises:
            TypeError: If ``allowed_licenses`` is not a mapping or contains
                invalid family or license values.
            ValueError: If the license mapping is empty, a family or license is
                unsupported, the year selection is invalid, ``max_img_num`` is
                negative, or no species are configured.
        """
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

    def _get_all_records_for_params(self, **params) -> list[dict[str, Any]]:
        """Fetch occurrences and project their media into a fixed image schema.

        Each returned dictionary represents one image. Occurrence metadata is
        repeated for all images belonging to the same occurrence, whose GBIF
        key is exposed as ``source_group_id`` for group-aware dataset splits.
        Missing GBIF fields are represented by ``None``.

        Returns:
            Image-level records with stable occurrence, image, taxonomy, and
            source-dataset fields.
        """
        image_records: list[dict[str, Any]] = []
        dataset_metadata: dict[str, tuple[Any, Any]] = {}
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
                if dataset_key and dataset_key not in dataset_metadata:
                    dataset = registry.datasets(uuid=dataset_key)
                    citation_value = dataset.get("citation")
                    citation = (
                        citation_value.get("text")
                        if isinstance(citation_value, dict)
                        else citation_value
                    )
                    dataset_metadata[dataset_key] = (citation, dataset.get("doi"))

                citation, doi = dataset_metadata.get(dataset_key, (None, None))
                for media in occurrence.get("media", []):
                    if media.get("type") != self.media_type:
                        continue

                    image_url = media.get("identifier")
                    if not image_url:
                        continue

                    image_records.append(
                        {
                            "source_group_id": occurrence.get("key"),
                            "scientific_name": occurrence.get("scientificName"),
                            "accepted_scientific_name": occurrence.get(
                                "acceptedScientificName"
                            ),
                            "taxon_key": occurrence.get("taxonKey"),
                            "species": occurrence.get("species"),
                            "species_key": occurrence.get("speciesKey"),
                            "genus": occurrence.get("genus"),
                            "family": occurrence.get("family"),
                            "order": occurrence.get("order"),
                            "class": occurrence.get("class"),
                            "phylum": occurrence.get("phylum"),
                            "kingdom": occurrence.get("kingdom"),
                            "year": occurrence.get("year"),
                            "event_date": occurrence.get("eventDate"),
                            "basis_of_record": occurrence.get("basisOfRecord"),
                            "country_code": occurrence.get("countryCode"),
                            "decimal_latitude": occurrence.get("decimalLatitude"),
                            "decimal_longitude": occurrence.get("decimalLongitude"),
                            "occurrence_source_url": occurrence.get("references"),
                            "image_url": image_url,
                            "image_license": media.get("license"),
                            "image_creator": media.get("creator"),
                            "image_references": media.get("references"),
                            "image_type": media.get("type"),
                            "image_format": media.get("format"),
                            "image_created": media.get("created"),
                            "dataset_key": dataset_key,
                            "publishing_org_key": occurrence.get("publishingOrgKey"),
                            "dataset_citation": citation,
                            "dataset_doi": doi,
                        }
                    )

            if response["endOfRecords"] or not page:
                break

            offset += len(page)

        return image_records

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
        records: list[dict[str, Any]] = []
        for year in self.years:
            self.logger.info("Fetching %s observations from %s", species, year)
            records_for_year = self._get_all_records_for_params(
                scientificName=species,
                mediatype=self.media_type,
                year=year,
            )
            records.extend(records_for_year)

        # README: this relies on the license not being contradictory internally, and one and only one being relevant at all times
        # This assumption is reasonable b/c we have only a single license per image (i.e., per record) or none at all
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
    ) -> None:
        pass

    def download(self) -> None:

        for sp in self.species:
            self.logger.info(f"species: {sp}")
            species_path = self.output_path / sp
            images_path = species_path / "imgs"
            species_path.mkdir(parents=True, exist_ok=True)
            images_path.mkdir(exist_ok=True)

            self.logger.info("Retrieving species records")
            records_df = self._get_species_records(sp)
            records_df.to_csv(species_path / "records.csv", index=False)

            self.logger.info("Downloading images")
            downloaded = self._download_species_images(records_df, images_path)
            self.logger.info("Downloaded %s images for %s", downloaded, sp)
