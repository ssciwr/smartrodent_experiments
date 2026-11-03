"""Download per-class camera-trap images from LILA datasets."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from smartrodent.base import DatasetLoader


class LilaDataset(DatasetLoader):
    """Select and download LILA camera-trap images for configured classes.

    Output follows the iNaturalist layout: ``<output>/<class>/records.csv`` and
    ``<output>/<class>/imgs/``. The LILA index, taxonomy mapping, and metadata
    zips are cached in ``<output>/metadata``.

    Attributes:
        output_path: Root directory for records, images, and the metadata cache.
        cache: Directory holding downloaded LILA index, taxonomy, and metadata.
        datasets: LILA dataset names mapped to their license.
        classes: Output class names, Latin taxa or raw LILA labels.
        allowed_licenses: Licenses a dataset must have to be used.
        index_url: URL of the LILA camera-trap dataset index CSV.
        taxonomy_url: URL of the LILA taxonomy-mapping CSV.
        max_img_num: Maximum images per class and dataset.
        seed: Seed for sampling sequences when capping.
    """

    # Most to least specific: a label goes to the first rank that is a configured class.
    RANKS = ("species", "subgenus", "genus", "tribe", "subfamily", "family",
             "superfamily", "suborder", "order", "class")

    def __init__(
        self,
        output_path: str | Path,
        datasets: Mapping[str, str],
        classes: Sequence[str],
        allowed_licenses: Sequence[str],
        index_url: str,
        taxonomy_url: str,
        max_img_num: int = 2500,
        seed: int = 42,
    ) -> None:
        """Store the download settings and create the metadata cache directory.

        Args:
            output_path: Root directory for records, images, and the cache.
            datasets: LILA dataset names, as in the LILA index, mapped to the
                license each dataset is published under.
            classes: Output classes: Latin taxon names of any rank, or raw LILA
                labels such as ``empty``. Matching ignores case.
            allowed_licenses: Dataset licenses that may be used; datasets with
                other licenses are skipped.
            index_url: URL of the LILA camera-trap dataset index CSV.
            taxonomy_url: URL of the LILA taxonomy-mapping CSV.
            max_img_num: Maximum images per class and dataset.
            seed: Seed for sampling sequences when capping.
        """
        self.output_path = Path(output_path).resolve()
        self.cache = self.output_path / "metadata"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.datasets = dict(datasets)
        self.classes = list(classes)
        self.allowed_licenses = set(allowed_licenses)
        self.index_url = index_url
        self.taxonomy_url = taxonomy_url
        self.max_img_num = max_img_num
        self.seed = seed
        self.logger = logging.getLogger(type(self).__name__)

    @classmethod
    def from_config(cls, config_path: str | Path) -> LilaDataset:
        """Build a downloader from a YAML config and copy it to the output root.

        Args:
            config_path: YAML file whose ``data.lila`` section holds the
                constructor arguments by name.

        Returns:
            A configured LILA downloader.

        Raises:
            KeyError: If the config has no ``data.lila`` section.
            TypeError: If ``data.lila`` lacks a required argument or has an
                unknown one.
        """
        config = yaml.safe_load(Path(config_path).read_text())["data"]["lila"]
        dataset = cls(**config)
        shutil.copy2(config_path, dataset.output_path / Path(config_path).name)
        return dataset

    def download(self) -> None:
        """Select records for all classes, then download their images."""
        self.retrieve_records()
        self.download_images()

    def retrieve_records(self) -> None:
        """Write ``records.csv`` for every class that does not have one yet.

        Existing ``records.csv`` files are kept; delete one to select that class
        again. Classes without matching images get an empty ``records.csv``.

        Raises:
            KeyError: If a configured dataset is not in the LILA index.
            ValueError: If a metadata zip does not hold exactly one JSON file.
            requests.HTTPError: If an index, taxonomy, or metadata request fails.
        """
        missing = [c for c in self.classes if not (self.output_path / c / "records.csv").is_file()]
        if not missing:
            return
        index = pd.read_csv(self._fetch(self.index_url), dtype=str, keep_default_na=False)
        index = index.set_index("name")
        taxonomy = pd.read_csv(self._fetch(self.taxonomy_url), dtype=str, keep_default_na=False)
        selected: dict[str, list[pd.DataFrame]] = {c: [] for c in missing}
        for name, license in self.datasets.items():
            if license not in self.allowed_licenses:
                self.logger.info("Skipping %s: license %s is not allowed", name, license)
                continue
            rows = taxonomy[taxonomy["dataset_name"] == name].drop_duplicates("query")
            labels = rows.set_index("query").to_dict("index")
            records = self._dataset_records(name, license, index.loc[name], labels)
            for class_name, frame in records.items():
                if class_name in selected:
                    selected[class_name].append(frame)
        for class_name, frames in selected.items():
            if frames:
                records = pd.concat(frames, ignore_index=True)
            else:
                records = pd.DataFrame(columns=["id", "photo.id", "image_url", "image_file"])
            (self.output_path / class_name).mkdir(exist_ok=True)
            records.to_csv(self.output_path / class_name / "records.csv", index=False)

    def _dataset_records(
        self,
        name: str,
        license: str,
        info: pd.Series,
        labels: dict[str, dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Return capped image rows per class for one dataset.

        An image with several classes gets one row in each class, holding only
        that class's labels and boxes.

        Args:
            name: LILA dataset name.
            license: License of the dataset, stored on every row.
            info: The dataset's row from the LILA index.
            labels: Taxonomy rows of this dataset, keyed by raw LILA label.

        Returns:
            Class names mapped to their capped image rows.
        """
        data = self._load_json(info["metadata_url_gcp"])
        category = {c["id"]: c["name"] for c in data["categories"]}
        annotations = pd.DataFrame(data["annotations"]).reindex(
            columns=["image_id", "category_id", "bbox"])
        annotations["category"] = annotations["category_id"].map(category)
        annotations["class_name"] = annotations["category"].map(
            lambda label: self._match(label, labels.get(label, {})))
        annotations = annotations.dropna(subset=["class_name"])
        if annotations.empty:
            return {}

        images = pd.DataFrame(data["images"]).rename(columns={"id": "image_id"})
        # Frames of a sequence are near-duplicates and must stay in one split;
        # datasets without sequences fall back to the camera location.
        group = images["seq_id"] if "seq_id" in images else images["location"]
        images["id"] = name + "/" + group.astype(str)
        images["photo.id"] = name + "/" + images["image_id"].astype(str)
        images["image_url"] = (info["image_base_url_gcp"].rstrip("/") + "/"
                               + images["file_name"].map(urllib.parse.quote))
        images["image_file"] = info["short_name"] + "_" + images["file_name"].str.replace("/", "__")

        rows = annotations.groupby(["image_id", "class_name"]).agg(
            category=("category", lambda s: "|".join(sorted(set(s)))),
            bbox=("bbox", lambda s: json.dumps([b for b in s if isinstance(b, list)])),
        ).reset_index().merge(images, on="image_id")
        rows["dataset"] = name
        rows["license"] = license
        return {c: self._cap(frame, name, c) for c, frame in rows.groupby("class_name")}

    def _match(self, label: str, taxon: Mapping[str, str]) -> str | None:
        """Return the most specific configured class for a LILA label, or None.

        The raw label counts as most specific, so classes such as ``empty`` match
        directly; then the taxonomy ranks are tried from finest to coarsest.

        Args:
            label: Raw LILA category name.
            taxon: The label's taxonomy row, empty if the label has none.

        Returns:
            The configured class name as written in the config, or None.
        """
        lookup = {c.lower(): c for c in self.classes}
        for candidate in [label, *(taxon.get(rank, "") for rank in self.RANKS)]:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        return None

    def _cap(self, frame: pd.DataFrame, dataset_name: str, class_name: str) -> pd.DataFrame:
        """Take whole sequences in seeded random order until ``max_img_num`` rows.

        Only the last sequence taken may be cut. Seeding with dataset and class
        keeps every selection independent of the others.

        Args:
            frame: Image rows of one class in one dataset.
            dataset_name: LILA dataset name, part of the seed.
            class_name: Class name, part of the seed.

        Returns:
            At most ``max_img_num`` rows, without the ``class_name`` column.
        """
        groups = sorted(frame["id"].unique())
        random.Random(f"{self.seed}|{dataset_name}|{class_name}").shuffle(groups)
        order = frame["id"].map({group: i for i, group in enumerate(groups)})
        capped = frame.assign(_order=order).sort_values(["_order", "photo.id"])
        return capped.head(self.max_img_num).drop(columns=["_order", "class_name"])

    def _fetch(self, url: str) -> Path:
        """Download ``url`` into the metadata cache once and return its path.

        Args:
            url: File to download; its basename names the cached file.

        Returns:
            Path of the cached file.

        Raises:
            requests.HTTPError: If the request fails.
        """
        path = self.cache / Path(urllib.parse.urlparse(url).path).name
        if not path.is_file():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        return path

    def _load_json(self, url: str) -> dict[str, Any]:
        """Load the single JSON file inside a cached metadata zip.

        Args:
            url: URL of the dataset's metadata zip.

        Returns:
            The parsed COCO Camera Traps metadata.

        Raises:
            ValueError: If the zip does not hold exactly one JSON file.
        """
        with zipfile.ZipFile(self._fetch(url)) as archive:
            (member,) = [n for n in archive.namelist() if n.endswith(".json")]
            return json.loads(archive.read(member))

    def download_images(self) -> None:
        """Download every record's image and write ``image_path`` into the records.

        Existing images are reused. Rows whose image returns 404 are dropped:
        LILA withholds some images (e.g. humans) from its public buckets.

        Raises:
            FileNotFoundError: If a class has no ``records.csv``; run
                ``retrieve_records()`` first.
            requests.HTTPError: If an image request fails with another status.
        """
        for class_name in self.classes:
            records_path = self.output_path / class_name / "records.csv"
            records = pd.read_csv(records_path)
            images_path = records_path.parent / "imgs"
            images_path.mkdir(exist_ok=True)
            found = [
                self._download_image(url, images_path / file)
                for url, file in tqdm(zip(records["image_url"], records["image_file"]),
                                      total=len(records), desc=class_name)
            ]
            records = records.loc[found].copy()
            records["image_path"] = [
                os.path.relpath(images_path / file, self.output_path.parent)
                for file in records["image_file"]
            ]
            records.to_csv(records_path, index=False)

    def _download_image(self, url: str, path: Path) -> bool:
        """Download one image unless it already exists.

        Args:
            url: Image URL in the LILA bucket.
            path: Where to save the image.

        Returns:
            True if the image exists or was downloaded, False if LILA withholds
            it (404).

        Raises:
            requests.HTTPError: If the request fails with another status.
        """
        if path.is_file():
            return True
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            return False
        else:
            response.raise_for_status()
            path.write_bytes(response.content)
            return True
