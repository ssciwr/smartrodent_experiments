#!/usr/bin/env python
"""Download and inspect external camera-trap datasets.

This script was converted from an exploratory notebook. The reusable data
loading and filtering steps live in functions, while dataset-specific choices
are kept together in the configuration structures in main().
"""

import json
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

RODENT_KEYWORDS = (
    "rat",
    "mouse",
    "vole",
    "shrew",
    "murid",
    "rodent",
    "sorex",
    "crocidura",
    "apodemus",
    "myodes",
    "arvicol",
    "mus",
    "rattus",
)
EXTENDED_KEYWORDS = RODENT_KEYWORDS + (
    "domestic",
    "human",
    "empty",
    "horse",
    "cow",
    "cat",
    "dog",
)


def load_inaturalist_helper() -> ModuleType:
    """Load only the local iNaturalist helper, without package-wide imports."""
    module_path = PROJECT_ROOT / "src" / "smartrodent" / "inaturalist.py"
    spec = spec_from_file_location("smartrodent_inaturalist", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load iNaturalist helper from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_and_extract_metadata(
    session: requests.Session,
    url: str,
    name: str,
    output_dir: Path = DATASETS_ROOT,
) -> Path:
    """Download a metadata ZIP and extract it into a named directory."""
    destination = output_dir / name
    archive_path = destination.with_suffix(".zip")
    destination.mkdir(parents=True, exist_ok=True)

    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with archive_path.open("wb") as archive:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    archive.write(chunk)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)

    return destination


def load_lila_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load LILA's COCO-style JSON and attach category names to annotations."""
    with metadata_path.open(encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)

    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])
    categories = pd.DataFrame(data["categories"]).rename(
        columns={"id": "category_id", "name": "category"}
    )

    return annotations.merge(
        images,
        left_on="image_id",
        right_on="id",
        suffixes=("_ann", "_img"),
    ).merge(categories, on="category_id")


def matching_categories(
    metadata: pd.DataFrame,
    keywords: tuple[str, ...],
    category_column: str = "category",
) -> list[str]:
    """Return sorted category labels containing at least one keyword."""
    categories = metadata[category_column].dropna().astype(str).unique()
    return sorted(
        category
        for category in categories
        if any(keyword.casefold() in category.casefold() for keyword in keywords)
    )


def filter_category(
    metadata: pd.DataFrame,
    category: str,
    category_column: str = "category",
) -> pd.DataFrame:
    """Select one category, ignoring surrounding whitespace and case."""
    normalized = metadata[category_column].astype("string").str.strip().str.casefold()
    return metadata.loc[normalized.eq(category.strip().casefold())]


def print_category_counts(
    metadata: pd.DataFrame,
    categories: list[str] | tuple[str, ...],
    category_column: str = "category",
) -> None:
    """Print the number of rows associated with each requested category."""
    for category in categories:
        count = len(filter_category(metadata, category, category_column))
        print(f"{category}: {count}")


def download_images(
    session: requests.Session,
    base_url: str,
    output_dir: Path,
    metadata: pd.DataFrame,
    category: str,
    *,
    mapped_category: str | None = None,
    category_column: str = "category",
    filename_column: str = "file_name",
    limit: int | None = None,
    random_state: int = 42,
) -> None:
    """Download unique images for one category into its mapped species folder."""
    selected = filter_category(metadata, category, category_column).drop_duplicates(
        subset=filename_column
    )

    # Sampling before truncation avoids taking a dataset-order-biased subset.
    if limit is not None:
        selected = selected.sample(frac=1, random_state=random_state).head(limit)

    category_dir = output_dir / (mapped_category or category)
    category_dir.mkdir(parents=True, exist_ok=True)

    rows = tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"Downloading {category}",
    )

    if Path(category_dir).exists():
        existing_images = set(
            [f.name for f in Path(category_dir).iterdir() if f.suffix == ".jpg"]
        )
    else:
        existing_images = set()

    for row_index, row in rows:
        remote_name = row[filename_column]

        if remote_name in existing_images:
            continue

        with session.get(
            f"{base_url.rstrip('/')}/{remote_name}",
            timeout=(30, 120),
        ) as response:
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print("HTTPerror thrown - skipping")
            except Exception as e:
                raise e
            (category_dir / f"{row_index}.jpg").write_bytes(response.content)


def download_category_map(
    session: requests.Session,
    metadata: pd.DataFrame,
    base_url: str,
    output_dir: Path,
    category_map: dict[str, str],
    *,
    limit: int | None = None,
) -> None:
    """Download several source categories using consistent species names."""
    for category, species_name in category_map.items():
        print(f"{category} -> {species_name}")
        download_images(
            session,
            base_url,
            output_dir,
            metadata,
            category,
            mapped_category=species_name,
            limit=limit,
        )


def inspect_lila_dataset(
    session: requests.Session,
    config: dict[str, Any],
) -> None:
    """Download, inspect, and optionally collect images from one LILA dataset."""
    download_and_extract_metadata(
        session,
        config["archive_url"],
        config["name"],
    )
    metadata = load_lila_metadata(DATASETS_ROOT / config["metadata_file"])
    categories = matching_categories(
        metadata,
        config.get("keywords", RODENT_KEYWORDS),
    )

    print(f"\n{config['name']} matching categories:")
    print_category_counts(metadata, categories)

    category_map = config.get("category_map")
    if category_map:
        download_category_map(
            session,
            metadata,
            config["base_url"],
            DATASETS_ROOT / config["output_dir"],
            category_map,
            limit=config.get("limit"),
        )


def inspect_nacti_dataset(session: requests.Session) -> None:
    """Inspect the North American Camera Trap CSV, whose schema differs."""
    download_and_extract_metadata(
        session,
        "https://storage.googleapis.com/public-datasets-lila/nacti/"
        "nacti_metadata.csv.zip",
        "north_america_camera_trap_data",
    )
    metadata_path = (
        DATASETS_ROOT / "north_america_camera_trap_data" / "nacti_metadata.csv"
    )
    metadata = pd.read_csv(metadata_path).dropna()

    family_matches = matching_categories(
        metadata,
        EXTENDED_KEYWORDS + ("canidae", "homo"),
        category_column="family",
    )
    print("\nnorth_america_camera_trap_data matching families:")
    print(family_matches)
    print_category_counts(
        metadata,
        ["canis familiaris", "muridae"],
        category_column="name",
    )


def main() -> None:
    """Run the exploratory downloads in their original notebook order."""
    with requests.Session() as session:
        # AMMonitor is handled separately because it uses a small, explicit
        # category set rather than the keyword-based inspection below.
        # download_and_extract_metadata(
        #     session,
        #     "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
        #     "ammonitor-camera-traps/ammonitor-camera-traps.zip",
        #     "ammonitor",
        # )
        # ammonitor = load_lila_metadata(
        #     DATASETS_ROOT / "ammonitor" / "ammonitor-camera-traps.json"
        # )
        # ammonitor_categories = {
        #     "empty": "empty",
        #     "domestic cat": "Felis catus",
        #     "domestic dog": "Canis familiaris",
        #     "mouse sp.": "Mouse sp.",
        # }
        # print("\nammonitor selected categories:")
        # print_category_counts(ammonitor, tuple(ammonitor_categories))
        # download_category_map(
        #     session,
        #     ammonitor,
        #     "https://storage.googleapis.com/public-datasets-lila/"
        #     "ammonitor-camera-traps",
        #     DATASETS_ROOT / "ammonitor_camera_traps",
        #     ammonitor_categories,
        #     limit=2000,
        # )

        # These datasets all use LILA's COCO-style JSON. Configuration keeps
        # their URLs, filters, and optional download mappings in one place.
        lila_datasets: list[dict[str, Any]] = [
            # {
            #     "name": "felidae_conservation",
            #     "archive_url": (
            #         "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            #         "felidae-conservation-fund/"
            #         "felidae_conservation_fund_2020_2025.zip"
            #     ),
            #     "metadata_file": (
            #         "felidae_conservation/felidae_conservation_fund_2020_2025.json"
            #     ),
            # },
            # {
            #     "name": "wsu_lynx",
            #     "archive_url": (
            #         "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            #         "wsu-lynx/wsu-lynx.26.02.13.1705.zip"
            #     ),
            #     "metadata_file": "wsu_lynx/wsu-lynx.json",
            # },
            # {
            #     "name": "california_small_animals",
            #     "archive_url": (
            #         "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            #         "california-small-animals/"
            #         "california_small_animals_with_sequences.zip"
            #     ),
            #     "metadata_file": (
            #         "california_small_animals/"
            #         "california_small_animals_with_sequences.json"
            #     ),
            #     "base_url": (
            #         "https://storage.googleapis.com/public-datasets-lila/"
            #         "california-small-animals"
            #     ),
            #     "output_dir": "california_small_animals",
            #     "category_map": {
            #         "brown rat": "Rattus norvegicus",
            #         "house mouse": "Mus musculus",
            #         "house rat": "Rattus rattus",
            #         "shrew-mole": "Soricidae",
            #         "sorex species": "Sorex sp.",
            #     },
            # },
            # {
            #     "name": "ohio_small_animals",
            #     "archive_url": (
            #         "https://storage.googleapis.com/public-datasets-lila/"
            #         "osu-small-animals/osu-small-animals.json.zip"
            #     ),
            #     "metadata_file": ("ohio_small_animals/osu-small-animals.json"),
            #     "base_url": (
            #         "https://storage.googleapis.com/public-datasets-lila/"
            #         "osu-small-animals"
            #     ),
            #     "output_dir": "ohio_small_animals",
            #     "category_map": {
            #         "brown_rat": "Rattus norvegicus",
            #         "masked_shrew": "Sorex cinereus",
            #     },
            #     "limit": 1000,
            # },
            # {
            #     "name": "swg_camera_traps_2018-2020",
            #     "archive_url": (
            #         "https://storage.googleapis.com/public-datasets-lila/"
            #         "swg-camera-traps/swg_camera_traps.zip"
            #     ),
            #     "metadata_file": ("swg_camera_traps_2018-2020/swg_camera_traps.json"),
            #     "keywords": EXTENDED_KEYWORDS,
            # },
            # {
            #     "name": "idaho_camera_trap_data",
            #     "archive_url": (
            #         "https://storage.googleapis.com/public-datasets-lila/"
            #         "idaho-camera-traps/idaho-camera-traps.json.zip"
            #     ),
            #     "metadata_file": ("idaho_camera_trap_data/idaho-camera-traps.json"),
            #     "keywords": EXTENDED_KEYWORDS,
            # },
            {
                "name": "caltech_camera_traps",
                "archive_url": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip",
                "metadata_file": "/mnt/dataLinux/Development/rodent_experiments/datasets/caltech_camera_traps/caltech_images_20210113.json",
                "keywords": EXTENDED_KEYWORDS,
            },
        ]

        for config in lila_datasets:
            inspect_lila_dataset(session, config)

        # inspect_nacti_dataset(session)

    # # Load only this helper module; importing the entire smartrodent package
    # # also initializes unrelated model and CUDA dependencies.
    # load_inaturalist_helper().download_inat_data(
    #     PROJECT_ROOT / "configs" / "data_config_central_europe.yaml"
    # )


if __name__ == "__main__":
    main()
