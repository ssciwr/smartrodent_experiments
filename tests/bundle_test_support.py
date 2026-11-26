"""Shared dataset-bundle fixture and helpers for the path-portability tests.

Registered as a pytest plugin in ``conftest.py``; the ``bundle`` fixture is
requested by parameter name, the helpers are imported explicitly.
"""

import pandas as pd
import pytest
import yaml
from PIL import Image


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Create originals under one stage and run from an unrelated directory."""
    dataset_root = tmp_path / "dataset"
    records_by_species = {}
    for species in ("Mus musculus", "Sorex araneus"):
        rows = []
        for index in range(6):
            image = (
                dataset_root / "inaturalist" / species / "imgs" / f"{100 + index}_0.jpg"
            )
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 10), "red").save(image)
            rows.append(
                {
                    "id": 100 + index,
                    "photo.id": 1000 + index,
                    "photo_index": 0,
                    "photo.url": f"https://example.test/{index}/square.jpg",
                    "photo.license_code": "cc-by",
                    "image_path": image.relative_to(dataset_root).as_posix(),
                }
            )
        records_by_species[species] = pd.DataFrame(rows)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    return dataset_root, records_by_species


def write_stage_config(path, configuration):
    """Write a temporary stage config without introducing new path settings.

    Args:
        path: Destination YAML file.
        configuration: Mapping to write.

    Returns:
        ``path``.
    """
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


def write_stage_records(stage_root, records_by_species):
    """Write stage CSVs without rebasing their shared-root image references.

    Args:
        stage_root: Stage directory receiving one ``<species>/records.csv`` each.
        records_by_species: Records to write, keyed by species directory name.
    """
    for species, records in records_by_species.items():
        records_path = stage_root / species / "records.csv"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(records_path, index=False)


def move_bundle(dataset_root):
    """Relocate the entire tree so the original location cannot mask broken links.

    Args:
        dataset_root: Root of the bundle to move.

    Returns:
        The bundle's new root.
    """
    destination = dataset_root.parent / "transferred" / "renamed_dataset"
    destination.parent.mkdir()
    dataset_root.rename(destination)
    return destination
