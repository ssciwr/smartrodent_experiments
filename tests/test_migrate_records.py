"""Tests for timestamped migration backups."""

from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from scripts.migrate_records import main


@pytest.mark.parametrize("backup_name", [None, "custom_backup"])
def test_backups_share_run_timestamp_and_preserve_original_records(
    tmp_path, monkeypatch, backup_name
):
    image_root = tmp_path / "records"
    old_root = tmp_path / "old_images"
    new_root = tmp_path / "new_images"
    new_root.mkdir()
    originals = {}
    for species in ("mouse", "rat"):
        species_dir = image_root / species
        species_dir.mkdir(parents=True)
        image_name = f"{species}.jpg"
        (new_root / image_name).write_bytes(b"image")
        original = pd.DataFrame(
            {"image_path": [str(old_root / image_name)], "photo_id": [1]}
        )
        original.to_csv(species_dir / "records.csv", index=False)
        originals[species] = original

    config = {
        "image_root": str(image_root),
        "old_root": str(old_root),
        "new_root": str(new_root),
    }
    if backup_name is not None:
        config["backup_name"] = backup_name
    config_path = tmp_path / "migration.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # A changing clock ensures the timestamp is chosen once, not per species.
    clock = Mock()
    clock.now.side_effect = [
        datetime(2026, 7, 8, 14, 30, 25),
        datetime(2026, 7, 8, 14, 30, 26),
    ]
    monkeypatch.setattr("scripts.migrate_records.datetime", clock)
    main(config_path)
    assert (image_root / config_path.name).read_bytes() == config_path.read_bytes()

    expected_name = f"{backup_name or 'records_original'}_143025_08072026.csv"
    for species, original in originals.items():
        species_dir = image_root / species
        assert {path.name for path in species_dir.iterdir()} == {
            "records.csv",
            expected_name,
        }
        pd.testing.assert_frame_equal(
            pd.read_csv(species_dir / expected_name), original
        )
        migrated = pd.read_csv(species_dir / "records.csv")
        assert migrated["image_path"].tolist() == [str(new_root / f"{species}.jpg")]
