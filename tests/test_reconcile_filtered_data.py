"""Tests for reconciling human-sorted images with their source records."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

from scripts.reconcile_filtered_data import main


@pytest.fixture
def reconciliation(tmp_path):
    """Create source records and materialized images for two configured stages."""
    input_root = tmp_path / "original"
    output_root = tmp_path / "corrected"
    originals = {}

    for offset, subdir in enumerate(("kept", "rejected")):
        source = input_root / subdir / "mouse"
        source.mkdir(parents=True)
        selected = source / f"{subdir}_selected.jpg"
        omitted = source / f"{subdir}_omitted.jpg"
        selected.write_bytes(b"selected image")
        omitted.write_bytes(b"omitted image")

        original = pd.DataFrame(
            {
                "id": [10 + offset, 20 + offset],
                "photo.id": [1 + offset, 2 + offset],
                "image_path": [str(selected), str(omitted)],
                f"{subdir}_animal": [False, True],
            }
        )
        original.to_csv(source / "records.csv", index=False)
        originals[subdir] = original

        destination = output_root / subdir / "mouse" / "imgs"
        destination.mkdir(parents=True)
        (destination / selected.name).symlink_to(selected)

    config = tmp_path / "reconcile.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "input_directory": str(input_root),
                "output_directory": str(output_root),
                "relevant_subdirs": ["kept", "rejected"],
                "output_columns": ["kept_human", "rejected_human"],
                "allowed_suffixes": [".jpg"],
            }
        ),
        encoding="utf-8",
    )
    return config, originals


def test_reconciliation_records_human_decisions_and_preserves_originals(
    tmp_path, reconciliation
):
    config, originals = reconciliation

    main(config)

    output_columns = {"kept": "kept_human", "rejected": "rejected_human"}
    for subdir, original in originals.items():
        output = tmp_path / "corrected" / subdir / "mouse"
        corrected = pd.read_csv(output / "records_corrected.csv", index_col=0)
        expected = original.iloc[[0]].copy()
        output_column = output_columns[subdir]
        expected[output_column] = True
        expected[f"not_{output_column}"] = False
        pd.testing.assert_frame_equal(corrected, expected)
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "original" / subdir / "mouse" / "records.csv"),
            original,
        )

    assert (tmp_path / "corrected" / config.name).read_bytes() == config.read_bytes()


@pytest.mark.parametrize("subdir", ["kept", "rejected"])
def test_empty_partition_is_rejected(tmp_path, reconciliation, subdir):
    config, _ = reconciliation
    image = (
        tmp_path / "corrected" / subdir / "mouse" / "imgs" / f"{subdir}_selected.jpg"
    )
    image.unlink()

    with pytest.raises(ValueError, match=f"Nothing {subdir}"):
        main(config)


def test_image_in_multiple_subdirs_is_rejected(tmp_path, reconciliation):
    config, originals = reconciliation
    shared_image = Path(originals["kept"].loc[0, "image_path"])
    rejected_source = tmp_path / "original" / "rejected" / "mouse" / "records.csv"
    rejected = pd.read_csv(rejected_source)
    rejected.loc[0, "image_path"] = str(shared_image)
    rejected.to_csv(rejected_source, index=False)

    rejected_imgs = tmp_path / "corrected" / "rejected" / "mouse" / "imgs"
    (rejected_imgs / "rejected_selected.jpg").unlink()
    (rejected_imgs / shared_image.name).symlink_to(shared_image)

    with pytest.raises(ValueError, match="more than one partition"):
        main(config)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=1_000_000),
            st.integers(min_value=1, max_value=1_000_000),
            st.sampled_from(["mouse_1.jpg", "mouse_2.jpg", "mouse_3.jpg"]),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_reconciliation_preserves_original_associations(rows):
    """Every materialized association must be copied unchanged from records.csv."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "original" / "kept" / "mouse"
        destination = root / "corrected" / "kept" / "mouse" / "imgs"
        source.mkdir(parents=True)
        destination.mkdir(parents=True)

        image_root = root / "images"
        image_root.mkdir()
        image_paths = {}
        for image_name in {row[2] for row in rows}:
            image = image_root / image_name
            image.write_bytes(b"image")
            image_paths[image_name] = str(image.resolve())

        original = pd.DataFrame(
            [(row[0], row[1], image_paths[row[2]]) for row in rows],
            columns=["id", "photo.id", "image_path"],
        )
        original.to_csv(source / "records.csv", index=False)

        selected_image = Path(image_paths[rows[0][2]])
        (destination / selected_image.name).symlink_to(selected_image)
        config = root / "reconcile.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "input_directory": str(root / "original"),
                    "output_directory": str(root / "corrected"),
                    "relevant_subdirs": ["kept"],
                    "output_columns": ["kept_human"],
                    "allowed_suffixes": [".jpg"],
                }
            ),
            encoding="utf-8",
        )

        main(config)

        corrected = pd.read_csv(
            destination.parent / "records_corrected.csv", index_col=0
        )
        association_columns = ["id", "photo.id", "image_path"]
        corrected_associations = set(
            corrected.loc[:, association_columns].itertuples(name=None, index=False)
        )
        original_associations = set(
            original.loc[:, association_columns].itertuples(name=None, index=False)
        )
        assert corrected_associations <= original_associations

        unassociated_image = image_root / "unassociated.jpg"
        unassociated_image.write_bytes(b"image")
        (destination / unassociated_image.name).symlink_to(unassociated_image)
        with pytest.raises(ValueError, match="Association"):
            main(config)
