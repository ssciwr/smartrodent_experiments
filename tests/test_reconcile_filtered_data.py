"""Tests for reconciling human-sorted image partitions with original records."""

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

from scripts.reconcile_filtered_data import check_association_consistency, main


@pytest.fixture
def reconciliation(tmp_path):
    """Create one species with an original image in each human partition."""
    source = tmp_path / "original" / "mouse"
    source.mkdir(parents=True)
    images = []
    for partition in ("kept", "rejected"):
        image = source / f"{partition}.jpg"
        image.write_bytes(b"image")
        images.append(str(image))
        destination = tmp_path / "corrected" / partition / "mouse" / "imgs"
        destination.mkdir(parents=True)
        (destination / image.name).symlink_to(image)

    records = pd.DataFrame(
        {
            "id": [10, 20],
            "photo.id": [1, 2],
            "image_path": images,
            # Human decisions must come from placement, not classifier flags.
            "kept_animal": [False, True],
            "rejected_animal": [True, False],
        }
    )
    records.to_csv(source / "original_records.csv", index=False)
    (tmp_path / "corrected" / "materialize_filtered_data.yaml").write_text(
        yaml.safe_dump({"input_dir": str(source.parent)}), encoding="utf-8"
    )
    config = tmp_path / "reconcile.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "input_directory": str(tmp_path / "corrected"),
                "output_directory": str(tmp_path / "output"),
                "allowed_suffixes": [".jpg"],
            }
        ),
        encoding="utf-8",
    )
    return config, records


def test_reconciliation_records_human_decisions_and_preserves_originals(
    tmp_path, reconciliation
):
    config, original = reconciliation

    main(config)

    output = tmp_path / "output" / "mouse"
    corrected_files = list(output.glob("original_records_corrected_*.csv"))
    assert len(corrected_files) == 1
    corrected = pd.read_csv(corrected_files[0], index_col=0).sort_index()
    expected = original.assign(kept_human=[True, False], rejected_human=[False, True])
    pd.testing.assert_frame_equal(corrected, expected)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "original_records_original.csv", index_col=0), original
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "original" / "mouse" / "original_records.csv"),
        original,
    )
    assert (tmp_path / "corrected" / config.name).read_bytes() == config.read_bytes()


@pytest.mark.parametrize("partition", ["kept", "rejected"])
def test_empty_partition_is_rejected(tmp_path, reconciliation, partition):
    config, _ = reconciliation
    image = tmp_path / "corrected" / partition / "mouse" / "imgs" / f"{partition}.jpg"
    image.unlink()

    with pytest.raises(ValueError, match=f"Nothing {partition}"):
        main(config)


def test_image_in_both_partitions_is_rejected(tmp_path, reconciliation):
    config, _ = reconciliation
    image = tmp_path / "original" / "mouse" / "kept.jpg"
    (tmp_path / "corrected" / "rejected" / "mouse" / "imgs" / image.name).symlink_to(
        image
    )

    with pytest.raises(ValueError, match="both"):
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
def test_association_validation_preserves_original_associations(rows):
    original = pd.DataFrame(rows, columns=["id", "photo.id", "image_path"])
    check_association_consistency(original.copy(), original)

    changed = original.copy()
    # Use an unseen ID so the changed association cannot match another row.
    changed.loc[0, "id"] = max(row[0] for row in rows) + 1
    with pytest.raises(ValueError, match="Association"):
        check_association_consistency(changed, original)
