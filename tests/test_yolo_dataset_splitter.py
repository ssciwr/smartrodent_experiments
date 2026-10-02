"""Behavioral contract for dataframe-only, leakage-safe YOLO partitioning.

The public seam is YoloDatasetSplitter.split(records_by_species). A nonempty
species with fewer than three independent groups must raise ValueError: three
nonempty splits cannot then coexist with group integrity and row preservation.
Empty species frames remain valid and return empty annotated dataframes.
"""

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal


@pytest.fixture
def splitter_type():
    """Load the proposed interface without preventing test collection."""
    from smartrodent.dataprocessing import YoloDatasetSplitter

    return YoloDatasetSplitter


def make_records(group_sizes, group_column="source_group_id"):
    """Build metadata rows for observations with unequal photo counts."""
    rows = [
        {
            group_column: group_id,
            "photo_index": photo_index,
            "image_url": f"https://example.test/{group_id}/{photo_index}.jpg",
            "local_path": f"/nonexistent/images/{group_id}_{photo_index}.jpg",
        }
        for group_id, size in enumerate(group_sizes)
        for photo_index in range(size)
    ]
    frame = pd.DataFrame(rows)
    # Nonconsecutive, duplicate indices must not be treated as row identifiers.
    frame.index = pd.Index([i // 2 + 10 for i in range(len(frame))], name="record")
    return frame


def make_splitter(splitter_type, **kwargs):
    """Configure the public interface with explicit GBIF grouping."""
    return splitter_type(group_columns=("source_group_id",), **kwargs)


@pytest.mark.parametrize("group_column", ["id", "source_group_id"])
def test_downloader_rows_are_preserved_and_inputs_are_not_modified(
    splitter_type, group_column
):
    records = {
        "Mus musculus": make_records([4, 3, 2, 1], group_column),
        "Rattus rattus": make_records([2, 2, 2, 2], group_column),
    }
    originals = {species: frame.copy(deep=True) for species, frame in records.items()}
    splitter = splitter_type(group_columns=(group_column,), rng_seed=42)

    result = splitter.split(records)

    assert set(result) == set(originals)
    for species, original in originals.items():
        annotated = result[species]
        assert annotated is not records[species]
        assert list(annotated.columns) == [*original.columns, "dataset_split"]
        assert_frame_equal(annotated.drop(columns="dataset_split"), original)
        assert_frame_equal(records[species], original)
        assert set(annotated["dataset_split"]) == {"train", "validation", "test"}
        assert annotated.groupby(group_column)["dataset_split"].nunique().eq(1).all()


def test_proportions_target_image_rows_not_observation_counts(splitter_type):
    records = make_records([70, 20, 10])
    splitter = make_splitter(splitter_type, rng_seed=42)

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    # These three indivisible observations admit an exact row-count allocation.
    assert result["dataset_split"].value_counts().to_dict() == {
        "train": 70,
        "validation": 20,
        "test": 10,
    }
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()


def test_small_species_get_nonempty_splits_despite_requested_proportions(splitter_type):
    splitter = make_splitter(splitter_type, train_val_test_split=(0.98, 0.01, 0.01))

    result = splitter.split({"Mus musculus": make_records([1, 1, 1])})["Mus musculus"]

    assert result["dataset_split"].value_counts().to_dict() == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


@pytest.mark.parametrize("group_sizes", [[1], [1, 1], [12], [12, 8]])
def test_fewer_than_three_independent_groups_are_rejected(splitter_type, group_sizes):
    records = make_records(group_sizes)
    original = records.copy(deep=True)
    splitter = make_splitter(splitter_type)

    # More images cannot compensate for too few indivisible observations.
    with pytest.raises(ValueError):
        splitter.split({"Mus musculus": records})

    assert_frame_equal(records, original)


def test_composite_group_keys_keep_observations_together(splitter_type):
    records = make_records([2] * 6)
    records["site"] = ["north"] * 6 + ["south"] * 6
    records["source_group_id"] = [0, 0, 1, 1, 2, 2] * 2
    splitter = splitter_type(
        group_columns=("site", "source_group_id"),
        train_val_test_split=(1 / 3, 1 / 3, 1 / 3),
    )

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    assert (
        result.groupby(["site", "source_group_id"])["dataset_split"]
        .nunique()
        .eq(1)
        .all()
    )
    assert result["dataset_split"].value_counts().to_dict() == {
        "train": 4,
        "validation": 4,
        "test": 4,
    }


def test_stratification_preserves_each_combination_of_columns(splitter_type):
    frames = []
    for country in ("DE", "FR"):
        for year in (2023, 2024):
            frame = make_records([2] * 10)
            frame["source_group_id"] += len(frames) * 10
            frame["country_code"] = country
            frame["year"] = year
            frames.append(frame)
    records = pd.concat(frames)
    splitter = make_splitter(splitter_type, stratify_columns=("country_code", "year"))

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    for _, stratum in result.groupby(["country_code", "year"]):
        assert stratum["dataset_split"].value_counts().to_dict() == {
            "train": 14,
            "validation": 4,
            "test": 2,
        }
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()


def test_assignments_are_repeatable_and_independent_of_input_order(splitter_type):
    records = {
        "Mus musculus": make_records([1, 2, 3, 4, 5, 6]),
        "Rattus rattus": make_records([2, 3, 4, 5, 6, 7]),
    }
    splitter = make_splitter(splitter_type, rng_seed=19)
    first = splitter.split(records)
    repeated = splitter.split(records)
    reordered = {
        species: frame.sample(frac=1, random_state=7)
        for species, frame in reversed(list(records.items()))
    }
    shuffled = make_splitter(splitter_type, rng_seed=19).split(reordered)

    for species in records:
        assert_frame_equal(first[species], repeated[species])
        assert_frame_equal(
            first[species].sort_values(["source_group_id", "photo_index"]),
            shuffled[species].sort_values(["source_group_id", "photo_index"]),
        )


def test_equal_sized_group_tie_breaking_is_row_order_invariant(splitter_type):
    records = make_records([2] * 12)
    splitter = make_splitter(splitter_type, rng_seed=19)
    expected = splitter.split({"Mus musculus": records})["Mus musculus"]

    for shuffle_seed in (7, 23, 41):
        shuffled = records.sample(frac=1, random_state=shuffle_seed)
        actual = make_splitter(splitter_type, rng_seed=19).split(
            {"Mus musculus": shuffled}
        )["Mus musculus"]

        assert_frame_equal(
            expected.sort_values(["source_group_id", "photo_index"]),
            actual.sort_values(["source_group_id", "photo_index"]),
        )


def test_empty_species_returns_an_empty_annotated_dataframe(splitter_type):
    records = make_records([1, 1, 1]).iloc[:0]

    result = make_splitter(splitter_type).split({"Mus musculus": records})[
        "Mus musculus"
    ]

    assert result.empty
    assert list(result.columns) == [*records.columns, "dataset_split"]
    assert_frame_equal(result.drop(columns="dataset_split"), records)


@pytest.mark.parametrize(
    "invalid_kind", ["missing_column", "null_id", "existing_split"]
)
def test_invalid_input_is_rejected_without_mutation(splitter_type, invalid_kind):
    records = make_records([2, 2, 2])
    if invalid_kind == "missing_column":
        records = records.drop(columns="source_group_id")
    elif invalid_kind == "null_id":
        records["source_group_id"] = records["source_group_id"].astype("Int64")
        records["source_group_id"] = pd.array([pd.NA, 0, 1, 1, 2, 2], dtype="Int64")
    elif invalid_kind == "existing_split":
        records["dataset_split"] = "train"
    else:
        raise AssertionError(f"Unknown test case: {invalid_kind}")
    original = records.copy(deep=True)

    with pytest.raises(ValueError):
        make_splitter(splitter_type).split({"Mus musculus": records})

    assert_frame_equal(records, original)


def test_missing_stratification_column_is_rejected(splitter_type):
    splitter = make_splitter(splitter_type, stratify_columns=("country_code",))

    with pytest.raises(ValueError):
        splitter.split({"Mus musculus": make_records([2, 2, 2])})


def test_conflicting_strata_within_an_observation_are_rejected(splitter_type):
    records = make_records([2, 2, 2])
    records["country_code"] = ["DE", "FR", "DE", "DE", "FR", "FR"]
    splitter = make_splitter(splitter_type, stratify_columns=("country_code",))

    with pytest.raises(ValueError):
        splitter.split({"Mus musculus": records})


@pytest.mark.parametrize(
    "fractions",
    [
        (0.7, 0.2, 0.2),
        (-0.1, 0.5, 0.6),
        (float("nan"), 0.2, 0.1),
        (float("inf"), 0.2, 0.1),
        (0.8, 0.2),
    ],
)
def test_invalid_fractions_are_rejected(splitter_type, fractions):
    with pytest.raises(ValueError):
        make_splitter(splitter_type, train_val_test_split=fractions)


@pytest.mark.parametrize(
    ("columns", "error"),
    [
        ((), ValueError),
        (("id", "id"), ValueError),
        (("",), ValueError),
        ("id", TypeError),
    ],
)
def test_invalid_group_columns_are_rejected(splitter_type, columns, error):
    with pytest.raises(error):
        splitter_type(group_columns=columns)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seeds_are_rejected(splitter_type, seed):
    with pytest.raises(ValueError):
        make_splitter(splitter_type, rng_seed=seed)


@pytest.mark.parametrize(
    "group_sizes,countries",
    [([4, 3, 2], ["DE", "FR", "GB"]), ([2, 2, 2, 2], ["DE", "DE", "FR", "FR"])],
)
def test_sparse_strata_still_populate_all_species_splits(
    splitter_type, group_sizes, countries
):
    records = make_records(group_sizes)
    records["country_code"] = [
        country
        for size, country in zip(group_sizes, countries, strict=True)
        for _ in range(size)
    ]
    splitter = make_splitter(splitter_type, stratify_columns=("country_code",))

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    assert set(result["dataset_split"]) == {"train", "validation", "test"}
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()
    assert_frame_equal(result.drop(columns="dataset_split"), records)


def test_zero_target_fractions_do_not_override_nonempty_splits(splitter_type):
    splitter = make_splitter(splitter_type, train_val_test_split=(1.0, 0.0, 0.0))

    result = splitter.split({"Mus musculus": make_records([4, 2, 1])})["Mus musculus"]

    assert result["dataset_split"].value_counts().to_dict() == {
        "train": 4,
        "validation": 2,
        "test": 1,
    } or result["dataset_split"].value_counts().to_dict() == {
        "train": 4,
        "validation": 1,
        "test": 2,
    }


def test_null_strata_and_duplicate_columns_are_rejected(splitter_type):
    records = make_records([2, 2, 2])
    records["country_code"] = pd.array([pd.NA, "DE", "FR", "FR", "GB", "GB"])
    splitter = make_splitter(splitter_type, stratify_columns=("country_code",))

    with pytest.raises(ValueError):
        splitter.split({"Mus musculus": records})

    duplicate_columns = pd.concat([records, records[["source_group_id"]]], axis=1)
    with pytest.raises(ValueError):
        make_splitter(splitter_type).split({"Mus musculus": duplicate_columns})


@given(
    group_sizes=st.lists(
        st.integers(min_value=1, max_value=12), min_size=1, max_size=25
    ),
    stratify_columns=st.sampled_from([(), ("country_code",), ("country_code", "year")]),
)
def test_assign_strata_contains_each_row_position_exactly_once(
    group_sizes, stratify_columns
):
    """Check the agreed private seam for omissions and duplicate membership."""
    from smartrodent.dataprocessing import YoloDatasetSplitter

    records = make_records(group_sizes)
    records["country_code"] = records["source_group_id"].map(
        lambda group_id: "DE" if group_id % 2 == 0 else "FR"
    )
    records["year"] = 2023 + records["source_group_id"] % 3
    # Shuffling interleaves groups while retaining duplicate index labels.
    records = records.sample(frac=1, random_state=7)
    splitter = YoloDatasetSplitter(
        group_columns=("source_group_id",), stratify_columns=stratify_columns
    )

    strata = splitter._assign_strata(records)

    positions = [
        int(position)
        for groups in strata.values()
        for group in groups
        for position in group
    ]
    # Sorted list equality checks multiplicity as well as coverage: a set
    # comparison alone would miss positions occurring in multiple groups.
    assert sorted(positions) == list(range(len(records)))
    for key, groups in strata.items():
        for group in groups:
            rows = records.iloc[group]
            assert rows["source_group_id"].nunique() == 1
            group_id = int(rows["source_group_id"].iloc[0])
            assert len(rows) == group_sizes[group_id]
            assert all(
                tuple(row) == key
                for row in rows[list(stratify_columns)].itertuples(
                    index=False, name=None
                )
            )


@given(
    group_sizes=st.lists(
        st.integers(min_value=1, max_value=12), min_size=3, max_size=25
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_generated_observations_never_leak_or_lose_rows(group_sizes, seed):
    """Exercise leakage and preservation through the same public seam as callers."""
    from smartrodent.dataprocessing import YoloDatasetSplitter

    records = make_records(group_sizes)
    original = records.copy(deep=True)
    splitter = YoloDatasetSplitter(group_columns=("source_group_id",), rng_seed=seed)

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    assert_frame_equal(result.drop(columns="dataset_split"), original)
    assert_frame_equal(records, original)
    assert set(result["dataset_split"]) == {"train", "validation", "test"}
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()
    assert_frame_equal(
        result, splitter.split({"Mus musculus": records})["Mus musculus"]
    )
