"""Behavioral contracts for DatasetSplitter and TrainingDatasetSampler.

DatasetSplitter performs dataframe-only, leakage-safe YOLO partitioning; its
public seam is DatasetSplitter.split(records_by_species). A nonempty
species with fewer than three independent groups must raise ValueError: three
nonempty splits cannot then coexist with group integrity and row preservation.
Empty species frames remain valid and return empty annotated dataframes.
"""

from copy import deepcopy
from math import ceil

import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from smartrodent.base import Configurable


@pytest.fixture
def splitter_type():
    """Load the proposed interface without preventing test collection."""
    from smartrodent.dataset_splitting import DatasetSplitter

    return DatasetSplitter


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
    """Configure the public interface with an explicit, non-default grouping column."""
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


@pytest.mark.parametrize("group_sizes", [[1], [12, 8]])
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
    from smartrodent.dataset_splitting import DatasetSplitter

    records = make_records(group_sizes)
    records["country_code"] = records["source_group_id"].map(
        lambda group_id: "DE" if group_id % 2 == 0 else "FR"
    )
    records["year"] = 2023 + records["source_group_id"] % 3
    # Shuffling interleaves groups while retaining duplicate index labels.
    records = records.sample(frac=1, random_state=7)
    splitter = DatasetSplitter(
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
    from smartrodent.dataset_splitting import DatasetSplitter

    records = make_records(group_sizes)
    original = records.copy(deep=True)
    splitter = DatasetSplitter(group_columns=("source_group_id",), rng_seed=seed)

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    assert_frame_equal(result.drop(columns="dataset_split"), original)
    assert_frame_equal(records, original)
    assert set(result["dataset_split"]) == {"train", "validation", "test"}
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()
    assert_frame_equal(
        result, splitter.split({"Mus musculus": records})["Mus musculus"]
    )


def test_from_config_loads_splitter_settings_and_creates_a_working_instance(
    tmp_path, splitter_type
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_splitter": {
                        "group_columns": ["source_group_id"],
                        "stratify_columns": ["photo_index"],
                        "train_val_test_split": [0.6, 0.3, 0.1],
                        "rng_seed": 9,
                    }
                }
            }
        )
    )

    splitter = splitter_type.from_config(config_path)
    result = splitter.split({"Mus musculus": make_records([1, 1, 1])})["Mus musculus"]

    assert isinstance(splitter, Configurable)
    assert splitter.group_columns == ("source_group_id",)
    assert splitter.stratify_columns == ("photo_index",)
    assert tuple(splitter.fractions) == pytest.approx((0.6, 0.3, 0.1))
    assert splitter.rng_seed == 9
    assert set(result["dataset_split"]) == {"train", "validation", "test"}


def test_from_config_uses_splitter_constructor_defaults(tmp_path, splitter_type):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"data": {"dataset_splitter": {"group_columns": ["source_group_id"]}}}
        )
    )

    splitter = splitter_type.from_config(config_path)

    assert splitter.group_columns == ("source_group_id",)
    assert splitter.stratify_columns == ()
    assert tuple(splitter.fractions) == pytest.approx((0.7, 0.2, 0.1))
    assert splitter.rng_seed == 42


@pytest.mark.parametrize(
    "content",
    ["[]\n", "data: []\n", "data:\n  dataset_splitter: []\n"],
)
def test_from_config_rejects_invalid_splitter_configuration_structure(
    tmp_path, splitter_type, content
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)

    with pytest.raises(TypeError):
        splitter_type.from_config(config_path)


def test_from_config_rejects_a_missing_splitter_file(tmp_path, splitter_type):
    with pytest.raises(FileNotFoundError):
        splitter_type.from_config(tmp_path / "missing.yaml")


def test_from_config_requires_splitter_group_column(tmp_path, splitter_type):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": {"dataset_splitter": {}}}))

    with pytest.raises(ValueError, match="group_columns"):
        splitter_type.from_config(config_path)


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"group_columns": "source_group_id"}, TypeError),
        (
            {
                "group_columns": ["source_group_id"],
                "train_val_test_split": [0.7, 0.2, 0.2],
            },
            ValueError,
        ),
        ({"group_columns": ["source_group_id"], "rng_seed": -1}, ValueError),
    ],
)
def test_from_config_delegates_splitter_setting_validation(
    tmp_path, splitter_type, settings, error
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": {"dataset_splitter": settings}}))

    with pytest.raises(error):
        splitter_type.from_config(config_path)


@pytest.fixture
def sampler_type():
    """Load the proposed interface without preventing test collection."""
    from smartrodent.dataset_splitting import TrainingDatasetSampler

    return TrainingDatasetSampler


def make_split_records(marker="mouse"):
    """Build one species frame with coherent observation-level strata.

    Training has four German and two French image rows. Validation and test each
    contain both countries, so their state can be checked independently.
    """
    rows = [
        (10, 0, "DE", 2023, "train"),
        (10, 1, "DE", 2023, "train"),
        (10, 2, "DE", 2023, "train"),
        (11, 0, "DE", 2023, "train"),
        (20, 0, "FR", 2024, "train"),
        (20, 1, "FR", 2024, "train"),
        (30, 0, "DE", 2023, "validation"),
        (30, 1, "DE", 2023, "validation"),
        (31, 0, "FR", 2024, "validation"),
        (40, 0, "DE", 2023, "test"),
        (41, 0, "FR", 2024, "test"),
        (41, 1, "FR", 2024, "test"),
    ]
    records = pd.DataFrame(
        rows,
        columns=[
            "source_group_id",
            "photo_index",
            "country_code",
            "year",
            "dataset_split",
        ],
    )
    records["image_url"] = [
        f"https://example.test/{marker}/{group}/{photo}.jpg"
        for group, photo in zip(
            records["source_group_id"], records["photo_index"], strict=True
        )
    ]
    records["marker"] = marker
    records.index = pd.Index([10 + index // 2 for index in range(len(records))])
    return records


def make_sampler(sampler_type, rule, **kwargs):
    """Configure the public interface with downloader-compatible columns."""
    return sampler_type(
        group_columns=("source_group_id",),
        stratify_columns=("country_code", "year"),
        oversampling_rule=rule,
        **kwargs,
    )


def constant_rule(multiplier):
    """Return an oversampling rule with the public three-argument signature."""

    def rule(_species, _frame, _all_frames):
        return multiplier

    return rule


def non_training_rows(frame):
    """Return evaluation rows in their original order and with their index."""
    return frame.loc[frame["dataset_split"].isin(["validation", "test"])]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"group_columns": "source_group_id"}, TypeError),
        ({"stratify_columns": ("country_code", "country_code")}, ValueError),
        ({"oversampling_rule": None}, TypeError),
        ({"rng_seed": -1}, ValueError),
        ({"rng_seed": True}, ValueError),  # the sampler's own bool check
    ],
)
def test_invalid_sampler_configuration_is_rejected(sampler_type, kwargs, error):
    configuration = {
        "group_columns": ("source_group_id",),
        "stratify_columns": ("country_code", "year"),
        "oversampling_rule": constant_rule(1.0),
    }
    configuration.update(kwargs)

    with pytest.raises(error):
        sampler_type(**configuration)


def test_multiplier_one_is_a_strict_no_op(sampler_type):
    records_by_species = {
        "Mus musculus": make_split_records("mouse"),
        "Rattus rattus": make_split_records("rat"),
    }
    originals = {
        species: frame.copy(deep=True) for species, frame in records_by_species.items()
    }

    result = make_sampler(sampler_type, constant_rule(1.0)).sample(records_by_species)

    for species, original in originals.items():
        assert result[species] is not records_by_species[species]
        assert_frame_equal(result[species], original)
        assert_frame_equal(records_by_species[species], original)


def test_sampler_reaches_each_training_stratum_target_and_preserves_evaluation_rows(
    sampler_type,
):
    records = make_split_records()
    original = records.copy(deep=True)

    result = make_sampler(sampler_type, constant_rule(1.5), rng_seed=42).sample(
        {"Mus musculus": records}
    )["Mus musculus"]

    train = result.loc[result["dataset_split"] == "train"]
    assert train.groupby(["country_code", "year"]).size().to_dict() == {
        ("DE", 2023): 6,
        ("FR", 2024): 3,
    }
    assert_frame_equal(non_training_rows(result), non_training_rows(original))
    assert_frame_equal(records, original)


def test_multiplier_scales_each_training_stratum(sampler_type):
    records = make_split_records()

    result = make_sampler(sampler_type, constant_rule(2.0), rng_seed=42).sample(
        {"Mus musculus": records}
    )["Mus musculus"]

    counts = (
        result.loc[result["dataset_split"] == "train"]
        .groupby(["country_code", "year"])
        .size()
        .to_dict()
    )
    assert counts == {("DE", 2023): 8, ("FR", 2024): 4}


def test_appended_rows_are_existing_training_rows_with_unchanged_metadata(sampler_type):
    records = make_split_records()

    result = make_sampler(sampler_type, constant_rule(2.0), rng_seed=42).sample(
        {"Mus musculus": records}
    )["Mus musculus"]
    copied_rows = result.iloc[len(records) :]
    original_train_records = records.loc[records["dataset_split"] == "train"].to_dict(
        "records"
    )

    assert len(copied_rows) == 6
    assert copied_rows["dataset_split"].eq("train").all()
    assert all(row in original_train_records for row in copied_rows.to_dict("records"))
    assert set(copied_rows["source_group_id"]).issubset({10, 11, 20})
    assert set(
        copied_rows[["country_code", "year"]].itertuples(index=False, name=None)
    ).issubset({("DE", 2023), ("FR", 2024)})


def test_rule_can_use_all_species_context_and_apply_distinct_multipliers(sampler_type):
    records_by_species = {
        "Mus musculus": make_split_records("mouse"),
        "Rattus rattus": make_split_records("rat"),
    }
    observed_contexts = []

    def rule(species, _frame, all_frames):
        observed_contexts.append((species, set(all_frames)))
        return 2.0 if species == "Mus musculus" else 1.0

    result = make_sampler(sampler_type, rule, rng_seed=42).sample(records_by_species)

    assert observed_contexts == [
        ("Mus musculus", {"Mus musculus", "Rattus rattus"}),
        ("Rattus rattus", {"Mus musculus", "Rattus rattus"}),
    ]
    assert (result["Mus musculus"]["dataset_split"] == "train").sum() == 12
    assert_frame_equal(result["Rattus rattus"], records_by_species["Rattus rattus"])


def test_same_seed_produces_identical_copies(sampler_type):
    records_by_species = {"Mus musculus": make_split_records()}

    first = make_sampler(sampler_type, constant_rule(4.0), rng_seed=19).sample(
        records_by_species
    )
    repeated = make_sampler(sampler_type, constant_rule(4.0), rng_seed=19).sample(
        records_by_species
    )

    assert_frame_equal(first["Mus musculus"], repeated["Mus musculus"])


@pytest.mark.parametrize("invalid_multiplier", [0.0, float("nan"), "two"])
def test_invalid_rule_result_is_rejected_without_mutating_inputs(
    sampler_type, invalid_multiplier
):
    records_by_species = {"Mus musculus": make_split_records()}
    originals = deepcopy(records_by_species)

    with pytest.raises(ValueError):
        make_sampler(sampler_type, constant_rule(invalid_multiplier)).sample(
            records_by_species
        )

    for species, original in originals.items():
        assert_frame_equal(records_by_species[species], original)


@pytest.mark.parametrize(
    "invalid_kind", ["missing_split", "missing_group", "null_group"]
)
def test_invalid_training_metadata_is_rejected_without_mutating_inputs(
    sampler_type, invalid_kind
):
    records = make_split_records()
    if invalid_kind == "missing_split":
        records = records.drop(columns="dataset_split")
    elif invalid_kind == "missing_group":
        records = records.drop(columns="source_group_id")
    elif invalid_kind == "null_group":
        records["source_group_id"] = records["source_group_id"].astype("Int64")
        records.iloc[0, 0] = pd.NA
    else:
        raise AssertionError(f"Unknown test case: {invalid_kind}")
    original = records.copy(deep=True)

    with pytest.raises(ValueError):
        make_sampler(sampler_type, constant_rule(2.0)).sample({"Mus musculus": records})

    assert_frame_equal(records, original)


@pytest.mark.parametrize(
    "invalid_kind", ["non_dataframe", "duplicate_columns", "invalid_split"]
)
def test_invalid_input_frame_is_rejected(sampler_type, invalid_kind):
    records = make_split_records()
    if invalid_kind == "non_dataframe":
        records = []
    elif invalid_kind == "duplicate_columns":
        records = pd.concat([records, records[["year"]]], axis=1)
    elif invalid_kind == "invalid_split":
        records.loc[records.index[0], "dataset_split"] = "development"
    else:
        raise AssertionError(f"Unknown test case: {invalid_kind}")

    with pytest.raises(ValueError):
        make_sampler(sampler_type, constant_rule(2.0)).sample({"Mus musculus": records})


def test_observation_with_conflicting_training_strata_is_rejected(sampler_type):
    records = make_split_records()
    records.loc[records["source_group_id"] == 10, "country_code"] = ["DE", "FR", "DE"]
    original = records.copy(deep=True)

    with pytest.raises(ValueError):
        make_sampler(sampler_type, constant_rule(2.0)).sample({"Mus musculus": records})

    assert_frame_equal(records, original)


def test_sampler_without_stratification_scales_all_training_rows(sampler_type):
    records = make_split_records()
    sampler = sampler_type(
        group_columns=("source_group_id",),
        stratify_columns=(),
        oversampling_rule=constant_rule(1.5),
        rng_seed=42,
    )

    result = sampler.sample({"Mus musculus": records})["Mus musculus"]

    assert (result["dataset_split"] == "train").sum() == 9
    assert_frame_equal(non_training_rows(result), non_training_rows(records))


def test_multiplier_above_one_requires_training_rows(sampler_type):
    records = make_split_records()
    records["dataset_split"] = "validation"

    with pytest.raises(ValueError):
        make_sampler(sampler_type, constant_rule(2.0)).sample({"Mus musculus": records})


@given(multiplier=st.floats(min_value=1.0, max_value=5.0, allow_nan=False))
def test_generated_valid_multiplier_preserves_per_stratum_allocation_invariants(
    multiplier,
):
    """Exercise target counts and copied-row validity through the public seam."""
    from smartrodent.dataset_splitting import TrainingDatasetSampler

    records = make_split_records()
    result = TrainingDatasetSampler(
        group_columns=("source_group_id",),
        stratify_columns=("country_code", "year"),
        oversampling_rule=constant_rule(multiplier),
        rng_seed=42,
    ).sample({"Mus musculus": records})["Mus musculus"]

    original_train = records.loc[records["dataset_split"] == "train"]
    sampled_train = result.loc[result["dataset_split"] == "train"]
    for stratum, source_rows in original_train.groupby(["country_code", "year"]):
        sampled_count = len(
            sampled_train.loc[
                (sampled_train["country_code"] == stratum[0])
                & (sampled_train["year"] == stratum[1])
            ]
        )
        assert sampled_count == ceil(len(source_rows) * multiplier)
    assert_frame_equal(non_training_rows(result), non_training_rows(records))
    source_records = original_train.to_dict("records")
    assert all(
        row in source_records for row in result.iloc[len(records) :].to_dict("records")
    )


def test_from_config_resolves_rule_and_uses_runtime_context(
    tmp_path, monkeypatch, sampler_type
):
    (tmp_path / "sampler_config_rules.py").write_text(
        "def double_when_context_is_complete(species, frame, all_frames):\n"
        "    if species == 'Mus musculus' and all_frames[species] is frame:\n"
        "        return 2.0\n"
        "    return 1.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "training_dataset_sampler": {
                        "group_columns": ["source_group_id"],
                        "stratify_columns": ["country_code"],
                        "oversampling_rule": (
                            "sampler_config_rules:double_when_context_is_complete"
                        ),
                        "rng_seed": 9,
                    }
                }
            }
        )
    )

    sampler = sampler_type.from_config(config_path)
    result = sampler.sample({"Mus musculus": make_split_records()})["Mus musculus"]

    assert sampler.rng_seed == 9
    assert isinstance(sampler, Configurable)
    assert (result["dataset_split"] == "train").sum() == 12


def test_from_config_uses_sampler_default_seed(tmp_path, monkeypatch, sampler_type):
    (tmp_path / "sampler_default_rules.py").write_text(
        "def unchanged(*args):\n    return 1.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "training_dataset_sampler": {
                        "group_columns": ["source_group_id"],
                        "stratify_columns": ["country_code"],
                        "oversampling_rule": "sampler_default_rules:unchanged",
                    }
                }
            }
        )
    )

    sampler = sampler_type.from_config(config_path)

    assert sampler.rng_seed == 42


@pytest.mark.parametrize(
    "content",
    [
        "[]\n",
        "data: []\n",
        "data:\n  training_dataset_sampler: []\n",
    ],
)
def test_from_config_rejects_invalid_sampler_configuration_structure(
    tmp_path, sampler_type, content
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content)

    with pytest.raises(TypeError):
        sampler_type.from_config(config_path)


@pytest.mark.parametrize(
    "settings,missing_setting",
    [
        (
            {"stratify_columns": ["country_code"], "oversampling_rule": "rules:rule"},
            "group_columns",
        ),
        (
            {"group_columns": ["source_group_id"], "oversampling_rule": "rules:rule"},
            "stratify_columns",
        ),
        (
            {
                "group_columns": ["source_group_id"],
                "stratify_columns": ["country_code"],
            },
            "oversampling_rule",
        ),
    ],
)
def test_from_config_requires_sampler_settings(
    tmp_path, sampler_type, settings, missing_setting
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"data": {"training_dataset_sampler": settings}})
    )

    with pytest.raises(ValueError, match=missing_setting):
        sampler_type.from_config(config_path)


@pytest.mark.parametrize("reference", [123, "no_separator", "missing_module:rule"])
def test_from_config_rejects_unresolvable_sampler_rule(
    tmp_path, sampler_type, reference
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "training_dataset_sampler": {
                        "group_columns": ["source_group_id"],
                        "stratify_columns": ["country_code"],
                        "oversampling_rule": reference,
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError):
        sampler_type.from_config(config_path)


def test_from_config_rejects_missing_or_non_callable_sampler_rule(
    tmp_path, monkeypatch, sampler_type
):
    (tmp_path / "invalid_sampler_rules.py").write_text("not_a_rule = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    for reference, error in [
        ("invalid_sampler_rules:missing_rule", ValueError),
        ("invalid_sampler_rules:not_a_rule", TypeError),
    ]:
        config_path = tmp_path / f"{reference.split(':')[1]}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "data": {
                        "training_dataset_sampler": {
                            "group_columns": ["source_group_id"],
                            "stratify_columns": ["country_code"],
                            "oversampling_rule": reference,
                        }
                    }
                }
            )
        )

        with pytest.raises(error):
            sampler_type.from_config(config_path)


def test_from_config_rejects_a_missing_sampler_file(tmp_path, sampler_type):
    with pytest.raises(FileNotFoundError):
        sampler_type.from_config(tmp_path / "missing.yaml")


def test_from_config_rejects_invalid_sampler_static_settings(
    tmp_path, monkeypatch, sampler_type
):
    (tmp_path / "sampler_invalid_settings_rules.py").write_text(
        "def unchanged(*args):\n    return 1.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "training_dataset_sampler": {
                        "group_columns": "source_group_id",
                        "stratify_columns": ["country_code"],
                        "oversampling_rule": "sampler_invalid_settings_rules:unchanged",
                        "rng_seed": -1,
                    }
                }
            }
        )
    )

    with pytest.raises(TypeError):
        sampler_type.from_config(config_path)
