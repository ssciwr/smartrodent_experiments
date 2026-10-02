"""Behavioral contract for stratified, train-only metadata oversampling."""

from copy import deepcopy
from math import ceil

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal


@pytest.fixture
def sampler_type():
    """Load the proposed interface without preventing test collection."""
    from smartrodent.dataprocessing import TrainingDatasetSampler

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
        ({"rng_seed": True}, ValueError),
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


@pytest.mark.parametrize(
    "invalid_multiplier", [0.0, -1.0, float("nan"), float("inf"), "two"]
)
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
    from smartrodent.dataprocessing import TrainingDatasetSampler

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
