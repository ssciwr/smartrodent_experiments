"""Behavioral contracts for DatasetSplitter and TrainingDatasetSampler.

DatasetSplitter performs dataframe-only, leakage-safe YOLO partitioning. A nonempty
species with fewer than three independent groups must raise ValueError: three
nonempty splits cannot then coexist with group integrity and row preservation.
TrainingDatasetSampler then oversamples only training rows, per stratum.
"""

from math import ceil

import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from smartrodent.dataset_splitting import DatasetSplitter, TrainingDatasetSampler


def make_records(group_sizes, group_column="source_group_id"):
    """Build metadata rows for observations with unequal photo counts."""
    rows = [
        {
            group_column: group_id,
            "photo_index": photo_index,
            "image_url": f"https://example.test/{group_id}/{photo_index}.jpg",
        }
        for group_id, size in enumerate(group_sizes)
        for photo_index in range(size)
    ]
    frame = pd.DataFrame(rows)
    # Nonconsecutive, duplicate indices must not be treated as row identifiers.
    frame.index = pd.Index([i // 2 + 10 for i in range(len(frame))], name="record")
    return frame


def make_splitter(**kwargs):
    """Configure the splitter with an explicit, non-default grouping column."""
    return DatasetSplitter(group_columns=("source_group_id",), **kwargs)


def split_counts(splitter, records):
    """Split one species and return its row count per split."""
    result = splitter.split({"Mus musculus": records})["Mus musculus"]
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()
    return result["dataset_split"].value_counts().to_dict()


@pytest.mark.parametrize("group_column", ["id", "source_group_id"])
def test_rows_are_annotated_without_modifying_inputs(group_column):
    records = {
        "Mus musculus": make_records([4, 3, 2, 1], group_column),
        "Rattus rattus": make_records([2, 2, 2, 2], group_column),
    }
    originals = {species: frame.copy(deep=True) for species, frame in records.items()}

    result = DatasetSplitter(group_columns=(group_column,)).split(records)

    for species, original in originals.items():
        annotated = result[species]
        assert_frame_equal(annotated.drop(columns="dataset_split"), original)
        assert_frame_equal(records[species], original)
        assert set(annotated["dataset_split"]) == {"train", "validation", "test"}
        assert annotated.groupby(group_column)["dataset_split"].nunique().eq(1).all()


def test_proportions_target_image_rows_not_observation_counts():
    # These three indivisible observations admit an exact row-count allocation.
    assert split_counts(make_splitter(), make_records([70, 20, 10])) == {
        "train": 70,
        "validation": 20,
        "test": 10,
    }


@pytest.mark.parametrize("fractions", [(0.98, 0.01, 0.01), (1.0, 0.0, 0.0)])
def test_every_split_stays_nonempty_whatever_the_requested_proportions(fractions):
    counts = split_counts(
        make_splitter(train_val_test_split=fractions), make_records([4, 2, 1])
    )
    assert set(counts) == {"train", "validation", "test"}


def test_composite_group_keys_keep_observations_together():
    records = make_records([2] * 6)
    records["site"] = ["north"] * 6 + ["south"] * 6
    records["source_group_id"] = [0, 0, 1, 1, 2, 2] * 2
    splitter = DatasetSplitter(
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


def test_stratification_preserves_each_combination_of_columns():
    frames = []
    for country in ("DE", "FR"):
        for year in (2023, 2024):
            frame = make_records([2] * 10)
            frame["source_group_id"] += len(frames) * 10
            frame["country_code"] = country
            frame["year"] = year
            frames.append(frame)
    splitter = make_splitter(stratify_columns=("country_code", "year"))

    result = splitter.split({"Mus musculus": pd.concat(frames)})["Mus musculus"]

    for _, stratum in result.groupby(["country_code", "year"]):
        assert stratum["dataset_split"].value_counts().to_dict() == {
            "train": 14,
            "validation": 4,
            "test": 2,
        }


@pytest.mark.parametrize(
    "group_sizes,countries",
    [([4, 3, 2], ["DE", "FR", "GB"]), ([2, 2, 2, 2], ["DE", "DE", "FR", "FR"])],
)
def test_sparse_strata_still_populate_all_species_splits(group_sizes, countries):
    records = make_records(group_sizes)
    records["country_code"] = [
        country
        for size, country in zip(group_sizes, countries, strict=True)
        for _ in range(size)
    ]
    counts = split_counts(make_splitter(stratify_columns=("country_code",)), records)
    assert set(counts) == {"train", "validation", "test"}


def test_assignments_are_independent_of_species_and_row_order():
    records = {
        "Mus musculus": make_records([1, 2, 3, 4, 5, 6]),
        "Rattus rattus": make_records([2] * 12),
    }
    first = make_splitter(rng_seed=19).split(records)
    reordered = {
        species: frame.sample(frac=1, random_state=7)
        for species, frame in reversed(list(records.items()))
    }
    shuffled = make_splitter(rng_seed=19).split(reordered)

    for species in records:
        assert_frame_equal(
            first[species].sort_values(["source_group_id", "photo_index"]),
            shuffled[species].sort_values(["source_group_id", "photo_index"]),
        )


def test_empty_species_returns_an_empty_annotated_dataframe():
    records = make_records([1, 1, 1]).iloc[:0]

    result = make_splitter().split({"Mus musculus": records})["Mus musculus"]

    assert result.empty
    assert list(result.columns) == [*records.columns, "dataset_split"]


def _invalid_split_records(kind):
    """Build species metadata that the splitter must reject."""
    records = make_records([2, 2, 2])
    records["country_code"] = ["DE", "DE", "FR", "FR", "GB", "GB"]
    if kind == "too_few_groups":
        records = make_records([12, 8])
        records["country_code"] = "DE"
    elif kind == "missing_group_column":
        records = records.drop(columns="source_group_id")
    elif kind == "missing_stratum_column":
        records = records.drop(columns="country_code")
    elif kind == "null_group":
        records["source_group_id"] = pd.array([pd.NA, 0, 1, 1, 2, 2], dtype="Int64")
    elif kind == "null_stratum":
        records["country_code"] = pd.array([pd.NA, "DE", "FR", "FR", "GB", "GB"])
    elif kind == "conflicting_strata":
        records["country_code"] = ["DE", "FR", "DE", "DE", "FR", "FR"]
    elif kind == "existing_split":
        records["dataset_split"] = "train"
    elif kind == "duplicate_columns":
        records = pd.concat([records, records[["source_group_id"]]], axis=1)
    else:
        raise AssertionError(f"Unknown case: {kind}")
    return records


@pytest.mark.parametrize(
    "kind",
    [
        "too_few_groups",
        "missing_group_column",
        "missing_stratum_column",
        "null_group",
        "null_stratum",
        "conflicting_strata",
        "existing_split",
        "duplicate_columns",
    ],
)
def test_ambiguous_metadata_is_rejected_without_mutation(kind):
    records = _invalid_split_records(kind)
    original = records.copy(deep=True)

    with pytest.raises(ValueError):
        make_splitter(stratify_columns=("country_code",)).split(
            {"Mus musculus": records}
        )

    assert_frame_equal(records, original)


# Our declared rules: at least one unique group column, fractions summing to 1.
@pytest.mark.parametrize(
    "settings",
    [
        {"group_columns": ()},
        {"group_columns": ("id", "id")},
        {"group_columns": ("id",), "train_val_test_split": (0.7, 0.2, 0.2)},
    ],
)
def test_invalid_splitter_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        DatasetSplitter(**settings)


@given(
    group_sizes=st.lists(
        st.integers(min_value=1, max_value=12), min_size=3, max_size=25
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_generated_observations_never_leak_or_lose_rows(group_sizes, seed):
    records = make_records(group_sizes)
    splitter = make_splitter(rng_seed=seed)

    result = splitter.split({"Mus musculus": records})["Mus musculus"]

    assert_frame_equal(result.drop(columns="dataset_split"), records)
    assert set(result["dataset_split"]) == {"train", "validation", "test"}
    assert result.groupby("source_group_id")["dataset_split"].nunique().eq(1).all()
    assert_frame_equal(
        result, splitter.split({"Mus musculus": records})["Mus musculus"]
    )


def test_splitter_from_config_splits_with_the_configured_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_splitter": {
                        "group_columns": ["source_group_id"],
                        "train_val_test_split": [0.7, 0.2, 0.1],
                        "rng_seed": 9,
                    }
                }
            }
        )
    )

    splitter = DatasetSplitter.from_config(config_path)

    assert split_counts(splitter, make_records([70, 20, 10])) == {
        "train": 70,
        "validation": 20,
        "test": 10,
    }


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
    records["marker"] = marker
    records.index = pd.Index([10 + index // 2 for index in range(len(records))])
    return records


def constant_rule(multiplier):
    """Return an oversampling rule with the public three-argument signature."""

    def rule(_species, _frame, _all_frames):
        return multiplier

    return rule


def make_sampler(rule, **kwargs):
    """Configure the sampler with downloader-compatible columns."""
    return TrainingDatasetSampler(
        group_columns=("source_group_id",),
        stratify_columns=("country_code", "year"),
        oversampling_rule=rule,
        **kwargs,
    )


def non_training_rows(frame):
    """Return evaluation rows in their original order and with their index."""
    return frame.loc[frame["dataset_split"].isin(["validation", "test"])]


def test_multiplier_one_is_a_strict_no_op():
    records = make_split_records()

    result = make_sampler(constant_rule(1.0)).sample({"Mus musculus": records})

    assert_frame_equal(result["Mus musculus"], records)


@given(multiplier=st.floats(min_value=1.0, max_value=5.0, allow_nan=False))
def test_each_training_stratum_reaches_its_target_with_copies_of_training_rows(
    multiplier,
):
    records = make_split_records()
    original = records.copy(deep=True)

    result = make_sampler(constant_rule(multiplier)).sample({"Mus musculus": records})[
        "Mus musculus"
    ]

    original_train = records.loc[records["dataset_split"] == "train"]
    sampled_train = result.loc[result["dataset_split"] == "train"]
    for stratum, source_rows in original_train.groupby(["country_code", "year"]):
        sampled = sampled_train.groupby(["country_code", "year"]).get_group(stratum)
        assert len(sampled) == ceil(len(source_rows) * multiplier)
    assert_frame_equal(result.iloc[: len(records)], original)
    assert_frame_equal(records, original)
    source_records = original_train.to_dict("records")
    assert all(
        row in source_records for row in result.iloc[len(records) :].to_dict("records")
    )


def test_sampling_without_strata_scales_all_training_rows():
    sampler = TrainingDatasetSampler(
        group_columns=("source_group_id",),
        stratify_columns=(),
        oversampling_rule=constant_rule(1.5),
    )

    result = sampler.sample({"Mus musculus": make_split_records()})["Mus musculus"]

    assert (result["dataset_split"] == "train").sum() == 9
    assert_frame_equal(
        non_training_rows(result), non_training_rows(make_split_records())
    )


def test_rule_sees_all_species_and_sets_each_species_multiplier():
    records_by_species = {
        "Mus musculus": make_split_records("mouse"),
        "Rattus rattus": make_split_records("rat"),
    }
    contexts = []

    def rule(species, _frame, all_frames):
        contexts.append((species, set(all_frames)))
        return 2.0 if species == "Mus musculus" else 1.0

    result = make_sampler(rule).sample(records_by_species)

    assert contexts == [
        ("Mus musculus", {"Mus musculus", "Rattus rattus"}),
        ("Rattus rattus", {"Mus musculus", "Rattus rattus"}),
    ]
    assert (result["Mus musculus"]["dataset_split"] == "train").sum() == 12
    assert_frame_equal(result["Rattus rattus"], records_by_species["Rattus rattus"])


def test_same_seed_produces_identical_copies():
    records = {"Mus musculus": make_split_records()}

    first = make_sampler(constant_rule(4.0), rng_seed=19).sample(records)
    repeated = make_sampler(constant_rule(4.0), rng_seed=19).sample(records)

    assert_frame_equal(first["Mus musculus"], repeated["Mus musculus"])


def _invalid_sample_records(kind):
    """Build split metadata that the sampler must reject."""
    records = make_split_records()
    if kind == "not_a_dataframe":
        return []
    elif kind == "missing_split":
        return records.drop(columns="dataset_split")
    elif kind == "unknown_split":
        records.loc[records.index[0], "dataset_split"] = "development"
    elif kind == "null_training_group":
        records["source_group_id"] = records["source_group_id"].astype("Int64")
        records.iloc[0, 0] = pd.NA
    elif kind == "conflicting_training_strata":
        records.loc[records["source_group_id"] == 10, "country_code"] = [
            "DE",
            "FR",
            "DE",
        ]
    elif kind == "no_training_rows":
        records["dataset_split"] = "validation"
    else:
        raise AssertionError(f"Unknown case: {kind}")
    return records


@pytest.mark.parametrize(
    "kind",
    [
        "not_a_dataframe",
        "missing_split",
        "unknown_split",
        "null_training_group",
        "conflicting_training_strata",
        "no_training_rows",
    ],
)
def test_unusable_training_metadata_is_rejected(kind):
    with pytest.raises(ValueError):
        make_sampler(constant_rule(2.0)).sample(
            {"Mus musculus": _invalid_sample_records(kind)}
        )


@pytest.mark.parametrize("multiplier", [0.5, float("nan")])
def test_a_rule_cannot_shrink_or_corrupt_the_training_rows(multiplier):
    records = make_split_records()
    original = records.copy(deep=True)

    with pytest.raises(ValueError, match="Mus musculus"):
        make_sampler(constant_rule(multiplier)).sample({"Mus musculus": records})

    assert_frame_equal(records, original)


def test_sampler_from_config_resolves_its_rule_reference(tmp_path, monkeypatch):
    (tmp_path / "sampler_rules.py").write_text(
        "def double_mice(species, frame, all_frames):\n"
        "    return 2.0 if species == 'Mus musculus' else 1.0\n"
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
                        "oversampling_rule": "sampler_rules:double_mice",
                    }
                }
            }
        )
    )

    sampler = TrainingDatasetSampler.from_config(config_path)
    result = sampler.sample({"Mus musculus": make_split_records()})["Mus musculus"]

    assert (result["dataset_split"] == "train").sum() == 12
