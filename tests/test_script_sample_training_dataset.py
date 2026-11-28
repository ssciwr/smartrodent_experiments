"""Entry-point behavior of scripts/sample_training_dataset.py.

Covers its checked-in oversampling rule and that sampled records keep portable
shared-root image paths.
"""

from copy import deepcopy

import pandas as pd
import pytest
from bundle_test_support import move_bundle, write_stage_config, write_stage_records
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from scripts import (
    sample_training_dataset,
    split_dataset,
)
from scripts.sample_training_dataset import compute_oversampling


def make_population(species_sizes):
    """Build metadata frames whose row counts represent species populations."""
    return {
        species: pd.DataFrame({"record": range(size)})
        for species, size in species_sizes.items()
    }


@pytest.mark.parametrize(
    "species_size,largest_size,expected_multiplier",
    [
        (1, 100, 10.0),
        (5, 100, 2.0),
        (10, 100, 1.0),
        (11, 100, 1.0),
        (100, 100, 1.0),
        (3, 100, 10.0 / 3.0),
        (1, 5, 1.0),
    ],
)
def test_rule_preserves_the_ten_percent_target_for_nonempty_species(
    species_size, largest_size, expected_multiplier
):
    """Rare species scale to the target; species at or above it are unchanged."""
    population = make_population({"mouse": species_size, "rat": largest_size})
    originals = deepcopy(population)

    multiplier = compute_oversampling("mouse", population["mouse"], population)

    assert multiplier == pytest.approx(expected_multiplier)
    for species, original in originals.items():
        assert_frame_equal(population[species], original)


def test_rule_returns_one_for_an_empty_species_among_nonempty_species():
    """An empty species remains empty rather than requesting invented samples."""
    population = make_population({"mouse": 0, "rat": 100})
    originals = deepcopy(population)

    multiplier = compute_oversampling("mouse", population["mouse"], population)

    assert multiplier == 1.0
    for species, original in originals.items():
        assert_frame_equal(population[species], original)


def test_rule_returns_one_when_every_species_is_empty():
    """An all-empty population needs no normalization or oversampling."""
    population = make_population({"mouse": 0, "rat": 0})
    originals = deepcopy(population)

    multipliers = {
        species: compute_oversampling(species, frame, population)
        for species, frame in population.items()
    }

    assert multipliers == {"mouse": 1.0, "rat": 1.0}
    for species, original in originals.items():
        assert_frame_equal(population[species], original)


def test_empty_species_do_not_change_nonempty_species_multipliers():
    """Adding an empty species must not affect the existing balancing target."""
    population = make_population({"mouse": 5, "rat": 100})
    with_empty_species = {**population, "shrew": population["mouse"].iloc[:0].copy()}

    for species, frame in population.items():
        assert compute_oversampling(
            species, frame, with_empty_species
        ) == pytest.approx(compute_oversampling(species, frame, population))


@given(
    species_sizes=st.lists(
        st.integers(min_value=0, max_value=300), min_size=1, max_size=5
    )
)
def test_generated_populations_have_safe_count_based_multipliers(species_sizes):
    """Empty and nonempty species obey the same count-based balancing contract."""
    population = make_population(
        {f"species_{index}": size for index, size in enumerate(species_sizes)}
    )
    reversed_population = dict(reversed(list(population.items())))
    largest_size = max(species_sizes)

    for species, frame in population.items():
        size = len(frame)
        if size == 0:
            expected = 1.0
        else:
            expected = max(1.0, (0.1 * largest_size) / size)

        multiplier = compute_oversampling(species, frame, population)
        reordered_multiplier = compute_oversampling(species, frame, reversed_population)

        assert multiplier == pytest.approx(expected)
        assert reordered_multiplier == pytest.approx(expected)


@pytest.mark.parametrize("absolute_input", [False, True])
def test_splitting_and_sampling_preserve_shared_root_paths_without_rebasing(
    bundle, tmp_path, absolute_input
):
    """Metadata-only stages keep the same original-image references verbatim."""
    dataset_root, records_by_species = bundle
    corrected_root = dataset_root / "stage5_corrected"
    for species, records in records_by_species.items():
        directory = corrected_root / "kept" / species
        directory.mkdir(parents=True)
        input_records = records.copy()
        if absolute_input:
            input_records["image_path"] = input_records["image_path"].map(
                lambda path: str(dataset_root / path)
            )
        input_records.to_csv(directory / "records_corrected.csv", index=False)
    split_root = dataset_root / "stage6_split"
    split_config = write_stage_config(
        tmp_path / "split.yaml",
        {
            "paths": {
                "dataset_splitter_input": str(corrected_root),
                "dataset_splitter_output": str(split_root),
                "records_glob": "kept/*/records_corrected.csv",
            },
            "data": {
                "dataset_splitter": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "train_val_test_split": [0.7, 0.15, 0.15],
                    "rng_seed": 42,
                }
            },
        },
    )
    sampled_root = dataset_root / "stage7_sampled"
    sample_config = write_stage_config(
        tmp_path / "sample.yaml",
        {
            "paths": {
                "training_dataset_sampler_input": str(split_root),
                "training_dataset_sampler_output": str(sampled_root),
            },
            "data": {
                "training_dataset_sampler": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "oversampling_rule": "scripts.sample_training_dataset:compute_oversampling",
                    "rng_seed": 42,
                }
            },
        },
    )

    split_dataset.main(split_config)
    sample_training_dataset.main(sample_config)

    moved_root = move_bundle(dataset_root)
    for stage in (split_root.name, sampled_root.name):
        for species, original in records_by_species.items():
            actual = pd.read_csv(moved_root / stage / species / "records.csv")
            assert actual["image_path"].tolist() == original["image_path"].tolist()
            assert set(actual["dataset_split"]) == {"train", "validation", "test"}
            assert all((moved_root / path).is_file() for path in actual["image_path"])


def test_sampling_writes_portable_paths_from_legacy_absolute_records(bundle, tmp_path):
    """Sampling also normalizes absolute inputs without altering their source CSVs."""
    dataset_root, records_by_species = bundle
    split_root = dataset_root / "stage6_split"
    input_records = {}
    for species, records in records_by_species.items():
        frame = records.copy()
        frame["dataset_split"] = ["train", "validation", "test"] * 2
        frame["image_path"] = frame["image_path"].map(
            lambda path: str(dataset_root / path)
        )
        input_records[species] = frame
    write_stage_records(split_root, input_records)
    sampled_root = dataset_root / "stage7_sampled"
    config = write_stage_config(
        tmp_path / "sample_legacy.yaml",
        {
            "paths": {
                "training_dataset_sampler_input": str(split_root),
                "training_dataset_sampler_output": str(sampled_root),
            },
            "data": {
                "training_dataset_sampler": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "oversampling_rule": "scripts.sample_training_dataset:compute_oversampling",
                }
            },
        },
    )

    sample_training_dataset.main(config)

    moved_root = move_bundle(dataset_root)
    for species, original in records_by_species.items():
        actual = pd.read_csv(moved_root / sampled_root.name / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all((moved_root / path).is_file() for path in actual["image_path"])
        pd.testing.assert_frame_equal(
            pd.read_csv(moved_root / split_root.name / species / "records.csv"),
            input_records[species],
        )
