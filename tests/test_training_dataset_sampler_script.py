"""Behavioral contract for the checked-in training oversampling rule."""

from copy import deepcopy

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

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
