"""Split and oversample downloader metadata without accessing images."""

from collections.abc import Callable, Mapping
from math import ceil
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import AfterValidator, Field, ImportString, ValidationError, validate_call

from .base import Configurable
from .config_loading import load_section
from .training_validation import (
    ColumnNames,
    FiniteNumber,
    NonnegativeNumber,
    Seed,
    parse_settings,
)


def _sums_to_one(fractions):
    """Require split fractions that describe the whole dataset."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    return fractions


type GroupColumns = Annotated[ColumnNames, Field(min_length=1)]
type SplitFractions = Annotated[
    tuple[NonnegativeNumber, NonnegativeNumber, NonnegativeNumber],
    AfterValidator(_sums_to_one),
]
type Multiplier = Annotated[FiniteNumber, Field(ge=1)]
# A callable from Python, or a "module:function" reference from YAML.
type OversamplingRule = ImportString[
    Callable[[str, pd.DataFrame, Mapping[str, pd.DataFrame]], float]
]


def _require_columns(species: str, frame: pd.DataFrame, columns) -> None:
    """Reject frames whose required metadata columns are ambiguous or absent."""
    if not frame.columns.is_unique:
        raise ValueError(f"{species}: dataframe column names must be unique")
    missing = [name for name in dict.fromkeys(columns) if name not in frame.columns]
    if missing:
        raise ValueError(f"{species}: missing columns {missing}")


class DatasetSplitter(Configurable):
    """Partition image metadata while keeping observation groups indivisible.

    Fractions target row counts within each species/stratum. A largest-first
    greedy allocation approximates them; leakage prevention and three nonempty
    species splits take precedence over exact proportions. No images are read.
    """

    @validate_call
    def __init__(
        self,
        group_columns: GroupColumns,
        stratify_columns: ColumnNames = (),
        train_val_test_split: SplitFractions = (0.7, 0.2, 0.1),
        rng_seed: Seed = 42,
    ):
        """Configure group-aware partitioning.

        Args:
            group_columns: Composite observation key, e.g. ``("id",)`` for
                iNaturalist and LILA records.
            stratify_columns: Columns defining strata within each species.
            train_val_test_split: Finite nonnegative train/validation/test row
                fractions summing to one. Nonempty splits take precedence even
                when a target fraction is zero.
            rng_seed: Nonnegative integer seed for reproducible tie-breaking.

        Raises:
            ValueError: If column names, fractions, or the seed are invalid.
        """
        self.group_columns = group_columns
        self.stratify_columns = stratify_columns
        self.fractions = np.asarray(train_val_test_split, dtype=float)
        self.rng_seed = rng_seed

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DatasetSplitter":
        """Build a splitter from the ``data.dataset_splitter`` YAML mapping.

        Args:
            config_path: YAML file containing a ``data.dataset_splitter`` mapping.

        Returns:
            A configured metadata splitter.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            ValueError: If the section is missing or its settings are invalid.
        """
        return cls(**load_section(config_path, "data", "dataset_splitter"))

    def split(
        self, records_by_species: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return copies annotated with ``train``, ``validation``, or ``test``.

        Preserves original columns, indices and row order without mutating inputs.
        Assignments are repeatable and independent of row/species ordering. Empty
        frames remain empty. Sparse strata may lack individual splits, but each
        nonempty species receives all three splits.

        Args:
            records_by_species: Species names mapped to downloader metadata.

        Returns:
            Per-species dataframes with an added ``dataset_split`` column.

        Raises:
            ValueError: For missing/null group or stratum values, conflicting
                strata within a group, existing dataset_split columns, duplicate
                column names, or fewer than three groups in a nonempty species.
        """
        result = {}
        # Reinitialize per call so previous calls cannot change assignments.
        rng = np.random.default_rng(self.rng_seed)
        for species in sorted(records_by_species):
            frame = records_by_species[species]
            self._validate_frame(species, frame)
            annotated = frame.copy(deep=True)
            if frame.empty:
                annotated["dataset_split"] = pd.Series(index=frame.index, dtype="str")
            else:
                annotated["dataset_split"] = self._assign_rows(frame, rng)
            result[species] = annotated
        return result

    def _validate_frame(self, species: str, frame: pd.DataFrame) -> None:
        """Reject ambiguous metadata before allocating observation groups."""
        if "dataset_split" in frame.columns:
            raise ValueError(f"{species}: dataset_split already exists")
        columns = list(dict.fromkeys((*self.group_columns, *self.stratify_columns)))
        _require_columns(species, frame, columns)
        if frame[columns].isna().any().any():
            raise ValueError(
                f"{species}: grouping and stratification values must not be null"
            )
        if frame.empty:
            return
        groups = frame.groupby(list(self.group_columns), sort=True, observed=True)
        if groups.ngroups < 3:
            raise ValueError(
                f"{species}: at least three independent groups are required"
            )
        if self.stratify_columns:
            conflicts = groups[list(self.stratify_columns)].nunique().gt(1)
            if conflicts.any().any():
                raise ValueError(
                    f"{species}: an observation belongs to conflicting strata"
                )

    def _assign_strata(self, frame: pd.DataFrame) -> dict[tuple, list[np.ndarray]]:
        """Organize indivisible observation groups by stratum using row positions.

        Expects validated metadata: each observation has one stratum. Every input
        row position belongs to exactly one returned group, regardless of whether
        dataframe index labels are duplicated.
        """
        # .indices maps each observation key to an array of row POSITIONS, not
        # dataframe index labels: e.g. {observation_id: array([0, 3, 5])}.
        # This matters because index labels can be duplicated. sort=True orders
        # the keys consistently, so shuffling rows does not change the order in
        # which groups consume random numbers. observed=True excludes unused
        # category values when grouping columns have pandas categorical dtype.
        group_indices = frame.groupby(
            list(self.group_columns), sort=True, observed=True
        ).indices
        # Each stratum key (e.g. ("DE", 2024)) maps to a list of observation
        # groups, each represented by its row-position array. Without configured
        # stratification columns, the key is () and all groups share one stratum.
        strata: dict[tuple, list[np.ndarray]] = {}
        for positions in group_indices.values():
            # .iloc selects by position. The first row identifies the stratum
            # because validation already checked that the whole group agrees.
            key = tuple(
                frame.iloc[int(positions[0])][name] for name in self.stratify_columns
            )
            # setdefault creates an empty list only if this key is new, then
            # returns the stored list so we can append the observation group.
            strata.setdefault(key, []).append(positions)
        return strata

    def _assign_rows(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Allocate positional indices so duplicate dataframe indices are safe."""
        strata = self._assign_strata(frame)
        # Split numbers 0, 1, 2 correspond to these three names throughout.
        names = np.array(["train", "validation", "test"])
        # Allocate one output slot per input row. np.empty is uninitialized;
        # every slot is filled below when its observation group is assigned.
        assignments = np.empty(len(frame), dtype=object)
        # "Global" here means the entire species, across all its strata.
        global_counts = np.zeros(3, dtype=int)
        remaining_groups = sum(len(groups) for groups in strata.values())
        for groups in strata.values():
            # permutation(n) produces a seeded shuffle of integers 0 through
            # n-1. Use those integers to reorder groups, then sort largest first.
            # Python's sort is stable: equal-sized groups keep their shuffled
            # relative order. Large groups go first because they are hardest to
            # fit into the remaining row-count targets without overshooting.
            ordered = [groups[int(i)] for i in rng.permutation(len(groups))]
            ordered.sort(key=len, reverse=True)
            # Multiply each fraction by this stratum's total IMAGE row count,
            # not its observation count. counts tracks actual allocated rows.
            targets = self.fractions * sum(len(positions) for positions in ordered)
            counts = np.zeros(3, dtype=int)
            # enumerate gives both the loop position and the observation group.
            for offset, positions in enumerate(ordered):
                # flatnonzero returns positions where a boolean array is True:
                # counts [4, 0, 0] gives empty split numbers [1, 2].
                global_empty = np.flatnonzero(global_counts == 0)
                local_empty = np.flatnonzero(counts == 0)
                if remaining_groups == len(global_empty):
                    # Each remaining group must fill a species-level empty split;
                    # assigning elsewhere would make three nonempty splits impossible.
                    candidates = global_empty
                elif len(groups) >= 3 and len(ordered) - offset == len(local_empty):
                    # This stratum has enough groups for all three splits. When
                    # groups remaining equals empty splits, reserve one for each.
                    candidates = local_empty
                elif len(groups) < 3 and len(global_empty) > 0:
                    # A sparse stratum cannot fill all three splits itself, so
                    # prefer helping populate missing species-level splits.
                    candidates = global_empty
                else:
                    # No reservation needed: arange(3) means all split numbers
                    # [0, 1, 2] are eligible for the row-count allocation.
                    candidates = np.arange(3)
                # Array indexing selects just the eligible splits. Positive
                # deficits mean rows are still needed; negative means overshoot.
                deficits = targets[candidates] - counts[candidates]
                # Boolean indexing retains every candidate tied for the largest
                # deficit. choice then breaks that tie using the seeded RNG.
                best = candidates[deficits == deficits.max()]
                destination = int(rng.choice(best))
                # Advanced indexing writes the same split name to ALL image
                # rows in this observation, preventing cross-split leakage.
                assignments[positions] = names[destination]
                counts[destination] += len(positions)
                global_counts[destination] += len(positions)
                # Row totals increase by group size; groups remaining drops by one.
                remaining_groups -= 1
        return assignments


class TrainingDatasetSampler(Configurable):
    """Oversample training metadata after a leakage-safe dataset split.

    Call :class:`DatasetSplitter` first, then pass its annotated dataframes
    to :meth:`sample`. This sampler duplicates only ``dataset_split == "train"``
    rows, separately within each configured stratum. Validation and test rows
    are retained exactly, so they continue to represent the unmodified dataset.

    Do not sample before splitting. Although group-aware splitting would keep a
    copied image with its source observation, duplicates would alter group row
    weights during allocation and may be assigned to validation or test. That
    distorts evaluation distributions and weakens the splitter's proportional
    allocation guarantee.
    """

    @validate_call
    def __init__(
        self,
        group_columns: GroupColumns,
        stratify_columns: ColumnNames,
        oversampling_rule: OversamplingRule,
        rng_seed: Seed = 42,
    ):
        """Configure reproducible, stratified train-only oversampling.

        Args:
            group_columns: Composite observation key. A training image is sampled
                by first choosing one of these groups, then one of its image rows.
            stratify_columns: Columns defining strata. The sampler applies the
                multiplier separately to each training stratum, retaining their
                proportions apart from integer rounding.
            oversampling_rule: Callable, or ``module:function`` reference to one,
                receiving a species name, that species' full split dataframe, and
                all species frames. It returns a finite multiplier of at least
                ``1.0`` for that species' training rows.
            rng_seed: Nonnegative integer seed for reproducible sampling.

        Raises:
            ValueError: If the columns, rule or seed are invalid.
        """
        self.group_columns = group_columns
        self.stratify_columns = stratify_columns
        self.oversampling_rule = oversampling_rule
        self.rng_seed = rng_seed

    @classmethod
    def from_config(cls, config_path: str | Path) -> "TrainingDatasetSampler":
        """Build a sampler from ``data.training_dataset_sampler`` YAML settings.

        The required ``oversampling_rule`` value uses ``module:function`` syntax.
        The resolved callable receives the same species, dataframe, and complete
        mapping arguments as a directly supplied rule.

        Args:
            config_path: YAML file containing sampler settings and rule reference.

        Returns:
            A configured train-only dataset sampler.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            ValueError: If the section is missing, the rule reference cannot be
                resolved, or the settings are invalid.
        """
        return cls(**load_section(config_path, "data", "training_dataset_sampler"))

    def sample(
        self, records_by_species: Mapping[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return split dataframes with additional, stratified training rows.

        Every original row remains in the same order and retains its index. New
        rows are appended and are exact copies of original training rows. For a
        multiplier ``m``, each training stratum receives
        ``ceil(m * original_stratum_rows)`` total rows. The sampling sequence is
        seeded: it chooses an observation uniformly, then an image uniformly
        from that observation. Validation, rule resolution, and sampling process
        species in sorted order, and rules receive a species-sorted population
        mapping. Results are independent of input species-key order, while the
        returned mapping retains that order. Input row order is preserved, not
        normalized: row-order independence is not guaranteed.

        This method must follow :class:`DatasetSplitter`; it relies on its
        ``dataset_split`` column to isolate training rows. Applying it first can
        cause copied rows to enter validation/test when the later splitter assigns
        their observation, changing the evaluation population.

        Args:
            records_by_species: Per-species dataframe output from the splitter.

        Returns:
            New per-species frames with original validation/test rows unchanged
            and copies appended only to their training portion.

        Raises:
            ValueError: If metadata is invalid, a rule result is not a finite
                multiplier of at least one, or a multiplier above one is requested
                for a species with no training rows.
        """
        # Materialize the mapping once: rules receive the complete, original
        # population and none of their inputs are modified by this sampler.
        frames = dict(records_by_species)
        species_order = tuple(frames)
        # Canonicalize both rule context and RNG consumption, independently of
        # caller insertion order. Keep the original keys for output presentation.
        frames = {species: frames[species] for species in sorted(frames)}
        # Validate all frames before calling a rule or sampling. A malformed
        # later species must not leave callers with a partially built result.
        for species, frame in frames.items():
            self._validate_frame(species, frame)

        # Resolve every multiplier before generating copies for the same reason:
        # rule failures are reported before sampling begins.
        multipliers = {
            species: self._validate_multiplier(
                species, self.oversampling_rule(species, frame, frames)
            )
            for species, frame in frames.items()
        }
        # A shared seeded generator consumes randomness in canonical species
        # order; output key order must not influence which copies are selected.
        rng = np.random.default_rng(self.rng_seed)
        sampled = {
            species: self._sample_species(frame, multipliers[species], rng)
            for species, frame in frames.items()
        }
        return {species: sampled[species] for species in species_order}

    def _validate_frame(self, species: str, frame: pd.DataFrame) -> None:
        """Validate fields needed to select coherent training observations."""
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{species}: records must be a pandas DataFrame")
        _require_columns(
            species,
            frame,
            ["dataset_split", *self.group_columns, *self.stratify_columns],
        )
        valid_splits = {"train", "validation", "test"}
        if not frame["dataset_split"].isin(valid_splits).all():
            raise ValueError(f"{species}: dataset_split contains an invalid value")

        # Evaluation rows are never candidates for copying, so null group or
        # stratum metadata there is irrelevant to sampling. Training rows need
        # complete metadata to select observations and retain strata safely.
        training = frame.loc[frame["dataset_split"] == "train"]
        metadata_columns = list(
            dict.fromkeys((*self.group_columns, *self.stratify_columns))
        )
        if training[metadata_columns].isna().any().any():
            raise ValueError(
                f"{species}: training group and stratum values must not be null"
            )
        if training.empty or not self.stratify_columns:
            return
        groups = training.groupby(list(self.group_columns), sort=True, observed=True)
        conflicts = groups[list(self.stratify_columns)].nunique().gt(1)
        if conflicts.any().any():
            raise ValueError(
                f"{species}: a training observation belongs to conflicting strata"
            )

    @staticmethod
    def _validate_multiplier(species: str, multiplier: object) -> float:
        """Return a finite multiplier that cannot remove training rows."""
        try:
            return parse_settings(Multiplier, multiplier)
        except ValidationError as error:
            # Name the species: the rule's result, not a setting, is at fault.
            raise ValueError(
                f"{species}: oversampling rule must return a finite multiplier >= 1.0"
            ) from error

    def _sample_species(
        self, frame: pd.DataFrame, multiplier: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Append copies selected within each training stratum."""
        if multiplier == 1.0:
            # A multiplier of one is a strict no-op: do not consume random
            # numbers or alter row/index ordering, but still return a new frame.
            return frame.copy(deep=True)

        # Start from the already split training subset. Validation/test rows stay
        # only in ``frame`` and are never included in a candidate stratum.
        training = frame.loc[frame["dataset_split"] == "train"]
        if training.empty:
            raise ValueError("Cannot oversample a species without training rows")

        if self.stratify_columns:
            # Apply the multiplier independently per stratum. This retains the
            # training strata's proportions, except for unavoidable ceil rounding.
            strata = [
                stratum
                for _, stratum in training.groupby(
                    list(self.stratify_columns), sort=True, observed=True
                )
            ]
        else:
            strata = [training]

        copies = []
        for stratum in strata:
            # ceil ensures the requested multiplier is reached rather than
            # rounded below it when the target cannot be an integral row count.
            target_count = ceil(multiplier * len(stratum))
            copies.extend(
                self._sample_stratum(stratum, target_count - len(stratum), rng)
            )
        # Keep every original row first, in its input order and with its index.
        # Appending copies makes the duplicated training population observable
        # without adding provenance columns to downloader metadata.
        return pd.concat([frame, *copies])

    def _sample_stratum(
        self, stratum: pd.DataFrame, copy_count: int, rng: np.random.Generator
    ) -> list[pd.DataFrame]:
        """Select complete image rows by uniformly sampling groups then images."""
        # Each value is an array of positional row locations for one observation.
        # Sampling groups uniformly prevents an observation with many photos from
        # being selected more often merely because it has more image rows.
        group_indices = stratum.groupby(
            list(self.group_columns), sort=True, observed=True
        ).indices
        groups = list(group_indices.values())
        copies = []
        for _ in range(copy_count):
            # First choose an observation, then an image within it. ``iloc`` uses
            # positions, so duplicate dataframe index labels remain safe. Copying
            # the complete row preserves the observation-to-photo metadata link.
            positions = groups[int(rng.integers(len(groups)))]
            position = int(positions[rng.integers(len(positions))])
            copies.append(stratum.iloc[[position]].copy(deep=True))
        return copies
