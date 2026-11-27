"""Resolve validation scorers and preserve published metric definitions."""

from collections.abc import Mapping
from copy import deepcopy
from functools import partial

import numpy as np
from sklearn import metrics

from smartrodent.components import ComponentSpec


class ValidationMetrics:
    """Prepare classification metrics chosen by alias and component spec.

    Args:
        specs: Scorers accepting arrays of true and predicted class IDs.

    Raises:
        ValueError: A configured scorer cannot produce a scalar score.
    """

    def __init__(self, specs: Mapping[str, ComponentSpec]):
        """Check metric availability and options before training starts."""
        self._specs = deepcopy(dict(specs))
        self._scorers = {}
        sample = np.array([0, 1])
        for alias, spec in self._specs.items():
            if "function" in spec.parameters:
                raise ValueError(
                    f"Metric '{alias}' uses the reserved 'function' identity key"
                )
            try:
                scorer = partial(spec.resolve(metrics), **spec.parameters)
                float(scorer(sample, sample))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Metric '{alias}' cannot score class IDs: {error}"
                ) from error
            self._scorers[alias] = scorer

    def score(self, targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        """Score all predictions over a complete validation split.

        Args:
            targets: True class IDs for the split.
            predictions: Predicted class IDs for the same samples.

        Returns:
            Scalar scores keyed by their configured aliases.
        """
        return {
            alias: float(scorer(targets, predictions))
            for alias, scorer in self._scorers.items()
        }

    def definition(self, alias: str) -> dict:
        """Return the original registry identity format for a metric.

        Args:
            alias: Name by which training reports the score.

        Returns:
            The scorer function name and its explicit options in one mapping.
        """
        spec = self._specs[alias]
        return {"function": spec.name, **deepcopy(spec.parameters)}
