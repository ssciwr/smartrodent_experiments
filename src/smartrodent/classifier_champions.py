"""Compare registered classifier champions using explicit validation provenance."""

import json
from collections.abc import Mapping, Sequence

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.training_validation import FiniteNumber, parse_settings


class ClassifierChampionRegistry:
    """Promote a classifier only when its validation result improves the incumbent.

    Args:
        configuration: Registry identity, alias, and optional dataset revision.
        metric_definition: Exact scoring function and configured options.
        class_names: Evaluated species population, canonically ordered by label.
        client: Optional client for the configured registry.
    """

    def __init__(
        self,
        configuration: MlflowConfiguration,
        metric_definition: Mapping,
        class_names: Sequence[str],
        client: MlflowClient | None = None,
    ):
        """Keep comparison separate from model registration and artifact publication."""
        self.configuration = configuration
        # Stable JSON preserves function/options across processes and invocations;
        # a shared display name must not make different scoring functions comparable.
        self.metric_definition = json.dumps(
            dict(metric_definition), sort_keys=True, separators=(",", ":")
        )
        self.class_names = json.dumps(sorted(class_names), separators=(",", ":"))
        self.client = (
            client
            if client is not None
            else MlflowClient(tracking_uri=configuration.tracking_uri)
        )

    def comparison_tags(self, result: ExperimentResult) -> dict:
        """Describe the original score and validation identity in portable registry tags.

        Args:
            result: Completed classifier's best-epoch result.

        Returns:
            Comparison metadata, excluding absent validation paths and revisions.
        """
        tags = {
            "tracked_metric": result.tracked_metric,
            "tracked_mode": result.tracked_mode,
            "tracked_metric_definition": self.metric_definition,
            "class_names": self.class_names,
            "best_value": result.best_value,
        }
        if result.validation_dataset is not None:
            tags["validation_dataset"] = str(result.validation_dataset.resolve())
        if self.configuration.dataset_revision is not None:
            tags["dataset_revision"] = self.configuration.dataset_revision
        return tags

    def promote(self, result: ExperimentResult) -> str | None:
        """Compare and optionally move the champion alias after successful publication.

        Args:
            result: Validated winner with its already-published registry version.

        Returns:
            The retained or newly promoted version; None for training-only runs.

        Raises:
            ValueError: The incumbent is not safely comparable, or no registered
                version identifies the candidate. Choose another alias/model or
                explicitly resolve historical metadata rather than overwriting it.
            MlflowException: Registry errors other than an absent model/alias.
        """
        if result.validation_dataset is None:
            return None
        config = self.configuration
        if (
            config.registered_model_name is None
            or result.registered_model_version is None
        ):
            raise ValueError(
                "Champion promotion requires a registered classifier version"
            )
        try:
            incumbent = self.client.get_model_version_by_alias(
                config.registered_model_name, config.champion_alias
            )
        except MlflowException as error:
            # MLflow's SQLite registry reports an absent alias as an invalid
            # parameter rather than RESOURCE_DOES_NOT_EXIST. Match that exact
            # response; other parameter/backend errors must still propagate.
            missing_alias = (
                error.error_code == "INVALID_PARAMETER_VALUE"
                and error.message
                == f"Registered model alias {config.champion_alias} not found."
            )
            if error.error_code != "RESOURCE_DOES_NOT_EXIST" and not missing_alias:
                raise
            incumbent = None
        if incumbent is not None:
            score = self._incumbent_score(incumbent.tags, result)
            improved = (
                result.best_value < score
                if result.tracked_mode == "min"
                else result.best_value > score
            )
            if not improved:
                return str(incumbent.version)
        self.client.set_registered_model_alias(
            config.registered_model_name,
            config.champion_alias,
            result.registered_model_version,
        )
        return result.registered_model_version

    def _incumbent_score(self, tags: dict, candidate: ExperimentResult) -> float:
        """Fail closed for legacy, unvalidated, or incompatible historical metadata."""
        expected = self.comparison_tags(candidate)
        # Compare the published identity fields, not the score being ranked.
        # An unspecified revision must still reject a specified historical one.
        del expected["best_value"]
        expected["validation_status"] = "passed"
        expected["dataset_revision"] = self.configuration.dataset_revision
        if any(tags.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "Existing champion validation metadata is incompatible or missing; an explicit decision is required"
            )
        try:
            return parse_settings(FiniteNumber, float(tags["best_value"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Existing champion needs finite score metadata; an explicit decision is required"
            ) from error
