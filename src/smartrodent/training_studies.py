"""Optuna studies over the SupCon and classifier trainers' configurations.

A study document is an ordinary stage configuration in which some values are
``!suggest-float``, ``!suggest-int`` or ``!suggest-categorical`` tags, plus a
``tuning:`` section with the study settings. Each trial resolves the tags to one
concrete configuration, trains it with the existing trainer in its own
``trial_NNNN`` directory, and is ranked by the trainer's configured tracked metric.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import mlflow

from smartrodent import config_utils
from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_training import (
    ClassifierConfiguration,
    ClassifierExperiment,
    FreezeConfigurationError,
)
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import validate_stopping
from smartrodent.hyperparameter_search import (
    EpochCallback,
    HyperparameterStudy,
    StudyConfiguration,
    StudyObjective,
    StudyResult,
    TrialConfigurationError,
)
from smartrodent.search_space import SearchSpace
from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration


def is_study_configuration(path: str | Path) -> bool:
    """Return whether a configuration file defines an Optuna study (a tuning section).

    Args:
        path: Configuration file.

    Returns:
        True when the document is a mapping with a ``tuning`` key.
    """
    document = config_utils.load_yaml(path)
    return isinstance(document, Mapping) and "tuning" in document


class _StageStudy(ABC):
    """Shared flow of a study over one trainer's standalone configuration.

    The document is prevalidated on construction; ``run`` trains the trials.

    Args:
        document: Stage configuration with ``!suggest-*`` tags and a tuning section.
        prefix: Dotted path of the stage inside a pipeline document, for naming.
    """

    # Settings that define what trials are compared on, or where they read and write.
    protected: ClassVar[tuple[str, ...]] = (
        "tracked_metric",
        "tracked_mode",
        "validation_metrics",
        "train_dataset",
        "validation_dataset",
        "class_names",
        "output_directory",
        "mlflow",
    )

    # Errors raised while preparing training (not by the configuration classes) that
    # mean an invalid combination of sampled values rather than a broken study.
    _combination_errors: ClassVar[tuple[type[Exception], ...]] = ()

    def __init__(self, document: Mapping, *, prefix: str = ""):
        """Parse the study settings and build every candidate configuration."""
        if not isinstance(document, Mapping):
            raise ValueError("A study configuration must be a mapping")
        values = deepcopy(dict(document))
        if "tuning" not in values:
            raise ValueError("An Optuna study requires a tuning section")
        self.settings = StudyConfiguration.from_mapping(values.pop("tuning"))
        # The space first: it rejects a tag on output_directory with a clear message.
        self.space = SearchSpace.from_document(
            values, protected=self.protected, prefix=prefix
        )
        if values.get("output_directory") is None:
            raise ValueError("An Optuna study requires an output_directory")
        self.output_directory = Path(values["output_directory"])
        reference_directory = self.output_directory / "trial_0000"
        self.configurations = tuple(
            self.space.validate_candidates(
                partial(self._configuration, directory=reference_directory)
            )
        )
        self.objective = self._objective(self.configurations[0])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "_StageStudy":
        """Load and prevalidate a study configuration file.

        Args:
            path: YAML file; paths inside are relative to the working directory.

        Returns:
            The prepared study.
        """
        return cls(config_utils.load_yaml(path))

    @property
    def reference(self):
        """Return the reference configuration (all parameters at their first value)."""
        return self.configurations[0]

    def run(self, *, parent_run_id: str | None = None) -> StudyResult:
        """Run the study, tracking trials under one parent run when MLflow is configured.

        Args:
            parent_run_id: Active pipeline run to parent trials to; when None and
                tracking is enabled, the study opens its own parent run.

        Returns:
            The study report and winner.
        """
        tracking = self.reference.mlflow
        if tracking is None or parent_run_id is not None:
            return self._finish(self._study(parent_run_id, self._identity()).run())
        with tracking.start_run(self.output_directory.name) as run:
            mlflow.set_tags(tracking.run_tags())
            try:
                result = self._study(run.info.run_id, self._identity()).run()
            finally:
                report = self.output_directory / "study.yaml"
                if report.is_file():
                    mlflow.log_artifact(str(report))
            mlflow.log_metric(
                f"best_validation_{self.objective.tracked_metric}",
                result.selected.best_value,
            )
            return self._finish(result)

    def preflight(self) -> None:
        """Check output and storage conflicts without training; see HyperparameterStudy.preflight."""
        self._study(None, {}).preflight()

    def _study(
        self, parent_run_id: str | None, identity: Mapping
    ) -> HyperparameterStudy:
        """Bind trial execution to the run that parents this invocation's trials."""
        return HyperparameterStudy(
            self.space,
            self.settings,
            partial(self._run_trial, parent_run_id=parent_run_id),
            output_directory=self.output_directory,
            objective=self.objective,
            identity=identity,
        )

    def _identity(self) -> dict[str, Any]:
        """Return stage inputs a resumed study must share besides space and objective."""
        return {}

    def _run_trial(
        self,
        document: dict,
        directory: Path,
        epoch_callback: EpochCallback,
        *,
        parent_run_id: str | None,
    ) -> ExperimentResult:
        """Validate one sampled document and train it."""
        # TypeError is anticipated, as in classifier sweeps: configuration
        # constructors report some invalid inputs that way.
        try:
            configuration = self._configuration(document, directory=directory)
        except (ValueError, TypeError) as error:
            raise TrialConfigurationError(
                f"Sampled configuration is invalid: {error}"
            ) from error
        try:
            return self._train(
                configuration,
                parent_run_id=parent_run_id,
                epoch_callback=epoch_callback,
            )
        except self._combination_errors as error:
            raise TrialConfigurationError(
                f"Sampled combination does not fit: {error}"
            ) from error

    def _finish(self, result: StudyResult) -> StudyResult:
        """Complete stage-specific work after all trials; nothing by default."""
        return result

    @abstractmethod
    def _configuration(self, document: dict, *, directory: Path):
        """Build one validated stage configuration writing to the given directory."""

    @abstractmethod
    def _objective(self, configuration) -> StudyObjective:
        """Return the ranking defined by a stage configuration."""

    @abstractmethod
    def _train(
        self, configuration, *, parent_run_id: str | None, epoch_callback: EpochCallback
    ) -> ExperimentResult:
        """Train one configuration with the stage's trainer."""


class SupConStudy(_StageStudy):
    """Optuna study over a standalone SupCon configuration.

    Args:
        document: SupCon configuration with ``!suggest-*`` tags and a tuning section.
    """

    # Loss values at different temperatures are not comparable (the sweep rule).
    protected: ClassVar[tuple[str, ...]] = (*_StageStudy.protected, "temperature")

    def _configuration(
        self, document: dict, *, directory: Path
    ) -> TrainingConfiguration:
        """Build a SupCon configuration whose stopping settings agree with the ranking."""
        configuration = TrainingConfiguration.from_mapping(
            {**document, "output_directory": directory}
        )
        validate_stopping(configuration, configuration.effective_tracked_mode)
        return configuration

    def _objective(self, configuration: TrainingConfiguration) -> StudyObjective:
        """Rank by the effective tracked metric, as SupCon sweeps do."""
        return StudyObjective(
            tracked_metric=configuration.effective_tracked_metric,
            tracked_mode=configuration.effective_tracked_mode,
            validation_dataset=configuration.validation_dataset,
        )

    def _train(
        self,
        configuration: TrainingConfiguration,
        *,
        parent_run_id: str | None,
        epoch_callback: EpochCallback,
    ) -> ExperimentResult:
        """Train one SupCon trial."""
        return SupConExperiment(configuration).run_with_result(
            parent_run_id=parent_run_id, epoch_callback=epoch_callback
        )


class ClassifierStudy(_StageStudy):
    """Optuna study over a standalone classifier configuration.

    As in classifier sweeps, every completed trial registers its model when a
    registry is configured; only the study winner is compared for promotion.

    Args:
        document: Classifier configuration with ``!suggest-*`` tags and a tuning section.
        encoder_provenance: Optional selected-SupCon metadata from the pipeline.
        prefix: Dotted path of the stage inside a pipeline document, for naming.
    """

    # Freeze validity depends on the encoder network, which a trial only loads when
    # training starts; e.g. a high freeze combined with a full head replacement.
    _combination_errors: ClassVar[tuple[type[Exception], ...]] = (
        FreezeConfigurationError,
    )

    def __init__(
        self,
        document: Mapping,
        *,
        encoder_provenance: Mapping | None = None,
        prefix: str = "",
    ):
        """Prevalidate the study and keep resources shared by all trials."""
        super().__init__(document, prefix=prefix)
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        self._head_cache = {}
        tracking = self.reference.mlflow
        self._registered = (
            tracking is not None and tracking.registered_model_name is not None
        )

    def _identity(self) -> dict[str, Any]:
        """Trials trained on different encoder bytes are not comparable.

        Every candidate is included, so a searched encoder choice is covered too.
        """
        checkpoints = {
            Path(configuration.initialization["encoder"]["checkpoint"]).resolve()
            for configuration in self.configurations
        }
        # Sorted by content, not path: moving an unchanged file keeps the identity.
        return {"encoder checkpoints": sorted(map(self._digest, checkpoints))}

    @staticmethod
    def _digest(path: Path) -> str:
        """Return a file's SHA-256."""
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def _configuration(
        self, document: dict, *, directory: Path
    ) -> ClassifierConfiguration:
        """Build one classifier configuration."""
        return ClassifierConfiguration.from_mapping(
            {**document, "output_directory": directory}
        )

    def _objective(self, configuration: ClassifierConfiguration) -> StudyObjective:
        """Rank by the configured classifier metric."""
        return StudyObjective(
            tracked_metric=configuration.tracked_metric,
            tracked_mode=configuration.effective_tracked_mode,
            validation_dataset=configuration.validation_dataset,
        )

    def _train(
        self,
        configuration: ClassifierConfiguration,
        *,
        parent_run_id: str | None,
        epoch_callback: EpochCallback,
    ) -> ExperimentResult:
        """Train one classifier trial, deferring promotion to the end of the study."""
        experiment = ClassifierExperiment(
            configuration,
            encoder_provenance=self.encoder_provenance,
            head_cache=self._head_cache,
        )
        return experiment.run_with_result(
            parent_run_id=parent_run_id,
            defer_promotion=self._registered,
            epoch_callback=epoch_callback,
        )

    def _finish(self, result: StudyResult) -> StudyResult:
        """Compare only the study winner against the registered champion."""
        if not self._registered:
            return result
        else:
            reference = self.reference
            registry = ClassifierChampionRegistry(
                reference.mlflow,
                reference.tracked_metric_definition,
                reference.class_names,
            )
            return replace(
                result, registered_champion_version=registry.promote(result.selected)
            )
