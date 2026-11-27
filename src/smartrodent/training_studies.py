"""Optuna studies over the SupCon and classifier trainers' configurations.

A study document is an ordinary stage configuration in which some values are
``!suggest-float``, ``!suggest-int`` or ``!suggest-categorical`` tags, plus a
``tuning:`` section with the study settings. Each trial resolves the tags to one
concrete configuration, trains it with the existing trainer in its own
``trial_NNNN`` directory, and is ranked by the trainer's configured tracked metric.
"""

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import mlflow

from smartrodent import config_loading
from smartrodent.experiment_results import ExperimentResult
from smartrodent.hyperparameter_search import (
    EpochCallback,
    HyperparameterStudy,
    StudyConfiguration,
    StudyObjective,
    StudyResult,
    TrialConfigurationError,
)
from smartrodent.search_space import SearchSpace
from smartrodent.stage_configurations import CheckpointEncoder
from smartrodent.stages import Stage


def is_study_configuration(path: str | Path) -> bool:
    """Return whether a configuration file defines an Optuna study (a tuning section).

    Args:
        path: Configuration file.

    Returns:
        True when the document is a mapping with a ``tuning`` key.
    """
    document = config_loading.load_yaml(path)
    return isinstance(document, Mapping) and "tuning" in document


class StageStudy:
    """Study over one stage's standalone configuration.

    The document is prevalidated on construction; ``run`` trains the trials.

    Args:
        stage: The stage whose configuration class and experiment the trials use.
        document: Stage configuration with ``!suggest-*`` tags and a tuning section.
        prefix: Dotted path of the stage inside a pipeline document, for naming.
        encoder_provenance: Optional selected-encoder metadata from the pipeline.
    """

    def __init__(
        self,
        stage: Stage,
        document: Mapping,
        *,
        prefix: str = "",
        encoder_provenance: Mapping | None = None,
    ):
        """Parse the study settings and build every candidate configuration."""
        if not isinstance(document, Mapping):
            raise ValueError("A study configuration must be a mapping")
        self.stage = stage
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        values = deepcopy(dict(document))
        if "tuning" not in values:
            raise ValueError("An Optuna study requires a tuning section")
        self.settings = StudyConfiguration.from_mapping(values.pop("tuning"))
        # The space first: it rejects a tag on output_directory with a clear message.
        self.space = SearchSpace.from_document(
            values, protected=stage.configuration.protected, prefix=prefix
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
        self._reject_inputs_inside_outputs()
        reference = self.configurations[0]
        self.objective = StudyObjective(
            tracked_metric=reference.objective.metric,
            tracked_mode=reference.objective.mode,
            validation_dataset=reference.validation_dataset,
        )

    @classmethod
    def from_yaml(cls, stage: Stage, path: str | Path) -> "StageStudy":
        """Load and prevalidate a study configuration file.

        Args:
            stage: The stage to study.
            path: YAML file; paths inside are relative to the working directory.

        Returns:
            The prepared study.
        """
        return cls(stage, config_loading.load_yaml(path))

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
        """Trials trained on different encoder bytes are not comparable.

        Every candidate is included, so a searched encoder choice is covered too.
        """
        checkpoints = {
            Path(configuration.encoder.checkpoint).resolve()
            for configuration in self.configurations
            if isinstance(configuration.encoder, CheckpointEncoder)
        }
        if not checkpoints:
            return {}
        # Sorted by content, not path: moving an unchanged file keeps the identity.
        return {"encoder checkpoints": sorted(map(self._digest, checkpoints))}

    @staticmethod
    def _digest(path: Path) -> str:
        """Return a file's SHA-256."""
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def _configuration(self, document: dict, *, directory: Path):
        """Build one validated stage configuration writing to the given directory."""
        return self.stage.configuration.from_mapping(
            {**document, "output_directory": directory}
        )

    def _run_trial(
        self,
        document: dict,
        directory: Path,
        epoch_callback: EpochCallback,
        *,
        parent_run_id: str | None,
    ) -> ExperimentResult:
        """Validate and prepare one sampled document, then train it."""
        # An invalid sampled combination, including a policy that does not fit the
        # sampled encoder, fails this trial only. Training errors propagate unchanged.
        try:
            configuration = self._configuration(document, directory=directory)
            experiment = self._experiment(configuration)
        except (ValueError, TypeError) as error:
            raise TrialConfigurationError(
                f"Sampled configuration is invalid: {error}"
            ) from error
        return experiment.run_with_result(
            parent_run_id=parent_run_id,
            defer_promotion=True,
            epoch_callback=epoch_callback,
        )

    def _experiment(self, configuration):
        """Prepare one trial's experiment, handing a classifier its encoder metadata."""
        if self.encoder_provenance:
            return self.stage.experiment(
                configuration, encoder_provenance=self.encoder_provenance
            )
        return self.stage.experiment(configuration)

    def _reject_inputs_inside_outputs(self) -> None:
        """Keep every candidate's inputs out of the study directory.

        Trials and ``study.yaml`` are written there under names only known at run
        time, so an input inside it could be overwritten by a later trial.
        """
        root = self.output_directory.resolve()
        for configuration in self.configurations:
            for source in configuration.input_files():
                if source is not None and Path(source).resolve().is_relative_to(root):
                    raise ValueError(
                        f"Input {source} lies inside the study output directory {root}, "
                        "where trials write their artifacts"
                    )

    def _finish(self, result: StudyResult) -> StudyResult:
        """Compare only the study winner against the registered champion."""
        version = self.stage.experiment.promote(self.reference, result.selected)
        if version is None:
            return result
        return replace(result, registered_champion_version=version)
