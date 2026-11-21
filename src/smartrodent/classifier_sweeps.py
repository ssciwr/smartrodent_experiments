"""Prepare and execute classifier sweeps shared by both training entry points."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from smartrodent import config_utils
from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_training import ClassifierConfiguration, ClassifierExperiment
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.training_validation import validate_output_paths


@dataclass(frozen=True)
class ClassifierSweepConfiguration:
    """Own concrete, mutually comparable runs and their original sweep document.

    Args:
        output_directory: Root for sweep configuration and selection summaries.
        configurations: Concrete classifier runs in existing sweep-product order.
        document: Original configuration, retained independently of expansion.
    """

    output_directory: Path
    configurations: tuple[ClassifierConfiguration, ...]
    document: Mapping

    def __post_init__(self):
        """Reject invalid ranking/output relationships before any training or writes."""
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        object.__setattr__(self, "document", deepcopy(dict(self.document)))
        if not self.configurations:
            raise ValueError("Classifier sweep must contain at least one candidate; empty sweep")
        reference = self.configurations[0]
        for configuration in self.configurations:
            if configuration.tracked_metric != reference.tracked_metric:
                raise ValueError("Classifier sweep must use the same tracked metric")
            if configuration.effective_tracked_mode != reference.effective_tracked_mode:
                raise ValueError("Classifier sweep must use the same tracked mode/direction")
            if configuration.tracked_metric_definition != reference.tracked_metric_definition:
                raise ValueError("Classifier sweep must use the same tracked metric definition/function/options")
            if self._validation_path(configuration.validation_dataset) != self._validation_path(reference.validation_dataset):
                raise ValueError("Classifier sweep must use the same validation dataset")
            if reference.validation_dataset is not None and sorted(configuration.class_names) != sorted(reference.class_names):
                raise ValueError("Validated classifier sweep must evaluate the same class population")
        self._validate_paths()

    @classmethod
    def from_mapping(cls, document: Mapping) -> "ClassifierSweepConfiguration":
        """Expand a standalone document with the project's existing sweep semantics.

        Args:
            document: Standalone classifier settings, optionally containing sweeps.

        Returns:
            Fully validated runs with stable, isolated output directories.
        """
        if not isinstance(document, Mapping):
            raise ValueError("Classifier configuration must be a mapping")
        values = deepcopy(dict(document))
        runs = config_utils.ConfigHandler(values).run_configs
        return cls.from_runs(Path(values["output_directory"]), runs, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassifierSweepConfiguration":
        """Read a standalone YAML file using the existing custom-tag loader.

        Args:
            path: Configuration file, with paths relative to the working directory.

        Returns:
            Prevalidated sweep configurations; no training is started.
        """
        document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=config_utils.get_loader())
        return cls.from_mapping(document)

    @classmethod
    def from_runs(
        cls, output_directory: Path, runs: Sequence[Mapping], document: Mapping
    ) -> "ClassifierSweepConfiguration":
        """Prepare already-expanded runs, including settings supplied by two-stage orchestration.

        Args:
            output_directory: Artifact root for these classifier runs.
            runs: Concrete mappings expanded with the appropriate document paths.
            document: Original settings for the sweep's configuration artifact.

        Returns:
            Validated concrete runs; singleton runs retain their original root.
        """
        configurations = []
        for index, mapping in enumerate(runs, start=1):
            values = deepcopy(dict(mapping))
            values["output_directory"] = (
                output_directory / f"run_{index:03d}" if len(runs) > 1 else output_directory
            )
            try:
                configurations.append(ClassifierConfiguration.from_mapping(values))
            except TypeError as error:
                raise ValueError(f"Invalid classifier settings: {error}") from error
        return cls(output_directory, tuple(configurations), document)

    @staticmethod
    def _validation_path(path: Path | None) -> Path | None:
        """Treat aliases of the same validation directory as the same input."""
        return path.resolve() if path is not None else None

    def _validate_paths(self) -> None:
        """Protect every run's inputs against every run's artifacts, including aliases."""
        sources = [configuration.model_card_template for configuration in self.configurations]
        sources.extend(node.get("checkpoint") for configuration in self.configurations for node in configuration.initialization.values())
        validate_output_paths(
            [configuration.output_directory for configuration in self.configurations],
            [path for configuration in self.configurations for path in (configuration.train_dataset, configuration.validation_dataset)],
            sources, (self.output_directory,),
        )


@dataclass(frozen=True)
class ClassifierSweepResult:
    """Report all completed runs and distinguish the local winner from the registry champion.

    Args:
        output_directory: Classifier sweep's artifact root.
        classifier_results: Completed runs in sweep order.
        selected_classifier: Best validated result, or None for training-only runs.
        registered_champion_version: Alias version after comparison, when applicable.
    """

    output_directory: Path
    classifier_results: tuple[ExperimentResult, ...]
    selected_classifier: ExperimentResult | None
    registered_champion_version: str | None = None

    def to_dict(self) -> dict:
        """Return a portable report including losing runs and original configured scores.

        Returns:
            YAML-compatible selection metadata.
        """
        return {
            "output_directory": str(self.output_directory),
            "classifier_results": [result.to_dict() for result in self.classifier_results],
            "selected_classifier": self.selected_classifier.to_dict() if self.selected_classifier is not None else None,
            "registered_champion_version": self.registered_champion_version,
        }


class ClassifierSweepExperiment:
    """Execute prepared classifiers and promote only a fully published, comparable winner.

    Args:
        configuration: Prevalidated classifier sweep settings.
        encoder_provenance: Optional selected-SupCon metadata supplied by the pipeline.
    """

    def __init__(self, configuration: ClassifierSweepConfiguration, *, encoder_provenance: Mapping | None = None):
        """Retain preparation separately from expensive training and filesystem writes."""
        self.configuration = configuration
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))

    def run(self, *, parent_run_id: str | None = None) -> ClassifierSweepResult:
        """Train every variant, save all results, and compare the validated winner.

        Args:
            parent_run_id: Active two-stage MLflow parent; standalone runs stay roots.

        Returns:
            Complete results and the local winner, without rescoring checkpoint files.

        Raises:
            ValueError: A reported result or historical champion is incompatible.
            Exception: Training/publication failures propagate without selecting a
                partial winner or promoting a candidate before all runs succeed.
        """
        configuration = self.configuration
        root = configuration.output_directory
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "selection.yaml"
        # Clear the active report before any previous run artifact is replaced.
        self._write_summary(summary_path, ClassifierSweepResult(root, (), None))
        (root / "config.yaml").write_text(yaml.safe_dump(dict(configuration.document)), encoding="utf-8")
        results = []
        for candidate in configuration.configurations:
            context = {"parent_run_id": parent_run_id} if parent_run_id is not None else {}
            if candidate.mlflow is not None and candidate.mlflow.registered_model_name is not None:
                context["defer_promotion"] = True
            result = ClassifierExperiment(candidate, encoder_provenance=self.encoder_provenance).run_with_result(**context)
            self._validate_result(candidate, result)
            results.append(result)
            self._write_summary(summary_path, ClassifierSweepResult(root, tuple(results), None))
        selected = select_best_experiment(results) if results[0].validation_dataset is not None else None
        champion_version = None
        if selected is not None:
            candidate = configuration.configurations[results.index(selected)]
            if candidate.mlflow is not None and candidate.mlflow.registered_model_name is not None:
                champion_version = ClassifierChampionRegistry(
                    candidate.mlflow, candidate.tracked_metric_definition, candidate.class_names,
                ).promote(selected)
        report = ClassifierSweepResult(root, tuple(results), selected, champion_version)
        self._write_summary(summary_path, report)
        return report

    @staticmethod
    def _validate_result(configuration: ClassifierConfiguration, result: ExperimentResult) -> None:
        """Check each result against its configuration, even for a singleton or uniform lie."""
        if result.tracked_metric != configuration.tracked_metric:
            raise ValueError("Classifier result tracked metric does not match configuration")
        if result.tracked_mode != configuration.effective_tracked_mode:
            raise ValueError("Classifier result tracked mode does not match configuration")
        if ClassifierSweepConfiguration._validation_path(result.validation_dataset) != ClassifierSweepConfiguration._validation_path(configuration.validation_dataset):
            raise ValueError("Classifier result validation dataset does not match configuration")

    @staticmethod
    def _write_summary(path: Path, result: ClassifierSweepResult) -> None:
        """Persist the current invocation's explicit results, never rediscover old checkpoints."""
        path.write_text(yaml.safe_dump(result.to_dict()), encoding="utf-8")
