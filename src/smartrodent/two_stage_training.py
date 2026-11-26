"""Compose a validated SupCon stage, configured-metric selection, and CE training.

Each stage runs either as a grid sweep or, when the root ``tuning`` section names
it, as an Optuna study. Stages run in order: the SupCon winner by its own metric
becomes the encoder of every classifier run or trial.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass as result_dataclass, field, replace

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import hashlib
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

from smartrodent.supcon_training import SupConExperiment, TrainingConfiguration
from smartrodent import config_utils
from smartrodent.classifier_training import ClassifierConfiguration
from smartrodent.classifier_sweeps import (
    ClassifierSweepConfiguration,
    ClassifierSweepExperiment,
)
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.experiment_support import MlflowConfiguration, validate_stopping
from smartrodent.hyperparameter_search import (
    ClassifierStudy,
    StudyConfiguration,
    SupConStudy,
)
from smartrodent.training_validation import (
    MappingInput,
    SharedData,
    parse_settings,
    settings_dict,
    validate_output_paths,
)


def _plain_values(value):
    """Copy configuration/result containers into YAML-compatible builtins."""
    if isinstance(value, Mapping):
        return {key: _plain_values(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_plain_values(item) for item in value]
    elif isinstance(value, Path):
        return str(value)
    else:
        return deepcopy(value)


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StageStudies:
    """Optuna study settings per pipeline stage; an omitted stage runs as a grid sweep.

    Args:
        supcon: Study settings for the SupCon stage, or None.
        classifier: Study settings for the classifier stage, or None.
    """

    supcon: MappingInput[StudyConfiguration] | None = None
    classifier: MappingInput[StudyConfiguration] | None = None

    def __post_init__(self):
        """An empty section is a mistake rather than a request for grid sweeps."""
        if self.supcon is None and self.classifier is None:
            raise ValueError("tuning must configure a study for at least one stage")


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class TwoStageConfiguration:
    """Validate the entire pipeline before running any candidate or writing outputs.

    Paths follow the standalone scripts: relative to the working directory.
    Stage sweeps expand independently. Coupled targets use full document paths.

    Args:
        output_directory: Root for the unified configuration and stage artifacts.
        data: Shared training/validation paths and class names, never a test split.
        supcon: Standalone SupCon settings without data or output overrides.
        classifier: Standalone CE settings with encoder source selected_supcon.
        mlflow: Optional shared tracking; registry publication applies only to CE.
        tuning: Optional per-stage Optuna study settings; tagged stages need one.
    """

    output_directory: Path
    data: SharedData
    supcon: dict[str, object]
    classifier: dict[str, object]
    mlflow: MappingInput[MlflowConfiguration] | None = None
    tuning: MappingInput[StageStudies] | None = None
    # Sweep runs, or a study's prevalidated candidates (first = reference).
    supcon_configurations: tuple[TrainingConfiguration, ...] = field(
        init=False, repr=False
    )
    classifier_candidates: tuple[ClassifierConfiguration, ...] = field(
        init=False, repr=False
    )
    supcon_study: SupConStudy | None = field(init=False, repr=False)
    _classifier_sweep: ClassifierSweepConfiguration | None = field(
        init=False, repr=False
    )

    def __post_init__(self):
        """Own independent settings and prevalidate all expanded stage configurations."""
        for name in ("supcon", "classifier"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        for name in ("supcon", "classifier"):
            self._validate_stage_ownership(getattr(self, name), name)
        supcon_study = self.tuning.supcon if self.tuning is not None else None
        classifier_study = self.tuning.classifier if self.tuning is not None else None
        configurations = self._prepare_supcon(supcon_study)
        reference = configurations[0]
        if any(
            config.temperature != reference.temperature for config in configurations
        ):
            raise ValueError("SupCon temperature must remain fixed within a sweep")
        if any(
            config.effective_tracked_metric != reference.effective_tracked_metric
            for config in configurations
        ):
            raise ValueError("SupCon sweep must use the same tracked metric")
        if any(
            config.effective_tracked_mode != reference.effective_tracked_mode
            for config in configurations
        ):
            raise ValueError("SupCon sweep must use the same tracked mode/direction")
        if any(
            config.tracked_metric_definition != reference.tracked_metric_definition
            for config in configurations
        ):
            raise ValueError(
                "SupCon sweep must use the same tracked metric definition/function/options"
            )
        object.__setattr__(self, "supcon_configurations", tuple(configurations))
        if classifier_study is not None:
            self._validate_classifier_encoder(self.classifier)
            sweep = None
            candidates = self.classifier_study(self.encoder_placeholder).configurations
        else:
            sweep = self._prepare_classifier_sweep(self.encoder_placeholder)
            candidates = sweep.configurations
        object.__setattr__(self, "_classifier_sweep", sweep)
        object.__setattr__(self, "classifier_candidates", tuple(candidates))
        self._validate_output_paths()

    @property
    def encoder_placeholder(self) -> Path:
        """Return a stand-in for the selected encoder before SupCon has run.

        The winning file does not exist yet; it validates settings and source/output
        relationships without loading any model weights.
        """
        return self.supcon_configurations[0].output_directory / "best.pt"

    def _prepare_supcon(
        self, study: StudyConfiguration | None
    ) -> list[TrainingConfiguration]:
        """Prepare the SupCon study, or expand its sweep, returning validated candidates."""
        if study is not None:
            supcon_study = SupConStudy(
                self._stage_document(
                    "supcon",
                    study,
                    self.output_directory / "supcon",
                    self._supcon_tracking(),
                ),
                prefix="supcon",
            )
            object.__setattr__(self, "supcon_study", supcon_study)
            return list(supcon_study.configurations)
        else:
            object.__setattr__(self, "supcon_study", None)
            # Expand one stage at a time: a classifier Cartesian product must never
            # multiply SupCon candidates or change which encoder is selected.
            runs = config_utils.ConfigHandler(
                {"supcon": deepcopy(self.supcon)}
            ).run_configs
            if not runs:
                raise ValueError(
                    "SupCon sweep must contain at least one candidate; empty sweep"
                )
            return [
                self._build_supcon_configuration(run["supcon"], index)
                for index, run in enumerate(runs, start=1)
            ]

    def _prepare_classifier_sweep(
        self, placeholder: Path
    ) -> ClassifierSweepConfiguration:
        """Expand and prevalidate the classifier sweep against a placeholder encoder."""
        classifier_runs = config_utils.ConfigHandler(
            {"classifier": deepcopy(self.classifier)}
        ).run_configs
        classifier_root = self.output_directory / "classifier"
        shared_classifier_settings = {
            **self.data,
            "output_directory": classifier_root,
            "mlflow": asdict(self.mlflow) if self.mlflow is not None else None,
        }
        values = []
        for run in classifier_runs:
            settings = run["classifier"]
            self._validate_stage_ownership(settings, "classifier")
            self._validate_classifier_encoder(settings)
            candidate = {**deepcopy(settings), **shared_classifier_settings}
            candidate["initialization"]["encoder"] = {
                "source": "supcon",
                "checkpoint": str(placeholder),
            }
            values.append(candidate)
        document = _plain_values(
            {**deepcopy(self.classifier), **shared_classifier_settings}
        )
        return ClassifierSweepConfiguration.from_runs(classifier_root, values, document)

    def _stage_document(
        self, section: str, study: StudyConfiguration, output_directory: Path, tracking
    ) -> dict:
        """Merge a tagged stage with the shared data, outputs and tracking into a study document."""
        return {
            **deepcopy(getattr(self, section)),
            **self.data,
            "output_directory": output_directory,
            "mlflow": tracking,
            "tuning": study.to_dict(),
        }

    def _supcon_tracking(self) -> dict | None:
        """Return SupCon tracking settings: pipeline candidates are never registry versions."""
        if self.mlflow is None:
            return None
        else:
            return asdict(replace(self.mlflow, registered_model_name=None))

    @staticmethod
    def _validate_classifier_encoder(settings: Mapping) -> None:
        """Require the deferred encoder placeholder, before or after sweep expansion."""
        initialization = settings.get("initialization")
        if not isinstance(initialization, Mapping) or initialization.get("encoder") != {
            "source": "selected_supcon"
        }:
            raise ValueError(
                "classifier encoder must be exactly {source: selected_supcon}, without checkpoint"
            )

    @staticmethod
    def _validate_stage_ownership(settings: Mapping, section: str) -> None:
        """Reject shared data/output overrides and deferred tracking before merging settings."""
        reserved = {
            "train_dataset",
            "validation_dataset",
            "class_names",
            "output_directory",
            "mlflow",
            "tuning",
        }
        conflicts = reserved & set(settings)
        if conflicts:
            raise ValueError(
                f"{section} cannot override shared or deferred settings: {sorted(conflicts)}"
            )

    def _build_supcon_configuration(
        self, settings: Mapping, index: int
    ) -> TrainingConfiguration:
        """Build and prevalidate one expanded candidate without training or output writes."""
        # A whole-section sweep can introduce keys absent before expansion.
        self._validate_stage_ownership(settings, "supcon")
        values = {
            **settings,
            **self.data,
            "output_directory": self.output_directory / "supcon" / f"run_{index:03d}",
            "mlflow": self._supcon_tracking(),
        }
        try:
            config = TrainingConfiguration.from_mapping(values)
        except TypeError as error:
            raise ValueError(f"Invalid supcon settings: {error}") from error
        # Standalone legacy configurations may infer direction. The pipeline
        # also enforces agreement when tracked_mode was omitted explicitly.
        validate_stopping(config, config.effective_tracked_mode)
        return config

    def _validate_output_paths(self) -> None:
        """Protect shared datasets and model/head inputs from every generated artifact."""
        stages = (*self.supcon_configurations, *self.classifier_candidates)
        # Selected encoder checkpoints are pipeline outputs, not external inputs.
        sources = [config.model_name for config in self.supcon_configurations]
        sources.extend(config.model_card_template for config in stages)
        sources.extend(
            config.initialization["head"].get("checkpoint")
            for config in self.classifier_candidates
        )
        supcon_outputs = self._candidate_outputs(
            self.supcon_configurations, self.supcon_study is not None
        )
        classifier_outputs = self._candidate_outputs(
            self.classifier_candidates, self._classifier_sweep is None
        )
        validate_output_paths(
            [*supcon_outputs, *classifier_outputs],
            (self.data["train_dataset"], self.data["validation_dataset"]),
            sources,
            (self.output_directory, self.output_directory / "classifier"),
            native_directories=classifier_outputs,
        )

    @staticmethod
    def _candidate_outputs(configurations, is_study: bool) -> list[Path]:
        """Return a stage's run directories; study candidates share their reference trial's."""
        if is_study:
            return [configurations[0].output_directory]
        else:
            return [configuration.output_directory for configuration in configurations]

    @classmethod
    def from_mapping(cls, values: Mapping) -> "TwoStageConfiguration":
        """Parse and validate a complete unified configuration.

        Args:
            values: Root mapping containing output_directory, data, supcon, classifier.

        Returns:
            Independent configuration with every expanded candidate prevalidated.

        Raises:
            ValueError: Required settings are missing, unknown, or incompatible.
        """
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TwoStageConfiguration":
        """Load a unified YAML document with the existing custom sweep tags.

        Args:
            path: Configuration filename, with paths relative to the working directory.

        Returns:
            Fully prevalidated pipeline settings.
        """
        document = yaml.load(
            Path(path).read_text(encoding="utf-8"), Loader=config_utils.get_loader()
        )
        return cls.from_mapping(document)

    def to_dict(self) -> dict:
        """Return an independent YAML-compatible document, retaining unresolved sweeps.

        Returns:
            Unified settings including the selected_supcon encoder placeholder.
        """
        values = {
            "output_directory": self.output_directory,
            "data": self.data,
            "supcon": self.supcon,
            "classifier": self.classifier,
        }
        if self.mlflow is not None:
            values["mlflow"] = asdict(self.mlflow)
        if self.tuning is not None:
            values["tuning"] = settings_dict(self.tuning)
        return _plain_values(values)

    def classifier_configuration(self, checkpoint: Path) -> ClassifierConfiguration:
        """Resolve the CE encoder source without changing the unified settings.

        Args:
            checkpoint: Winning candidate's best.pt checkpoint.

        Returns:
            Ordinary standalone CE configuration using that SupCon source.
        """
        configurations = self.classifier_sweep_configuration(checkpoint).configurations
        if len(configurations) != 1:
            raise ValueError(
                "Multiple classifier configurations; use classifier_sweep_configuration"
            )
        return configurations[0]

    def classifier_sweep_configuration(
        self, checkpoint: Path
    ) -> ClassifierSweepConfiguration:
        """Inject the same selected encoder into every prepared classifier variant.

        Args:
            checkpoint: Winning SupCon candidate's best.pt file.

        Returns:
            The shared sweep settings with concrete encoder paths for every run.

        Raises:
            ValueError: The classifier stage runs as an Optuna study.
        """
        if self._classifier_sweep is None:
            raise ValueError(
                "The classifier stage runs as an Optuna study; use classifier_study"
            )
        return self._classifier_sweep.with_supcon_encoder(checkpoint)

    def classifier_study(
        self, checkpoint: Path, encoder_provenance: Mapping | None = None
    ) -> ClassifierStudy:
        """Prepare the classifier study on one SupCon encoder checkpoint.

        Args:
            checkpoint: Selected SupCon best.pt (or a placeholder for prevalidation).
            encoder_provenance: Selected encoder and parent-run metadata.

        Returns:
            A prevalidated study whose trials all use this encoder.

        Raises:
            ValueError: The classifier stage runs as a grid sweep.
        """
        if self.tuning is None or self.tuning.classifier is None:
            raise ValueError(
                "The classifier stage runs as a grid sweep; use classifier_sweep_configuration"
            )
        tracking = asdict(self.mlflow) if self.mlflow is not None else None
        document = self._stage_document(
            "classifier",
            self.tuning.classifier,
            self.output_directory / "classifier",
            tracking,
        )
        document["initialization"]["encoder"] = {
            "source": "supcon",
            "checkpoint": str(checkpoint),
        }
        return ClassifierStudy(
            document, encoder_provenance=encoder_provenance, prefix="classifier"
        )


@result_dataclass(frozen=True)
class TwoStageResult:
    """Report completed stage results without rescoring or scanning artifacts.

    Args:
        output_directory: Pipeline artifact root.
        supcon_results: Completed candidates in sweep-product order.
        selected_supcon: Configured-metric winner, retaining the first exact tie.
        classifier_results: Completed CE runs in classifier sweep order.
        selected_classifier: Best validated CE result for this invocation.
        mlflow_run_id: Pipeline parent run ID, or None for untracked execution.
    """

    output_directory: Path
    supcon_results: tuple[ExperimentResult, ...]
    selected_supcon: ExperimentResult
    classifier_results: tuple[ExperimentResult, ...]
    selected_classifier: ExperimentResult
    mlflow_run_id: str | None = None

    @property
    def classifier_result(self) -> ExperimentResult:
        """Return the legacy single-run result, rejecting ambiguous multi-run access.

        Returns:
            The only completed classifier result.

        Raises:
            ValueError: Multiple classifiers ran; use classifier_results instead.
        """
        if len(self.classifier_results) != 1:
            raise ValueError(
                "Classifier sweep has multiple results; use classifier_results or selected_classifier"
            )
        return self.classifier_results[0]


class TwoStageExperiment:
    """Compose existing trainers with configured-metric selection and explicit failure propagation.

    Args:
        configuration: Fully prevalidated unified experiment settings.
    """

    def __init__(self, configuration: TwoStageConfiguration):
        """Prepare every stage's independent inputs before training or output creation."""
        self.configuration = configuration
        self._validate_classifier_freeze(configuration)
        # Study stages create their trainers per trial; sweeps prepare them all now.
        self.supcon_experiments = (
            tuple(
                SupConExperiment(candidate)
                for candidate in configuration.supcon_configurations
            )
            if configuration.supcon_study is None
            else ()
        )
        self.classifiers = (
            ClassifierSweepExperiment(configuration._classifier_sweep)
            if configuration._classifier_sweep is not None
            else None
        )

    @staticmethod
    def _validate_classifier_freeze(configuration: TwoStageConfiguration) -> None:
        """Reject classifier freeze settings that do not fit every SupCon candidate's network.

        The winning encoder is unknown until the SupCon sweep finishes. Its network
        keeps the layer count of its ``model_name``, so checking every candidate
        architecture now avoids failing only after all SupCon training.
        """
        model_names = sorted(
            {
                str(candidate.model_name)
                for candidate in configuration.supcon_configurations
            }
        )
        for model_name in model_names:
            layer_count = len(YOLO(model_name, task="classify").model.model)
            for classifier in configuration.classifier_candidates:
                try:
                    classifier.native_freeze(layer_count)
                except ValueError as error:
                    raise ValueError(
                        f"Classifier freeze does not fit SupCon model {model_name}: {error}"
                    ) from error

    def run(self) -> TwoStageResult:
        """Run all candidates, select best.pt, and train a fresh CE classifier.

        Returns:
            Structured results for all candidates, their winner, and the classifier.

        Raises:
            ValueError: Reported candidate metadata is incompatible for ranking.
            OSError: Artifacts cannot be read or written. Stage exceptions propagate
                unchanged; failed candidates are never skipped.
        """
        config = self.configuration
        self._preflight_studies()
        if config.mlflow is None:
            return self._run_pipeline()
        with config.mlflow.start_run(config.output_directory.name) as run:
            mlflow.set_tags(config.mlflow.run_tags())
            result = self._run_pipeline(run.info.run_id)
            for name in ("config.yaml", "selection.yaml"):
                mlflow.log_artifact(str(config.output_directory / name))
            return result

    def _preflight_studies(self) -> None:
        """Find study storage/output conflicts of both stages before any training."""
        config = self.configuration
        if config.supcon_study is not None:
            config.supcon_study.preflight()
        if config._classifier_sweep is None:
            config.classifier_study(config.encoder_placeholder).preflight()

    def _run_pipeline(self, run_id: str | None = None) -> TwoStageResult:
        """Run stage composition, using explicit child ownership only when tracking is enabled."""
        config = self.configuration
        root = config.output_directory
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "selection.yaml"
        if summary_path.exists():
            # Invalidate only this invocation's active summary before replacing
            # config/checkpoints. Retain prior stage files and unrelated runs.
            summary_path.write_text(
                yaml.safe_dump(
                    {
                        "supcon_results": [],
                        "selected_supcon": None,
                        "classifier_result": None,
                        "classifier_results": [],
                        "selected_classifier": None,
                    }
                ),
                encoding="utf-8",
            )
        (root / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict()), encoding="utf-8"
        )
        results, selected = self._run_supcon_stage(run_id)
        with selected.best_checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        summary = {
            "supcon_results": [result.to_dict() for result in results],
            "selected_supcon": {**selected.to_dict(), "checkpoint_sha256": checksum},
            "classifier_result": None,
            "classifier_results": [],
            "selected_classifier": None,
            "mlflow_run_id": run_id,
        }
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        encoder_provenance = {
            "encoder_run_id": selected.mlflow_run_id,
            "encoder_epoch": selected.best_epoch,
            "encoder_tracked_metric": selected.tracked_metric,
            "encoder_tracked_mode": selected.tracked_mode,
            "encoder_best_value": selected.best_value,
            "pipeline_run_id": run_id,
        }
        classifier_results, selected_classifier, champion_version = (
            self._run_classifier_stage(selected, encoder_provenance, run_id)
        )
        summary["classifier_results"] = [
            result.to_dict() for result in classifier_results
        ]
        summary["selected_classifier"] = (
            selected_classifier.to_dict() if selected_classifier is not None else None
        )
        summary["registered_champion_version"] = champion_version
        if len(classifier_results) == 1:
            summary["classifier_result"] = classifier_results[0].to_dict()
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        return TwoStageResult(
            root,
            tuple(results),
            selected,
            tuple(classifier_results),
            selected_classifier,
            run_id,
        )

    def _run_supcon_stage(
        self, run_id: str | None
    ) -> tuple[list[ExperimentResult], ExperimentResult]:
        """Run the SupCon study or sweep and return its results and configured-metric winner."""
        config = self.configuration
        if config.supcon_study is not None:
            study = config.supcon_study.run(parent_run_id=run_id)
            return list(study.results), study.selected
        else:
            results = []
            context = {"parent_run_id": run_id} if run_id is not None else {}
            for experiment in self.supcon_experiments:
                candidate = experiment.configuration
                result = experiment.run_with_result(**context)
                if result.tracked_metric != candidate.effective_tracked_metric:
                    raise ValueError(
                        "SupCon result tracked metric does not match configuration"
                    )
                if result.tracked_mode != candidate.effective_tracked_mode:
                    raise ValueError(
                        "SupCon result tracked mode does not match configuration"
                    )
                if result.validation_dataset != candidate.validation_dataset:
                    raise ValueError(
                        "SupCon result validation dataset does not match configuration"
                    )
                results.append(result)
            return results, select_best_experiment(results)

    def _run_classifier_stage(
        self,
        selected: ExperimentResult,
        encoder_provenance: Mapping,
        run_id: str | None,
    ) -> tuple[tuple[ExperimentResult, ...], ExperimentResult | None, str | None]:
        """Train the classifier study or sweep on the selected encoder.

        Returns:
            All completed results, the winner, and the champion version after promotion.
        """
        if self.classifiers is None:
            study = self.configuration.classifier_study(
                selected.best_checkpoint.resolve(), encoder_provenance
            ).run(parent_run_id=run_id)
            return study.results, study.selected, study.registered_champion_version
        else:
            self.classifiers.bind_supcon_encoder(
                selected.best_checkpoint, encoder_provenance
            )
            context = {"parent_run_id": run_id} if run_id is not None else {}
            sweep = self.classifiers.run(**context)
            return (
                sweep.classifier_results,
                sweep.selected_classifier,
                sweep.registered_champion_version,
            )
