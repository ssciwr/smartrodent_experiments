"""Compose a validated SupCon stage, configured-metric selection, and CE training.

Each stage is either a single run or, when the root ``tuning`` section names it, an
Optuna study. Stages run in order: the SupCon winner by its own metric becomes the
encoder of the classifier run or of every classifier trial.
"""

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from dataclasses import dataclass as result_dataclass
from pathlib import Path

import mlflow
import yaml
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from smartrodent import config_loading
from smartrodent.encoders import build_encoder, preflight_encoders
from smartrodent.experiment_results import ExperimentResult
from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.hyperparameter_search import StudyConfiguration
from smartrodent.model_card import load_training_template
from smartrodent.output_paths import validate_output_paths
from smartrodent.search_space import reject_search_tags
from smartrodent.stage_configurations import (
    ClassifierConfiguration,
    SupConConfiguration,
)
from smartrodent.stages import CLASSIFIER, SUPCON
from smartrodent.training_studies import StageStudy
from smartrodent.training_validation import (
    MappingInput,
    SharedData,
    parse_settings,
    settings_dict,
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


def _validate_stage_ownership(settings: Mapping, section: str) -> None:
    """Reject stage overrides of the settings the pipeline owns or supplies."""
    reserved = {
        "train_dataset",
        "validation_dataset",
        "class_names",
        "output_directory",
        "mlflow",
        "tuning",
    }
    if section == "classifier":
        reserved.add("encoder")  # supplied by the pipeline: the winning SupCon encoder
    conflicts = reserved & set(settings)
    if conflicts:
        raise ValueError(
            f"{section} cannot override shared or deferred settings: {sorted(conflicts)}"
        )


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class StageStudies:
    """Optuna study settings per pipeline stage; an omitted stage runs once.

    Args:
        supcon: Study settings for the SupCon stage, or None.
        classifier: Study settings for the classifier stage, or None.
    """

    supcon: MappingInput[StudyConfiguration] | None = None
    classifier: MappingInput[StudyConfiguration] | None = None

    def __post_init__(self):
        """An empty section is a mistake rather than a request for single runs."""
        if self.supcon is None and self.classifier is None:
            raise ValueError("tuning must configure a study for at least one stage")


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid"))
class TwoStageConfiguration:
    """The settings of one pipeline, as written; ``TwoStagePlan`` prepares its stages.

    Paths follow the standalone scripts: relative to the working directory. Coupled
    targets use full document paths.

    Args:
        output_directory: Root for the unified configuration and stage artifacts.
        data: Shared training/validation paths and class names, never a test split.
        supcon: Standalone SupCon settings without data or output overrides.
        classifier: Standalone CE settings without an encoder, which SupCon supplies.
        mlflow: Optional shared tracking; registry publication applies only to CE.
        tuning: Optional per-stage Optuna study settings; tagged stages need one.
    """

    output_directory: Path
    data: SharedData
    supcon: dict[str, object]
    classifier: dict[str, object]
    mlflow: MappingInput[MlflowConfiguration] | None = None
    tuning: MappingInput[StageStudies] | None = None

    def __post_init__(self):
        """Keep shared data, outputs, tracking and the encoder owned by the pipeline root."""
        for section in ("supcon", "classifier"):
            _validate_stage_ownership(getattr(self, section), section)

    def study(self, stage: str) -> StudyConfiguration | None:
        """Return a stage's study settings, or None when the stage runs once.

        Args:
            stage: ``supcon`` or ``classifier``.

        Returns:
            The stage's ``tuning`` settings, if any.
        """
        return getattr(self.tuning, stage) if self.tuning is not None else None

    @classmethod
    def from_mapping(cls, values: Mapping) -> "TwoStageConfiguration":
        """Parse a complete unified configuration.

        Args:
            values: Root mapping containing output_directory, data, supcon, classifier.

        Returns:
            Independent configuration; its stages are prepared by ``TwoStagePlan``.

        Raises:
            ValueError: Required settings are missing, unknown, or owned by the root.
        """
        return parse_settings(cls, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TwoStageConfiguration":
        """Load a unified YAML document with the study tags.

        Args:
            path: Configuration filename, with paths relative to the working directory.

        Returns:
            The parsed pipeline settings.
        """
        return cls.from_mapping(config_loading.load_yaml(path))

    def to_dict(self) -> dict:
        """Return an independent YAML-compatible document, retaining search tags.

        Returns:
            Unified settings; the classifier encoder is supplied at run time.
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


class TwoStagePlan:
    """Validate every stage candidate of a pipeline before anything trains or is written.

    The SupCon winner is unknown until SupCon has run, so classifier candidates are
    validated against a placeholder encoder checkpoint, and their encoder policies
    must fit every SupCon candidate's encoder.

    Args:
        configuration: Parsed pipeline settings.
    """

    def __init__(self, configuration: TwoStageConfiguration):
        """Prepare both stages' candidate configurations and check their inputs."""
        self.configuration = configuration
        root = configuration.output_directory
        self.supcon_root = root / "supcon"
        self.classifier_root = root / "classifier"
        supcon_study = configuration.study("supcon")
        self.supcon_study = (
            StageStudy(
                SUPCON,
                self._stage_document("supcon", self.supcon_root, supcon_study),
                prefix="supcon",
            )
            if supcon_study is not None
            else None
        )
        # A study's candidates are its prevalidated reference and edge configurations.
        self.supcon_configurations = (
            self.supcon_study.configurations
            if self.supcon_study is not None
            else (self._supcon_run(),)
        )
        # The selected encoder does not exist yet; classifiers are validated against
        # the path of the reference SupCon candidate's best.ckpt instead.
        placeholder = self.supcon_configurations[0].artifacts.best_checkpoint
        self.classifier_candidates = (
            self.classifier_study(placeholder, None).configurations
            if configuration.study("classifier") is not None
            else (self.classifier_run(placeholder),)
        )
        self._validate_output_paths()
        self._preflight_encoders()
        # Read every template now: a malformed one must not fail after SupCon trained.
        for candidate in self.classifier_candidates:
            load_training_template(Path(candidate.model_card_template))

    def classifier_run(self, checkpoint: Path) -> ClassifierConfiguration:
        """Return the single classifier run on a SupCon encoder.

        Args:
            checkpoint: The selected SupCon best.ckpt, or a placeholder for validation.

        Returns:
            Ordinary standalone CE configuration using that SupCon source.
        """
        reject_search_tags(self.configuration.classifier, prefix="classifier")
        return ClassifierConfiguration.from_mapping(
            self._classifier_document(checkpoint, self.classifier_root)
        )

    def classifier_study(
        self,
        checkpoint: Path,
        encoder_key: str | None,
        encoder_provenance: Mapping | None = None,
    ) -> StageStudy:
        """Prepare the classifier study on one SupCon encoder.

        Each selected encoder gets its own study: trials on different encoders are
        not comparable, so a new SupCon winner starts a fresh classifier study while
        an unchanged winner can resume its own.

        Args:
            checkpoint: Selected SupCon best.ckpt, or a placeholder for validation.
            encoder_key: Short checksum of the selected encoder; it names the study
                (``<study_name>-<key>``) and its directory (``classifier/<key>``).
                None validates against the whole classifier stage directory.
            encoder_provenance: Selected encoder and parent-run metadata.

        Returns:
            A prevalidated study whose trials all use this encoder.
        """
        document = self._classifier_document(
            checkpoint,
            self.classifier_root / encoder_key
            if encoder_key is not None
            else self.classifier_root,
            self.configuration.study("classifier"),
        )
        if encoder_key is not None:
            document["tuning"]["study_name"] += f"-{encoder_key}"
        return StageStudy(
            CLASSIFIER,
            document,
            prefix="classifier",
            encoder_provenance=encoder_provenance,
        )

    def _supcon_run(self) -> SupConConfiguration:
        """Return the single SupCon run, which writes to the stage directory."""
        reject_search_tags(self.configuration.supcon, prefix="supcon")
        return SupConConfiguration.from_mapping(
            self._stage_document("supcon", self.supcon_root)
        )

    def _stage_document(
        self,
        section: str,
        output_directory: Path,
        study: StudyConfiguration | None = None,
    ) -> dict:
        """Merge a stage with the shared data, its outputs and tracking, and its study."""
        document = {
            **deepcopy(getattr(self.configuration, section)),
            **self.configuration.data,
            "output_directory": output_directory,
            "mlflow": self._tracking(section),
        }
        if study is not None:
            document["tuning"] = study.to_dict()
        return document

    def _classifier_document(
        self,
        checkpoint: Path,
        output_directory: Path,
        study: StudyConfiguration | None = None,
    ) -> dict:
        """Return the classifier stage document with the SupCon checkpoint as its encoder."""
        document = self._stage_document("classifier", output_directory, study)
        document["encoder"] = {"checkpoint": str(checkpoint)}
        return document

    def _tracking(self, section: str) -> dict | None:
        """Return a stage's tracking settings; only classifiers become registry versions."""
        tracking = self.configuration.mlflow
        if tracking is None:
            return None
        elif section == "supcon":
            return asdict(replace(tracking, registered_model_name=None))
        return asdict(tracking)  # the classifier stage

    def _validate_output_paths(self) -> None:
        """Protect shared datasets and model inputs from every generated artifact."""
        candidates = [*self.supcon_configurations, *self.classifier_candidates]
        # A study's candidates share their reference trial's directory.
        runs = list(dict.fromkeys(c.artifacts for c in candidates))
        # The selected encoder is an output of the SupCon stage, not an external input.
        sources = [
            *(f for c in self.supcon_configurations for f in c.input_files()),
            *(
                f
                for c in self.classifier_candidates
                for f in c.input_files()
                if f != c.encoder.checkpoint
            ),
        ]
        data = self.configuration.data
        root = self.configuration.output_directory
        validate_output_paths(
            runs,
            (data["train_dataset"], data["validation_dataset"]),
            sources,
            files=(root / "config.yaml", root / "selection.yaml"),
        )

    def _preflight_encoders(self) -> None:
        """Check every classifier candidate's encoder policies against every SupCon encoder.

        The winning encoder is unknown until the SupCon stage finishes, so a policy
        that fits only some candidates would fail after all SupCon training.
        """
        encoders = [
            build_encoder(configuration.encoder)
            for configuration in self.supcon_configurations
        ]
        for classifier in self.classifier_candidates:
            preflight_encoders(
                encoders,
                freeze=classifier.freeze,
                reinitialization=classifier.reinitialization,
                native=classifier.native_checkpoint,
                full_head=classifier.head == "full",
            )


@result_dataclass(frozen=True)
class TwoStageResult:
    """Report completed stage results without rescoring or scanning artifacts.

    Args:
        output_directory: Pipeline artifact root.
        supcon_results: Completed SupCon runs or trials in order.
        selected_supcon: Configured-metric winner, retaining the first exact tie.
        classifier_results: Completed CE runs or trials in order.
        selected_classifier: Best validated CE result for this invocation.
        mlflow_run_id: Pipeline parent run ID, or None for untracked execution.
    """

    output_directory: Path
    supcon_results: tuple[ExperimentResult, ...]
    selected_supcon: ExperimentResult
    classifier_results: tuple[ExperimentResult, ...]
    selected_classifier: ExperimentResult
    mlflow_run_id: str | None = None


class TwoStageExperiment:
    """Compose the stage trainers with configured-metric selection and explicit failure propagation.

    Args:
        configuration: Parsed pipeline settings; preparation validates every stage.
    """

    def __init__(self, configuration: TwoStageConfiguration):
        """Validate every stage candidate before training or output creation."""
        self.configuration = configuration
        self.plan = TwoStagePlan(configuration)
        self.supcon_experiment = (
            SUPCON.experiment(self.plan.supcon_configurations[0])
            if self.plan.supcon_study is None
            else None
        )

    def run(self) -> TwoStageResult:
        """Run the SupCon stage, select its best.ckpt, and train the classifier stage on it.

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
        """Find SupCon study storage/output conflicts before any training.

        A classifier study is keyed by the encoder SupCon selects, so it has no
        location to check yet, and no conflict with studies of other encoders. One
        case remains late: persistent classifier storage with load_if_exists false
        and a rerun selecting the same encoder fails after SupCon as a duplicate.
        """
        if self.plan.supcon_study is not None:
            self.plan.supcon_study.preflight()

    def _run_pipeline(self, run_id: str | None = None) -> TwoStageResult:
        """Run stage composition, using explicit child ownership only when tracking is enabled."""
        config = self.configuration
        root = config.output_directory
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "selection.yaml"
        empty_summary = {
            "supcon_results": [],
            "selected_supcon": None,
            "classifier_results": [],
            "selected_classifier": None,
        }
        if summary_path.exists():
            # Invalidate only this invocation's active summary before replacing
            # config/checkpoints. Retain prior stage files and unrelated runs.
            summary_path.write_text(yaml.safe_dump(empty_summary), encoding="utf-8")
        (root / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict()), encoding="utf-8"
        )
        results, selected = self._run_supcon_stage(run_id)
        with selected.best_checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        summary = {
            **empty_summary,
            "supcon_results": [result.to_dict() for result in results],
            "selected_supcon": {**selected.to_dict(), "checkpoint_sha256": checksum},
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
            self._run_classifier_stage(selected, checksum, encoder_provenance, run_id)
        )
        summary.update(
            classifier_results=[result.to_dict() for result in classifier_results],
            selected_classifier=selected_classifier.to_dict(),
            registered_champion_version=champion_version,
        )
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
        """Run the SupCon study or single run and return its results and winner."""
        if self.plan.supcon_study is not None:
            study = self.plan.supcon_study.run(parent_run_id=run_id)
            return list(study.results), study.selected
        else:
            result = self.supcon_experiment.run_with_result(parent_run_id=run_id)
            return [result], result

    def _run_classifier_stage(
        self,
        selected: ExperimentResult,
        checksum: str,
        encoder_provenance: Mapping,
        run_id: str | None,
    ) -> tuple[tuple[ExperimentResult, ...], ExperimentResult, str | None]:
        """Train the classifier study or single run on the selected encoder.

        Args:
            selected: Selected SupCon result.
            checksum: SHA-256 of the selected best.pt; keys the classifier study.
            encoder_provenance: Selected encoder and parent-run metadata.
            run_id: Pipeline parent run, or None when untracked.

        Returns:
            All completed results, the winner, and the champion version after promotion.
        """
        encoder = selected.best_checkpoint.resolve()
        if self.configuration.study("classifier") is not None:
            study = self.plan.classifier_study(
                encoder, checksum[:12], encoder_provenance
            ).run(parent_run_id=run_id)
            return study.results, study.selected, study.registered_champion_version
        configuration = self.plan.classifier_run(encoder)
        experiment = CLASSIFIER.experiment(
            configuration, encoder_provenance=encoder_provenance
        )
        # Promote explicitly, as a study does, to report the champion version.
        result = experiment.run_with_result(parent_run_id=run_id, defer_promotion=True)
        champion = CLASSIFIER.experiment.promote(configuration, result)
        return (result,), result, champion
