"""Compose a validated SupCon sweep, configured-metric selection, and CE training."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path

import mlflow
import torch
import yaml

from scripts.train_supcon import SupConExperiment, TrainingConfiguration
from smartrodent import config_utils
from smartrodent.classifier_training import ClassifierConfiguration
from smartrodent.classifier_sweeps import ClassifierSweepConfiguration, ClassifierSweepExperiment
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.experiment_support import ImageTransforms, MlflowConfiguration, resolve_validation_metrics
from smartrodent.model_card import load_card_template


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


@dataclass(frozen=True, kw_only=True)
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
    """

    output_directory: Path
    data: Mapping
    supcon: Mapping
    classifier: Mapping
    mlflow: MlflowConfiguration | None = None
    supcon_configurations: tuple[TrainingConfiguration, ...] = field(init=False, repr=False)
    _classifier_sweep: ClassifierSweepConfiguration = field(init=False, repr=False)

    def __post_init__(self):
        """Own independent settings and prevalidate all expanded stage configurations."""
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        for name in ("data", "supcon", "classifier"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(values)))
        if self.mlflow is not None and not isinstance(self.mlflow, MlflowConfiguration):
            raise TypeError("mlflow must be a MlflowConfiguration or null")
        required_data = {"train_dataset", "validation_dataset", "class_names"}
        if set(self.data) != required_data:
            raise ValueError(f"data requires exactly {sorted(required_data)}; got {sorted(self.data)}")
        if self.data["validation_dataset"] is None:
            raise ValueError("Automatic SupCon selection requires validation_dataset")
        for name in ("supcon", "classifier"):
            self._validate_stage_ownership(getattr(self, name), name)
        self._reject_sweeps(self.data, "data")
        # Expand one stage at a time: a classifier Cartesian product must never
        # multiply SupCon candidates or change which encoder is selected.
        runs = config_utils.ConfigHandler({"supcon": deepcopy(self.supcon)}).run_configs
        if not runs:
            raise ValueError("SupCon sweep must contain at least one candidate; empty sweep")
        configurations = [
            self._build_supcon_configuration(run["supcon"], index)
            for index, run in enumerate(runs, start=1)
        ]
        reference = configurations[0]
        if any(config.temperature != reference.temperature for config in configurations):
            raise ValueError("SupCon temperature must remain fixed within a sweep")
        if any(config.effective_tracked_metric != reference.effective_tracked_metric for config in configurations):
            raise ValueError("SupCon sweep must use the same tracked metric")
        if any(config.effective_tracked_mode != reference.effective_tracked_mode for config in configurations):
            raise ValueError("SupCon sweep must use the same tracked mode/direction")
        object.__setattr__(self, "supcon_configurations", tuple(configurations))
        classifier_runs = config_utils.ConfigHandler({"classifier": deepcopy(self.classifier)}).run_configs
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
            initialization = settings.get("initialization")
            if not isinstance(initialization, Mapping) or initialization.get("encoder") != {"source": "selected_supcon"}:
                raise ValueError("classifier encoder must be exactly {source: selected_supcon}, without checkpoint")
            self._validate_seed(settings.get("seed", 42))
            candidate = {**deepcopy(settings), **shared_classifier_settings}
            # The winning file does not exist yet; this validates settings and
            # source/output relationships without loading any model weights.
            candidate["initialization"]["encoder"] = {
                "source": "supcon", "checkpoint": str(reference.output_directory / "best.pt"),
            }
            values.append(candidate)
        document = _plain_values({**deepcopy(self.classifier), **shared_classifier_settings})
        sweep = ClassifierSweepConfiguration.from_runs(classifier_root, values, document)
        object.__setattr__(self, "_classifier_sweep", sweep)
        self._validate_output_paths()

    @staticmethod
    def _validate_stage_ownership(settings: Mapping, section: str) -> None:
        """Reject shared data/output overrides and deferred tracking before merging settings."""
        reserved = {
            "train_dataset", "validation_dataset", "class_names", "output_directory", "mlflow",
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
        self._validate_seed(settings.get("seed", 42))
        values = {
            **settings,
            **self.data,
            "output_directory": self.output_directory / "supcon" / f"run_{index:03d}",
            # Pipeline candidates are encoders, never classifier registry versions.
            "mlflow": asdict(replace(self.mlflow, registered_model_name=None)) if self.mlflow is not None else None,
        }
        try:
            config = TrainingConfiguration.from_mapping(values)
        except TypeError as error:
            raise ValueError(f"Invalid supcon settings: {error}") from error
        # Standalone legacy configurations may infer direction. The pipeline
        # also enforces agreement when tracked_mode was omitted explicitly.
        if config.scheduler is not None and config.scheduler.mode != config.effective_tracked_mode:
            raise ValueError("SupCon scheduler direction must agree with tracked_mode")
        if config.early_stopping is not None:
            mode = "min" if config.early_stopping["comparison"] == "less" else "max"
            if mode != config.effective_tracked_mode:
                raise ValueError("SupCon early_stopping direction must agree with tracked_mode")
        ImageTransforms(config.transforms).training()
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], **config.optimizer)
        resolve_validation_metrics(config.validation_metrics)
        load_card_template(Path(config.model_card_template))
        return config

    @staticmethod
    def _validate_seed(seed) -> None:
        """Use the intersection of the three trainers' Python/NumPy/PyTorch RNG ranges."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")

    @classmethod
    def _reject_sweeps(cls, values, section: str) -> None:
        """Reject sweeps in shared data, which must remain comparable across stages."""
        if isinstance(values, Mapping):
            if values.get("type") in ("sweep", "coupled-sweep", "range", "random_uniform"):
                raise ValueError(f"{section} sweeps are not supported; sweep stage settings instead")
            for value in values.values():
                cls._reject_sweeps(value, section)
        elif isinstance(values, (list, tuple)):
            for value in values:
                cls._reject_sweeps(value, section)

    def _validate_output_paths(self) -> None:
        """Protect shared datasets and model/head inputs from every generated artifact."""
        stage_directories = [config.output_directory for config in self.supcon_configurations]
        stage_directories.extend(config.output_directory for config in self._classifier_sweep.configurations)
        for directory in [self.output_directory, *stage_directories]:
            for split in ("train_dataset", "validation_dataset"):
                if directory.resolve().is_relative_to(Path(self.data[split]).resolve()):
                    raise ValueError("output_directory and child outputs must be outside both datasets")
        resolved = [directory.resolve() for directory in stage_directories]
        for index, directory in enumerate(resolved):
            for other in resolved[index + 1:]:
                if directory.is_relative_to(other) or other.is_relative_to(directory):
                    raise ValueError("Stage output_directory paths must not overlap")
        names = (
            "config.yaml", "history.yaml", "last.pt", "best.pt",
            ".last.pt.tmp", ".best.pt.tmp", "best.model_card.yaml", "best.model_card.md",
        )
        roots = (self.output_directory, self._classifier_sweep.output_directory)
        artifacts = [root / name for root in roots for name in ("config.yaml", "selection.yaml")]
        artifacts.extend(directory / name for directory in stage_directories for name in names)
        sources = [Path(config.model_name) for config in self.supcon_configurations]
        sources.extend(Path(config.model_card_template) for config in self.supcon_configurations)
        for classifier in self._classifier_sweep.configurations:
            sources.append(Path(classifier.model_card_template))
            head = classifier.initialization["head"]
            if head.get("checkpoint") is not None:
                sources.append(Path(head["checkpoint"]))
        for source in sources:
            for artifact in artifacts:
                same_file = source.exists() and artifact.exists() and source.samefile(artifact)
                if source.resolve() == artifact.resolve() or same_file:
                    raise ValueError(f"Output artifact {artifact} collides with source {source}")

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
        if not isinstance(values, Mapping):
            raise ValueError("Two-stage configuration must be a mapping")
        required = {"output_directory", "data", "supcon", "classifier"}
        allowed = required | {"mlflow"}
        if not required.issubset(values) or set(values) - allowed:
            raise ValueError(f"Two-stage configuration requires {sorted(required)}; missing {sorted(required - set(values))}; unexpected {sorted(set(values) - allowed)}")
        values = deepcopy(dict(values))
        values["mlflow"] = MlflowConfiguration.from_mapping(values.get("mlflow"))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TwoStageConfiguration":
        """Load a unified YAML document with the existing custom sweep tags.

        Args:
            path: Configuration filename, with paths relative to the working directory.

        Returns:
            Fully prevalidated pipeline settings.
        """
        document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=config_utils.get_loader())
        return cls.from_mapping(document)

    def to_dict(self) -> dict:
        """Return an independent YAML-compatible document, retaining unresolved sweeps.

        Returns:
            Unified settings including the selected_supcon encoder placeholder.
        """
        values = {
            "output_directory": self.output_directory, "data": self.data,
            "supcon": self.supcon, "classifier": self.classifier,
        }
        if self.mlflow is not None:
            values["mlflow"] = asdict(self.mlflow)
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
            raise ValueError("Multiple classifier configurations; use classifier_sweep_configuration")
        return configurations[0]

    def classifier_sweep_configuration(self, checkpoint: Path) -> ClassifierSweepConfiguration:
        """Inject the same selected encoder into every prepared classifier variant.

        Args:
            checkpoint: Winning SupCon candidate's best.pt file.

        Returns:
            The shared sweep settings with concrete encoder paths for every run.
        """
        configurations = []
        for configuration in self._classifier_sweep.configurations:
            initialization = deepcopy(configuration.initialization)
            initialization["encoder"] = {"source": "supcon", "checkpoint": str(checkpoint.resolve())}
            configurations.append(replace(configuration, initialization=initialization))
        return replace(self._classifier_sweep, configurations=tuple(configurations))


@dataclass(frozen=True)
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
            raise ValueError("Classifier sweep has multiple results; use classifier_results or selected_classifier")
        return self.classifier_results[0]


class TwoStageExperiment:
    """Compose existing trainers with configured-metric selection and explicit failure propagation.

    Args:
        configuration: Fully prevalidated unified experiment settings.
    """

    def __init__(self, configuration: TwoStageConfiguration):
        """Keep training stages independent; no training or output creation happens here."""
        self.configuration = configuration

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
        if config.mlflow is None:
            return self._run_pipeline()
        with config.mlflow.start_run(config.output_directory.name) as run:
            mlflow.set_tags(config.mlflow.run_tags())
            result = self._run_pipeline(run.info.run_id)
            for name in ("config.yaml", "selection.yaml"):
                mlflow.log_artifact(str(config.output_directory / name))
            return result

    def _run_pipeline(self, run_id: str | None = None) -> TwoStageResult:
        """Run stage composition, using explicit child ownership only when tracking is enabled."""
        config = self.configuration
        root = config.output_directory
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "selection.yaml"
        if summary_path.exists():
            # Invalidate only this invocation's active summary before replacing
            # config/checkpoints. Retain prior stage files and unrelated runs.
            summary_path.write_text(yaml.safe_dump({
                "supcon_results": [], "selected_supcon": None, "classifier_result": None,
                "classifier_results": [], "selected_classifier": None,
            }), encoding="utf-8")
        (root / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
        results = []
        context = {"parent_run_id": run_id} if run_id is not None else {}
        for candidate in config.supcon_configurations:
            result = SupConExperiment(candidate).run_with_result(**context)
            if result.tracked_metric != candidate.effective_tracked_metric:
                raise ValueError("SupCon result tracked metric does not match configuration")
            if result.tracked_mode != candidate.effective_tracked_mode:
                raise ValueError("SupCon result tracked mode does not match configuration")
            if result.validation_dataset != candidate.validation_dataset:
                raise ValueError("SupCon result validation dataset does not match configuration")
            results.append(result)
        selected = select_best_experiment(results)
        with selected.best_checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        summary = {
            "supcon_results": [result.to_dict() for result in results],
            "selected_supcon": {**selected.to_dict(), "checkpoint_sha256": checksum},
            "classifier_result": None, "classifier_results": [], "selected_classifier": None,
            "mlflow_run_id": run_id,
        }
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        encoder_provenance = {
            "encoder_run_id": selected.mlflow_run_id, "encoder_epoch": selected.best_epoch,
            "encoder_tracked_metric": selected.tracked_metric,
            "encoder_tracked_mode": selected.tracked_mode, "encoder_best_value": selected.best_value,
            "pipeline_run_id": run_id,
        }
        classifiers = ClassifierSweepExperiment(
            config.classifier_sweep_configuration(selected.best_checkpoint), encoder_provenance=encoder_provenance,
        ).run(**context)
        classifier_report = classifiers.to_dict()
        for name in ("classifier_results", "selected_classifier", "registered_champion_version"):
            summary[name] = classifier_report[name]
        if len(classifiers.classifier_results) == 1:
            summary["classifier_result"] = classifiers.classifier_results[0].to_dict()
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        return TwoStageResult(
            root, tuple(results), selected, classifiers.classifier_results, classifiers.selected_classifier, run_id,
        )
