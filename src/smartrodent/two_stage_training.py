"""Compose a validated SupCon sweep, configured-metric selection, and CE training."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path

import torch
import yaml

from scripts.train_supcon import SupConExperiment, TrainingConfiguration
from smartrodent import config_utils
from smartrodent.classifier_training import ClassifierConfiguration, ClassifierExperiment
from smartrodent.experiment_results import ExperimentResult, select_best_experiment
from smartrodent.experiment_support import ImageTransforms, resolve_validation_metrics
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
    Only SupCon settings may sweep. Coupled targets use full document paths.

    Args:
        output_directory: Root for the unified configuration and stage artifacts.
        data: Shared training/validation paths and class names, never a test split.
        supcon: Standalone SupCon settings without data or output overrides.
        classifier: Standalone CE settings with encoder source selected_supcon.
    """

    output_directory: Path
    data: Mapping
    supcon: Mapping
    classifier: Mapping
    supcon_configurations: tuple[TrainingConfiguration, ...] = field(init=False, repr=False)
    _classifier_configuration: ClassifierConfiguration = field(init=False, repr=False)

    def __post_init__(self):
        """Own independent settings and prevalidate all expanded stage configurations."""
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        for name in ("data", "supcon", "classifier"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(values)))
        required_data = {"train_dataset", "validation_dataset", "class_names"}
        if set(self.data) != required_data:
            raise ValueError(f"data requires exactly {sorted(required_data)}; got {sorted(self.data)}")
        if self.data["validation_dataset"] is None:
            raise ValueError("Automatic SupCon selection requires validation_dataset")
        for name in ("supcon", "classifier"):
            settings = getattr(self, name)
            conflicts = (required_data | {"output_directory", "mlflow"}) & set(settings)
            if conflicts:
                raise ValueError(f"{name} cannot override shared or deferred settings: {sorted(conflicts)}")
        self._reject_sweeps(self.data, "data")
        self._reject_sweeps(self.classifier, "classifier")
        initialization = self.classifier.get("initialization")
        if not isinstance(initialization, Mapping) or initialization.get("encoder") != {"source": "selected_supcon"}:
            raise ValueError("classifier encoder must be exactly {source: selected_supcon}, without checkpoint")
        runs = config_utils.ConfigHandler(self.to_dict()).run_configs
        if not runs:
            raise ValueError("SupCon sweep must contain at least one candidate; empty sweep")
        configurations = []
        for index, run in enumerate(runs, start=1):
            # A whole-section sweep can introduce keys absent before expansion.
            conflicts = (required_data | {"output_directory", "mlflow"}) & set(run["supcon"])
            if conflicts:
                raise ValueError(f"supcon cannot override shared or deferred settings: {sorted(conflicts)}")
            self._validate_seed(run["supcon"].get("seed", 42))
            values = {**run["supcon"], **self.data, "output_directory": self.output_directory / "supcon" / f"run_{index:03d}"}
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
            configurations.append(config)
        reference = configurations[0]
        if any(config.temperature != reference.temperature for config in configurations):
            raise ValueError("SupCon temperature must remain fixed within a sweep")
        if any(config.effective_tracked_metric != reference.effective_tracked_metric for config in configurations):
            raise ValueError("SupCon sweep must use the same tracked metric")
        if any(config.effective_tracked_mode != reference.effective_tracked_mode for config in configurations):
            raise ValueError("SupCon sweep must use the same tracked mode/direction")
        object.__setattr__(self, "supcon_configurations", tuple(configurations))
        values = {**deepcopy(self.classifier), **self.data, "output_directory": self.output_directory / "classifier"}
        # This path only prevalidates CE settings; no checkpoint is loaded until
        # all candidates complete and the actual winning best.pt is known.
        values["initialization"]["encoder"] = {"source": "supcon", "checkpoint": str(reference.output_directory / "best.pt")}
        self._validate_seed(values.get("seed", 42))
        try:
            classifier = ClassifierConfiguration.from_mapping(values)
        except TypeError as error:
            raise ValueError(f"Invalid classifier settings: {error}") from error
        load_card_template(Path(classifier.model_card_template))
        object.__setattr__(self, "_classifier_configuration", classifier)
        self._validate_output_paths()

    @staticmethod
    def _validate_seed(seed) -> None:
        """Use the intersection of the three trainers' Python/NumPy/PyTorch RNG ranges."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")

    @classmethod
    def _reject_sweeps(cls, values, section: str) -> None:
        """Reject sweep directives outside SupCon, even one-value sweeps."""
        if isinstance(values, Mapping):
            if values.get("type") in ("sweep", "coupled-sweep", "range", "random_uniform"):
                raise ValueError(f"{section} sweeps are not supported; only supcon may sweep")
            for value in values.values():
                cls._reject_sweeps(value, section)
        elif isinstance(values, (list, tuple)):
            for value in values:
                cls._reject_sweeps(value, section)

    def _validate_output_paths(self) -> None:
        """Protect shared datasets and model/head inputs from every generated artifact."""
        stage_directories = [config.output_directory for config in self.supcon_configurations]
        stage_directories.append(self._classifier_configuration.output_directory)
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
        artifacts = [self.output_directory / name for name in ("config.yaml", "selection.yaml")]
        artifacts.extend(directory / name for directory in stage_directories for name in names)
        sources = [Path(config.model_name) for config in self.supcon_configurations]
        sources.extend(Path(config.model_card_template) for config in self.supcon_configurations)
        sources.append(Path(self._classifier_configuration.model_card_template))
        head = self._classifier_configuration.initialization["head"]
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
        if set(values) != required:
            raise ValueError(f"Two-stage configuration requires {sorted(required)}; missing {sorted(required - set(values))}; unexpected {sorted(set(values) - required)}")
        return cls(**deepcopy(dict(values)))

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
        return _plain_values({
            "output_directory": self.output_directory, "data": self.data,
            "supcon": self.supcon, "classifier": self.classifier,
        })

    def classifier_configuration(self, checkpoint: Path) -> ClassifierConfiguration:
        """Resolve the CE encoder source without changing the unified settings.

        Args:
            checkpoint: Winning candidate's best.pt checkpoint.

        Returns:
            Ordinary standalone CE configuration using that SupCon source.
        """
        initialization = deepcopy(self._classifier_configuration.initialization)
        initialization["encoder"] = {"source": "supcon", "checkpoint": str(checkpoint.resolve())}
        return replace(self._classifier_configuration, initialization=initialization)


@dataclass(frozen=True)
class TwoStageResult:
    """Report completed stage results without rescoring or scanning artifacts.

    Args:
        output_directory: Pipeline artifact root.
        supcon_results: Completed candidates in sweep-product order.
        selected_supcon: Configured-metric winner, retaining the first exact tie.
        classifier_result: Completed CE training result.
    """

    output_directory: Path
    supcon_results: tuple[ExperimentResult, ...]
    selected_supcon: ExperimentResult
    classifier_result: ExperimentResult


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
        root = config.output_directory
        root.mkdir(parents=True, exist_ok=True)
        summary_path = root / "selection.yaml"
        if summary_path.exists():
            # Invalidate only this invocation's active summary before replacing
            # config/checkpoints. Retain prior stage files and unrelated runs.
            summary_path.write_text(yaml.safe_dump({
                "supcon_results": [], "selected_supcon": None, "classifier_result": None,
            }), encoding="utf-8")
        (root / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
        results = []
        for candidate in config.supcon_configurations:
            result = SupConExperiment(candidate).run_with_result()
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
            "supcon_results": [_plain_values(asdict(result)) for result in results],
            "selected_supcon": {**_plain_values(asdict(selected)), "checkpoint_sha256": checksum},
            "classifier_result": None,
        }
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        classifier = ClassifierExperiment(config.classifier_configuration(selected.best_checkpoint)).run_with_result()
        summary["classifier_result"] = _plain_values(asdict(classifier))
        summary_path.write_text(yaml.safe_dump(summary), encoding="utf-8")
        return TwoStageResult(root, tuple(results), selected, classifier)
