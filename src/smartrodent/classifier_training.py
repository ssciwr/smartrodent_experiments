"""Prepare and run one cross-entropy classifier experiment."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from smartrodent.classifier_champions import ClassifierChampionRegistry
from smartrodent.classifier_task import ClassifierTask
from smartrodent.datasets import ImageDataset
from smartrodent.experiment_results import ExperimentResult
from smartrodent.stage_configurations import CheckpointEncoder, ClassifierConfiguration
from smartrodent.stage_experiment import StageExperiment
from smartrodent.training_validation import settings_dict
from smartrodent.transforms import ImageTransforms


class ClassifierExperiment(StageExperiment):
    """Own prepared inputs and publication around Lightning classifier training.

    Args:
        configuration: Validated classifier settings.
        encoder_provenance: Optional encoder-selection metadata from orchestration.

    Raises:
        ValueError: An encoder policy, such as a native export request, does not fit.
    """

    card_task = "image-classification"

    def __init__(
        self,
        configuration: ClassifierConfiguration,
        *,
        encoder_provenance: Mapping | None = None,
    ):
        """Build the seeded task and data loaders; nothing is written yet."""
        super().__init__(configuration)
        self.encoder_provenance = deepcopy(dict(encoder_provenance or {}))
        transforms = ImageTransforms(configuration.transforms)
        training = self._dataset(
            configuration.train_dataset,
            configuration.class_names,
            transforms.training(),
        )
        self.label_names = training.label_names
        validation = (
            self._dataset(
                configuration.validation_dataset,
                self.label_names,
                transforms.validation(),
            )
            if configuration.validation_dataset is not None
            else None
        )
        options = {
            "batch_size": configuration.batch_size,
            "num_workers": configuration.workers,
        }
        self.train_loader = DataLoader(
            training,
            shuffle=True,
            generator=torch.Generator().manual_seed(configuration.seed),
            **options,
        )
        self.validation_loader = (
            DataLoader(validation, **options) if validation is not None else None
        )
        torch.manual_seed(configuration.seed)
        values = settings_dict(configuration)
        self.task = ClassifierTask(
            values["encoder"],
            label_names=self.label_names,
            optimizer=values["optimizer"],
            scheduler=values["scheduler"],
            freeze=values["freeze"],
            head=values["head"],
            reinitialization=values["reinitialization"],
            native_checkpoint=configuration.native_checkpoint,
            metrics=values["validation_metrics"],
            tracked_metric=configuration.tracked_metric,
            tracked_mode=configuration.tracked_mode,
        )

    @staticmethod
    def _dataset(path: Path, labels, transform) -> ImageDataset:
        """Index a split and require every requested class to be present."""
        dataset = ImageDataset(Path(path), labels, transform=transform)
        if any(count == 0 for count in dataset.class_counts.values()):
            raise ValueError("Every requested dataset class must be nonempty")
        return dataset

    def _base_model(self) -> str:
        """Name the encoder's source checkpoint or component."""
        encoder = self.configuration.encoder
        if isinstance(encoder, CheckpointEncoder):
            return str(encoder.checkpoint)
        return encoder.name

    def _export(self, checkpoint: Path) -> Path:
        """Export class probabilities with the shared ONNX exporter."""
        return self.exporter.classifier(checkpoint)

    def _published_model(self, checkpoint: Path) -> nn.Module:
        """Register the native model for YOLO-style exports, else the task itself."""
        task = ClassifierTask.load_from_checkpoint(checkpoint, map_location="cpu")
        task.eval()
        return task.native_model() if self.configuration.native_checkpoint else task

    def _publication_tags(self, result: ExperimentResult) -> dict:
        """Add the champion comparison identity and the encoder's provenance."""
        return {
            **self.encoder_provenance,
            **self._registry(self.configuration).comparison_tags(result),
        }

    @staticmethod
    def _registry(configuration: ClassifierConfiguration) -> ClassifierChampionRegistry:
        """Return the champion registry of a configuration with registration enabled."""
        return ClassifierChampionRegistry(
            configuration.mlflow,
            configuration.metric_definition,
            configuration.class_names,
        )

    @classmethod
    def promote(
        cls, configuration: ClassifierConfiguration, result: ExperimentResult
    ) -> str | None:
        """Compare a registered result with the champion; None when not registered."""
        mlflow_settings = configuration.mlflow
        if mlflow_settings is None or mlflow_settings.registered_model_name is None:
            return None
        return cls._registry(configuration).promote(result)
