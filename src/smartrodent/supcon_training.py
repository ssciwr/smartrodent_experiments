"""Prepare and run one supervised contrastive encoder experiment."""

from pathlib import Path

import torch
from torch import nn

from smartrodent.stage_configurations import SupConConfiguration
from smartrodent.stage_experiment import StageExperiment
from smartrodent.supcon_task import SupConData, SupConTask
from smartrodent.training_validation import settings_dict
from smartrodent.transforms import ImageTransforms


class SupConExperiment(StageExperiment):
    """Own prepared inputs and publication around Lightning SupCon training.

    Args:
        configuration: Validated SupCon settings.
    """

    card_task = "contrastive-encoder"
    promotes_on_publication = True

    def __init__(self, configuration: SupConConfiguration):
        """Build the task and the sampler-backed data once; nothing is written yet."""
        super().__init__(configuration)
        data = SupConData(
            configuration.train_dataset,
            configuration.validation_dataset,
            configuration.class_names,
            ImageTransforms(configuration.transforms),
            classes_per_batch=configuration.classes_per_batch,
            samples_per_class=configuration.samples_per_class,
            samples_per_epoch=configuration.samples_per_epoch,
            workers=configuration.workers,
            prefetch_factor=configuration.prefetch_factor,
            seed=configuration.seed,
            pin_memory=configuration.trainer.get("accelerator") in ("gpu", "cuda"),
        )
        self.label_names = data.training_dataset.label_names
        self.train_loader = data.train_dataloader()
        self.validation_loader = data.val_dataloader()
        torch.manual_seed(configuration.seed)
        values = settings_dict(configuration)
        self.task = SupConTask(
            values["encoder"],
            optimizer=values["optimizer"],
            scheduler=values["scheduler"],
            freeze=values["freeze"],
            output_features=configuration.output_features,
            dropout=configuration.dropout,
            temperature=configuration.temperature,
        )

    def _base_model(self) -> str:
        """Name the encoder component."""
        return self.configuration.encoder.name

    def _export(self, checkpoint: Path) -> Path:
        """Export the pooled encoder features with the shared ONNX exporter."""
        return self.exporter.encoder(checkpoint)

    def _published_model(self, checkpoint: Path) -> nn.Module:
        """Register the trained encoder of the best epoch."""
        task = SupConTask.load_from_checkpoint(checkpoint, map_location="cpu")
        return task.encoder.eval()
