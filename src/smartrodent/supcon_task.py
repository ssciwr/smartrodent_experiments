"""Sample multiview batches and train a contrastive encoder with Lightning."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from smartrodent.components import ComponentSpec
from smartrodent.datasets import ImageDataset, MultiViewImageDataset, collate_views
from smartrodent.encoders import FreezePolicy, build_encoder, encoder_checkpoint_spec
from smartrodent.experiment_results import Objective
from smartrodent.training_optimization import SchedulerPhase, configure_optimizers
from smartrodent.training_validation import parse_settings, settings_dict
from smartrodent.transforms import ImageTransforms


class SupConData:
    """Prepare two-view training batches and a fixed single-view validation cohort.

    Args:
        train_dataset: Class-folder training split.
        validation_dataset: Optional class-folder validation split.
        class_names: Species directories to include.
        transforms: Prepared training and validation image transforms.
        classes_per_batch: Number of distinct classes in each batch.
        samples_per_class: Number of source images per selected class.
        samples_per_epoch: Minimum number of training samples requested of the
            sampler, rounded down to a complete batch.
        workers: Number of loader processes.
        prefetch_factor: Batches prefetched per worker when using loader processes.
        seed: Seed for the loader generators and the validation cohort.
        pin_memory: Whether loaders should use pinned memory.
    """

    def __init__(
        self,
        train_dataset: Path,
        validation_dataset: Path | None,
        class_names: Sequence[str],
        transforms: ImageTransforms,
        *,
        classes_per_batch: int,
        samples_per_class: int,
        samples_per_epoch: int,
        workers: int,
        prefetch_factor: int = 4,
        seed: int,
        pin_memory: bool = False,
    ):
        self.batch_size = classes_per_batch * samples_per_class
        self.samples_per_class = samples_per_class
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._options = {
            "batch_size": self.batch_size,
            "num_workers": workers,
            "pin_memory": pin_memory,
            "drop_last": True,
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            self._options["prefetch_factor"] = prefetch_factor
        self.training_dataset = MultiViewImageDataset(
            train_dataset, labels=class_names, transform=transforms.training()
        )
        if (
            sum(count > 0 for count in self.training_dataset.class_counts.values())
            < classes_per_batch
        ):
            raise ValueError(
                "Training dataset has too few nonempty classes for the requested batch"
            )
        self.validation_dataset = None
        if validation_dataset is not None:
            dataset = ImageDataset(
                validation_dataset,
                labels=self.training_dataset.label_names,
                transform=transforms.validation(),
            )
            if len(dataset) < self.batch_size:
                raise ValueError(
                    "Validation dataset must contain at least one full batch"
                )
            if (
                sum(count > 0 for count in dataset.class_counts.values())
                < classes_per_batch
            ):
                raise ValueError(
                    "Validation dataset has too few nonempty classes for the requested batch"
                )
            self.validation_dataset = dataset

    def _sampler(self, dataset: ImageDataset, length: int) -> MPerClassSampler:
        """Apply the same per-class sampling policy to both splits."""
        return MPerClassSampler(
            labels=dataset.labels,
            m=self.samples_per_class,
            batch_size=self.batch_size,
            length_before_new_iter=length,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training batches with two views of each P–K source sample."""
        dataset = self.training_dataset
        return DataLoader(
            dataset,
            sampler=self._sampler(dataset, self.samples_per_epoch),
            collate_fn=collate_views,
            generator=torch.Generator().manual_seed(self.seed),
            **self._options,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return the same deterministic, single-view validation cohort on every call."""
        dataset = self.validation_dataset
        if dataset is None:
            return None
        length = len(dataset) - len(dataset) % self.batch_size
        # The sampler draws from NumPy's global RNG. Materialize validation indices
        # once per loader and restore the RNG so training draws are unaffected.
        numpy_state = np.random.get_state()
        try:
            np.random.seed(self.seed)
            indices = list(self._sampler(dataset, length))
        finally:
            np.random.set_state(numpy_state)
        return DataLoader(
            dataset,
            sampler=indices,
            generator=torch.Generator().manual_seed(self.seed),
            **self._options,
        )


class SupConTask(L.LightningModule):
    """Optimize supervised contrastive loss over a configured encoder and projector.

    Args:
        encoder: Component that constructs a plain PyTorch encoder.
        optimizer: Component that constructs a PyTorch optimizer.
        scheduler: Optional sequence of learning-rate scheduler phases.
        freeze: Parameter prefixes and BatchNorm behavior in training mode.
        output_features: Number of projected embedding features.
        dropout: Dropout probability before the projection layer.
        temperature: Contrastive loss temperature.
    """

    objective = Objective(metric="supcon_loss_mean", mode="min")

    def __init__(
        self,
        encoder: ComponentSpec | Mapping,
        *,
        optimizer: ComponentSpec | Mapping,
        scheduler: Sequence[SchedulerPhase | Mapping] = (),
        freeze: FreezePolicy | Mapping | None = None,
        output_features: int = 128,
        dropout: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        spec = parse_settings(ComponentSpec, encoder)
        self.encoder = build_encoder(spec)
        self.encoder_spec = encoder_checkpoint_spec(spec, self.encoder)
        self.freeze_policy = (
            FreezePolicy() if freeze is None else parse_settings(FreezePolicy, freeze)
        )
        self.freeze_policy.apply(self.encoder)
        self.projector = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.feature_dim, output_features)
        )
        self.criterion = SupConLoss(temperature=temperature)
        self.optimizer_spec = parse_settings(ComponentSpec, optimizer)
        self.scheduler_phases = tuple(
            parse_settings(SchedulerPhase, phase) for phase in scheduler
        )
        self.save_hyperparameters(
            {
                "encoder": settings_dict(self.encoder_spec),
                "optimizer": settings_dict(self.optimizer_spec),
                "scheduler": [settings_dict(phase) for phase in self.scheduler_phases],
                "freeze": settings_dict(self.freeze_policy),
                "output_features": output_features,
                "dropout": dropout,
                "temperature": temperature,
            }
        )
        self._training_losses: list[torch.Tensor] = []
        self._validation_losses: list[torch.Tensor] = []
        self.train()

    def train(self, mode: bool = True) -> "SupConTask":
        """Set module modes, applying the encoder's BatchNorm policy during training."""
        super().train(mode)
        if mode:
            self.freeze_policy.apply_batchnorm(self.encoder)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images as unit-length projected embeddings for SupCon."""
        return F.normalize(self.projector(self.encoder(images)), dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the contrastive loss of paired training views."""
        labels, images = batch
        loss = self.criterion(self(images), labels)
        self._training_losses.append(loss.detach())
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Collect the contrastive loss of a deterministic validation batch."""
        labels, images = batch
        loss = self.criterion(self(images), labels)
        self._validation_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self) -> None:
        """Record training loss statistics and use the mean if validation is absent."""
        primary = not self.trainer.enable_validation
        self._log_losses(self._training_losses, primary=primary)

    def on_validation_epoch_end(self) -> None:
        """Record validation statistics with the mean as the checkpoint objective."""
        self._log_losses(self._validation_losses, primary=True)

    def _log_losses(self, losses: list[torch.Tensor], *, primary: bool) -> None:
        """Aggregate detached losses and show only the objective in the progress bar."""
        values = torch.stack(losses).float()
        losses.clear()
        prefix = "supcon_loss" if primary else "train_supcon_loss"
        self.log(f"{prefix}_mean", values.mean(), prog_bar=primary)
        self.log_dict(
            {
                f"{prefix}_min": values.min(),
                f"{prefix}_std": values.std(correction=int(values.numel() > 1)),
                f"{prefix}_max": values.max(),
            }
        )

    def configure_optimizers(self):
        """Build the optimizer and scheduler for the task's trainable parameters."""
        parameters = (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        return configure_optimizers(
            parameters, self.optimizer_spec, self.scheduler_phases, self.objective
        )
