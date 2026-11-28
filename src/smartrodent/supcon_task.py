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
        self._validation_anchor_losses: list[torch.Tensor] = []
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
        """Collect the batch and per-anchor contrastive losses of a validation batch."""
        labels, images = batch
        embeddings = self(images)
        loss = self.criterion(embeddings, labels)
        self._validation_losses.append(loss.detach())
        # float32: a half-precision trainer yields half-precision losses, which
        # torch.quantile rejects; the statistics need no half-precision speed.
        self._validation_anchor_losses.append(
            self._anchor_losses(embeddings, labels).detach().float()
        )
        return loss

    def on_train_epoch_start(self) -> None:
        """Start each epoch's training statistics from nothing.

        With validation, the statistics are logged by the validation hook, which does
        not run in every epoch when ``check_val_every_n_epoch`` or ``val_check_interval``
        says so. Clearing here makes logged statistics cover only their own epoch's
        batches; an epoch without validation logs no train statistics.
        """
        self._training_losses.clear()

    def on_train_epoch_end(self) -> None:
        """Log training loss statistics as the objective; with validation they go below."""
        if self.trainer.enable_validation:
            return  # logged with the validation metrics, which keeps one CSV row per epoch
        self._log_losses(self._training_losses, primary=True)

    def on_validation_epoch_end(self) -> None:
        """Log validation statistics as the objective, with the epoch's training statistics.

        Validation runs after the epoch's training batches and before its
        ``on_train_epoch_end``. Logging everything in this one hook makes the CSV
        logger write a single complete row per epoch.
        """
        self._log_losses(self._validation_losses, primary=True)
        self._log_anchor_losses()
        if not self.trainer.sanity_checking:  # no training has happened yet then
            self._log_losses(self._training_losses, primary=False)

    def _anchor_losses(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Return the loss of each anchor that has a positive (another sample of its class).

        The mask comes from the labels instead of from the loss values: the library
        gives an anchor without positives a loss of 0 that the objective's reducer
        ignores, so including it would pull the statistics below the objective's mean.
        The library also returns a bare 0 instead of a tensor for a batch without any
        positive pair or without any negative pair (a single class), hence the
        explicit empty case.
        """
        has_positive = (labels[:, None] == labels[None, :]).sum(dim=1) > 1
        has_negative = labels.unique().numel() > 1
        if not (has_positive.any() and has_negative):
            return embeddings.new_empty(0)
        losses = self.criterion.compute_loss(embeddings, labels, None, None, None)
        return losses["loss"]["losses"][has_positive]

    def _log_anchor_losses(self) -> None:
        """Log the validation epoch's per-anchor loss distribution, then reset it.

        The batch-level statistics hide a few badly embedded samples inside a batch's
        mean. The median and 95th percentile are linear-interpolated quantiles; with no
        anchor that has a positive in the whole epoch the statistics are NaN.
        """
        anchors = torch.cat(self._validation_anchor_losses)
        self._validation_anchor_losses.clear()
        if anchors.numel() == 0:
            nan = torch.full((), float("nan"), device=anchors.device)
            statistics = dict.fromkeys(("min", "median", "p95", "max"), nan)
        else:
            statistics = {
                "min": anchors.min(),
                "median": anchors.quantile(0.5),
                "p95": anchors.quantile(0.95),
                "max": anchors.max(),
            }
        self.log_dict(
            {f"supcon_anchor_loss_{name}": value for name, value in statistics.items()}
        )

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
