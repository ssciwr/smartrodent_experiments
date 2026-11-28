"""Train an encoder and a linear head with cross-entropy using Lightning."""

from collections.abc import Mapping, Sequence
from typing import Literal

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from smartrodent.components import ComponentSpec
from smartrodent.encoders import (
    FreezePolicy,
    Reinitialization,
    build_encoder,
    encoder_checkpoint_spec,
)
from smartrodent.experiment_results import Objective
from smartrodent.metrics import ValidationMetrics
from smartrodent.training_optimization import SchedulerPhase, configure_optimizers
from smartrodent.training_validation import (
    Mode,
    parse_settings,
    settings_dict,
)

CE_METRIC = "ce_loss_mean"


def _checkpoint_encoder(path: str) -> tuple[ComponentSpec, nn.Module]:
    """Rebuild the encoder stored in a Lightning checkpoint and load its weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    spec = parse_settings(ComponentSpec, checkpoint["hyper_parameters"]["encoder"])
    encoder = build_encoder(spec)
    prefix = "encoder."
    encoder.load_state_dict(
        {
            key.removeprefix(prefix): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
    )
    return spec, encoder


class ClassifierTask(L.LightningModule):
    """Fine-tune an encoder with a linear head under whole-split cross-entropy.

    Args:
        encoder: Component spec of the encoder, or ``{checkpoint: path}`` naming a
            SupCon or classifier ``.ckpt`` it is restored from.
        label_names: Class name to class ID mapping; its size is the head width.
        optimizer: Component that constructs a PyTorch optimizer.
        scheduler: Optional sequence of learning-rate scheduler phases.
        freeze: Parameter prefixes and BatchNorm behavior in training mode.
        head: ``linear`` adds only a new final layer; ``full`` also renews the
            encoder's post-pooling part through its ``reinitialize_post_pooling``.
        reinitialization: Optional random reset of encoder modules.
        native_checkpoint: Whether the run exports a native checkpoint; needs an
            encoder with ``to_native``.
        metrics: Optional sklearn scorers by alias, evaluated on the validation split.
        tracked_metric: ``ce_loss_mean`` or a metric alias selecting the best epoch.
        tracked_mode: Direction; defaults to ``min`` for the loss, ``max`` for scores.

    Raises:
        ValueError: A policy does not fit the encoder, or the metric is unknown.
    """

    def __init__(
        self,
        encoder: ComponentSpec | Mapping,
        *,
        label_names: Mapping[str, int],
        optimizer: ComponentSpec | Mapping,
        scheduler: Sequence[SchedulerPhase | Mapping] = (),
        freeze: FreezePolicy | Mapping | None = None,
        head: Literal["linear", "full"] = "linear",
        reinitialization: Reinitialization | Mapping | None = None,
        native_checkpoint: bool = False,
        metrics: Mapping[str, ComponentSpec | Mapping] | None = None,
        tracked_metric: str = CE_METRIC,
        tracked_mode: Mode | None = None,
    ):
        super().__init__()
        if isinstance(encoder, Mapping) and "checkpoint" in encoder:
            spec, self.encoder = _checkpoint_encoder(str(encoder["checkpoint"]))
        else:
            spec = parse_settings(ComponentSpec, encoder)
            self.encoder = build_encoder(spec)
        self.encoder_spec = encoder_checkpoint_spec(spec, self.encoder)
        if native_checkpoint and not callable(getattr(self.encoder, "to_native", None)):
            raise ValueError(
                f"{type(self.encoder).__name__} cannot export a native checkpoint: "
                "it has no to_native method"
            )
        self.label_names = dict(label_names)
        self.head = self._build_head(head)
        self.freeze_policy = (
            FreezePolicy() if freeze is None else parse_settings(FreezePolicy, freeze)
        )
        # Reinitialize before freezing so that frozen modules are never redrawn.
        if reinitialization is not None:
            parse_settings(Reinitialization, reinitialization).apply(self.encoder)
        self.freeze_policy.apply(self.encoder)
        self.native_checkpoint = native_checkpoint
        self.optimizer_spec = parse_settings(ComponentSpec, optimizer)
        self.scheduler_phases = tuple(
            parse_settings(SchedulerPhase, phase) for phase in scheduler
        )
        self.metrics = ValidationMetrics(
            {
                alias: parse_settings(ComponentSpec, spec)
                for alias, spec in (metrics or {}).items()
            }
        )
        if tracked_metric != CE_METRIC and tracked_metric not in (metrics or {}):
            raise ValueError(f"tracked_metric '{tracked_metric}' is not available")
        self.objective = Objective(
            tracked_metric,
            tracked_mode or ("min" if tracked_metric == CE_METRIC else "max"),
        )
        # Head renewal and reinitialization are training-time actions: the saved
        # weights already contain their result, so they are not restored.
        self.save_hyperparameters(
            {
                "encoder": settings_dict(self.encoder_spec),
                "label_names": self.label_names,
                "optimizer": settings_dict(self.optimizer_spec),
                "scheduler": [settings_dict(phase) for phase in self.scheduler_phases],
                "freeze": settings_dict(self.freeze_policy),
                "native_checkpoint": native_checkpoint,
                "metrics": {
                    alias: settings_dict(parse_settings(ComponentSpec, spec))
                    for alias, spec in (metrics or {}).items()
                },
                "tracked_metric": self.objective.metric,
                "tracked_mode": self.objective.mode,
            }
        )
        self._training_loss = torch.zeros(())
        self._training_count = 0
        self._validation_loss = torch.zeros(())
        self._validation_targets: list[torch.Tensor] = []
        self._validation_predictions: list[torch.Tensor] = []
        self.train()

    def _build_head(self, head: str) -> nn.Linear:
        """Create the new final layer; ``full`` first renews the post-pooling part."""
        if head == "full":
            renew = getattr(self.encoder, "reinitialize_post_pooling", None)
            if not callable(renew):
                raise ValueError(
                    f"{type(self.encoder).__name__} cannot renew its post-pooling "
                    "part: it has no reinitialize_post_pooling method"
                )
            renew()
        elif head != "linear":
            raise ValueError(f"Unsupported head: {head}")
        return nn.Linear(self.encoder.feature_dim, len(self.label_names))

    def train(self, mode: bool = True) -> "ClassifierTask":
        """Set module modes, applying the encoder's BatchNorm policy during training."""
        super().train(mode)
        if mode:
            self.freeze_policy.apply_batchnorm(self.encoder)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for RGB images in [0, 1]."""
        return self.head(self.encoder(images))

    def native_model(self) -> nn.Module:
        """Return the encoder's native model carrying this task's head and labels."""
        return self.encoder.to_native(self.head, self.label_names)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the batch's cross-entropy while accumulating the epoch's sum."""
        labels, images = batch
        loss = F.cross_entropy(self(images), labels)
        self._training_loss = self._training_loss.to(loss.device) + loss.detach() * len(
            labels
        )
        self._training_count += len(labels)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Accumulate summed cross-entropy and predictions of a validation batch."""
        labels, images = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self._validation_loss = self._validation_loss.to(loss.device) + (
            loss.detach() * len(labels)
        )
        self._validation_targets.append(labels.detach().cpu())
        self._validation_predictions.append(logits.argmax(dim=1).detach().cpu())
        return loss

    def on_train_epoch_start(self) -> None:
        """Start each epoch's training mean from zero.

        With validation, the mean is logged by the validation hook, which does not run
        in every epoch when ``check_val_every_n_epoch`` or ``val_check_interval`` says
        so. Resetting here makes a logged mean cover only its own epoch's batches; an
        epoch without validation logs no train mean.
        """
        self._training_loss = torch.zeros(())
        self._training_count = 0

    def on_train_epoch_end(self) -> None:
        """Log the training mean as the objective; with validation it is logged there."""
        if self.trainer.enable_validation:
            return  # logged with the validation metrics, which keeps one CSV row per epoch
        self._log_training_mean(CE_METRIC, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log the whole-split mean loss, the configured scores and the epoch's training mean.

        Validation runs after the epoch's training batches and before its
        ``on_train_epoch_end``. Logging everything in this one hook makes the CSV
        logger write a single complete row per epoch.
        """
        targets = torch.cat(self._validation_targets)
        predictions = torch.cat(self._validation_predictions)
        mean = self._validation_loss / len(targets)
        self._validation_loss = torch.zeros(())
        self._validation_targets.clear()
        self._validation_predictions.clear()
        self.log(CE_METRIC, mean, prog_bar=True, batch_size=len(targets))
        scores = self.metrics.score(targets.numpy(), np.asarray(predictions.numpy()))
        self.log_dict(scores, batch_size=len(targets))
        if not self.trainer.sanity_checking:  # no training has happened yet then
            self._log_training_mean(f"train_{CE_METRIC}", prog_bar=False)

    def _log_training_mean(self, name: str, *, prog_bar: bool) -> None:
        """Log the epoch's mean training loss and reset the accumulators."""
        count = self._training_count
        mean = self._training_loss / count
        self._training_loss = torch.zeros(())
        self._training_count = 0
        self.log(name, mean, prog_bar=prog_bar, batch_size=count)

    def configure_optimizers(self):
        """Build the optimizer and scheduler for the task's trainable parameters."""
        parameters = (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        return configure_optimizers(
            parameters, self.optimizer_spec, self.scheduler_phases, self.objective
        )
