"""Configurable supervised-contrastive extensions for YOLO classification."""

import math
import os
from dataclasses import dataclass
from inspect import signature
from numbers import Real
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch import nn
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.nn.modules.head import Classify
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import unwrap_model


@dataclass(frozen=True)
class SupConClassificationOutput:
    """Hold the two training outputs produced by a SupCon classifier head.

    Attributes:
        logits: Unnormalized class scores used by cross-entropy.
        embeddings: Projected representations used by supervised contrastive loss.
    """

    logits: torch.Tensor
    embeddings: torch.Tensor


@dataclass(frozen=True)
class SupConTrainerConfiguration:
    """Configure P-K batches and the supervised-contrastive projection head.

    Attributes:
        enabled: Whether supervised-contrastive training is enabled.
        classes_per_batch: Number of distinct classes sampled in every batch.
        samples_per_class: Number of examples sampled for each selected class.
        epoch_size: Number of sampled examples in one training epoch. ``None``
            delegates the choice to the trainer.
        projection_hidden_dimension: Width of the projection MLP hidden layer.
        embedding_dimension: Width of the final contrastive embedding.
        temperature: Temperature used by the supervised-contrastive loss.
        cross_entropy_weight: Multiplier for weighted cross-entropy.
        contrastive_weight: Multiplier for supervised-contrastive loss.
    """

    enabled: bool = False
    classes_per_batch: int = 8
    samples_per_class: int = 4
    epoch_size: int | None = None
    projection_hidden_dimension: int = 512
    embedding_dimension: int = 128
    temperature: float = 0.1
    cross_entropy_weight: float = 1.0
    contrastive_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate values that would otherwise fail late inside training."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")

        self._require_integer_at_least(
            "classes_per_batch", self.classes_per_batch, minimum=2
        )
        self._require_integer_at_least(
            "samples_per_class", self.samples_per_class, minimum=2
        )
        self._require_integer_at_least(
            "projection_hidden_dimension",
            self.projection_hidden_dimension,
            minimum=1,
        )
        self._require_integer_at_least(
            "embedding_dimension", self.embedding_dimension, minimum=1
        )

        if self.epoch_size is not None:
            self._require_integer_at_least("epoch_size", self.epoch_size, minimum=1)
            if self.epoch_size % self.batch_size != 0:
                raise ValueError(
                    "epoch_size must be divisible by the derived P-K batch size"
                )

        self._require_real("temperature", self.temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        self._require_real("cross_entropy_weight", self.cross_entropy_weight)
        self._require_real("contrastive_weight", self.contrastive_weight)
        if self.cross_entropy_weight < 0:
            raise ValueError("cross_entropy_weight must be non-negative")
        if self.contrastive_weight < 0:
            raise ValueError("contrastive_weight must be non-negative")
        if self.cross_entropy_weight == 0 and self.contrastive_weight == 0:
            raise ValueError("At least one loss weight must be greater than 0")

    @classmethod
    def from_mapping(
        cls, configuration: Mapping[str, Any] | None
    ) -> "SupConTrainerConfiguration":
        """Build a configuration from YAML-compatible values.

        Args:
            configuration: Optional ``supcon`` mapping from a trainer YAML file.

        Returns:
            A validated supervised-contrastive trainer configuration. A missing
            mapping produces a disabled configuration with default values.

        Raises:
            TypeError: The supplied configuration is not a mapping.
        """
        if configuration is None:
            return cls()
        if not isinstance(configuration, Mapping):
            raise TypeError("The 'supcon' configuration must be a mapping")
        return cls(**configuration)

    @property
    def batch_size(self) -> int:
        """Return the P-K batch size derived from the sampling configuration."""
        return self.classes_per_batch * self.samples_per_class

    @staticmethod
    def _require_integer_at_least(name: str, value: Any, minimum: int) -> None:
        """Require an integer configuration value at or above a lower bound."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            qualifier = "positive" if minimum == 1 else f"at least {minimum}"
            raise ValueError(f"{name} must be {qualifier}")

    @staticmethod
    def _require_real(name: str, value: Any) -> None:
        """Require a real numeric configuration value, excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")


class PKSamplerFactory:
    """Build a P-K sampler across supported pytorch-metric-learning versions."""

    def __init__(self, sampler_type: type = MPerClassSampler) -> None:
        """Initialize the factory with the installed or a compatible sampler type.

        Args:
            sampler_type: ``MPerClassSampler`` implementation to construct.
        """
        self.sampler_type = sampler_type

    def build(
        self,
        labels: list[int],
        configuration: SupConTrainerConfiguration,
        epoch_size: int,
        seed: int,
    ) -> Any:
        """Create a seeded sampler using the dependency's supported size keyword.

        Args:
            labels: Class index corresponding to every dataset index.
            configuration: P-K batch composition.
            epoch_size: Number of sample indices produced per epoch.
            seed: Random seed used when the sampler accepts a PyTorch generator.

        Returns:
            A configured ``MPerClassSampler``-compatible instance.

        Raises:
            RuntimeError: The sampler exposes neither known sample-count keyword.
        """
        parameters = signature(self.sampler_type).parameters
        sampler_arguments: dict[str, Any] = {
            "labels": labels,
            "m": configuration.samples_per_class,
            "batch_size": configuration.batch_size,
        }
        if "num_samples" in parameters:
            sampler_arguments["num_samples"] = epoch_size
        elif "length_before_new_iter" in parameters:
            sampler_arguments["length_before_new_iter"] = epoch_size
        else:
            raise RuntimeError(
                "Unsupported MPerClassSampler interface: expected num_samples or "
                "length_before_new_iter"
            )

        if "generator" in parameters:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler_arguments["generator"] = generator
        return self.sampler_type(**sampler_arguments)


class WeightedSupConLoss:
    """Combine class-weighted cross-entropy and supervised contrastive loss."""

    def __init__(
        self,
        configuration: SupConTrainerConfiguration,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """Initialize the joint classification objective.

        Args:
            configuration: Multipliers and temperature for the two loss terms.
            class_weights: Optional one-dimensional cross-entropy weight per class.
        """
        self.class_weights = (
            None if class_weights is None else class_weights.detach().clone()
        )
        self.cross_entropy_weight = configuration.cross_entropy_weight
        self.contrastive_weight = configuration.contrastive_weight
        self.supcon = SupConLoss(temperature=configuration.temperature)

    def __call__(
        self,
        predictions: SupConClassificationOutput | tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the differentiable joint loss and detached logging values.

        Args:
            predictions: Dual training output, or the standard evaluation tuple of
                probabilities and logits.
            batch: Ultralytics batch containing integer class labels under ``cls``.

        Returns:
            Joint loss and detached total, cross-entropy, and SupCon terms.

        Raises:
            TypeError: Predictions match neither supported output contract.
        """
        if isinstance(predictions, SupConClassificationOutput):
            logits = predictions.logits
            embeddings = predictions.embeddings
        elif (
            isinstance(predictions, tuple)
            and len(predictions) == 2
            and all(isinstance(value, torch.Tensor) for value in predictions)
        ):
            _, logits = predictions
            embeddings = None
        else:
            raise TypeError(
                "WeightedSupConLoss requires SupCon training output or standard "
                "classification validation predictions"
            )

        labels = batch["cls"]
        class_weights = (
            None
            if self.class_weights is None
            else self.class_weights.to(device=logits.device, dtype=logits.dtype)
        )
        cross_entropy = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            reduction="mean",
        )
        contrastive = (
            self.supcon(embeddings, labels)
            if embeddings is not None
            else logits.new_zeros(())
        )
        loss = (
            self.cross_entropy_weight * cross_entropy
            + self.contrastive_weight * contrastive
        )
        return loss, {
            "loss": loss.detach(),
            "ce_loss": cross_entropy.detach(),
            "supcon_loss": contrastive.detach(),
        }


class SupConClassificationHead(Classify):
    """Add a contrastive projection branch to a pretrained YOLO classifier head.

    The existing convolution, pooling, dropout, and linear classifier modules are
    reused rather than recreated, preserving their pretrained parameters. During
    training, the module returns class logits and projected embeddings. Evaluation
    follows the normal Ultralytics ``Classify`` return contract.
    """

    def __init__(
        self,
        source_head: Classify,
        configuration: SupConTrainerConfiguration,
    ) -> None:
        """Initialize the dual head from an existing YOLO classification head.

        Args:
            source_head: Pretrained Ultralytics classification head to adapt.
            configuration: Projection-head dimensions and SupCon settings.

        Raises:
            TypeError: ``source_head`` is not an Ultralytics ``Classify`` module.
        """
        if not isinstance(source_head, Classify):
            raise TypeError("source_head must be an Ultralytics Classify module")
        nn.Module.__init__(self)
        self.conv = source_head.conv
        self.pool = source_head.pool
        self.drop = source_head.drop
        self.linear = source_head.linear
        self.projector = nn.Sequential(
            nn.Linear(
                self.linear.in_features,
                configuration.projection_hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                configuration.projection_hidden_dimension,
                configuration.embedding_dimension,
            ),
        )
        self.export = source_head.export
        if hasattr(source_head, "i"):
            self.i = source_head.i
        if hasattr(source_head, "f"):
            self.f = source_head.f
        self.type = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    @classmethod
    def from_head(
        cls,
        source_head: Classify,
        configuration: SupConTrainerConfiguration,
    ) -> "SupConClassificationHead":
        """Create a dual head while retaining the source classifier parameters.

        Args:
            source_head: Existing Ultralytics ``Classify`` module.
            configuration: Projection-head dimensions and SupCon settings.

        Returns:
            A classification head with classifier and projection branches.
        """
        return cls(source_head, configuration)

    def forward(
        self, x: list[torch.Tensor] | torch.Tensor
    ) -> SupConClassificationOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return dual training outputs or the standard evaluation output.

        Args:
            x: Feature map, or feature maps to concatenate channel-wise.

        Returns:
            Logits and projected embeddings during training. During evaluation,
            returns probabilities for export or ``(probabilities, logits)`` for the
            normal Ultralytics inference path.
        """
        if isinstance(x, list):
            x = torch.cat(x, 1)
        shared_features = self.drop(self.pool(self.conv(x)).flatten(1))
        logits = self.linear(shared_features)
        if self.training:
            return SupConClassificationOutput(
                logits=logits,
                embeddings=self.projector(shared_features),
            )

        probabilities = logits.softmax(1)
        if self.export:
            return probabilities
        return probabilities, logits


class YoloSupConClassificationTrainer(ClassificationTrainer):
    """Train a YOLO classifier with exact P-K batches for SupCon learning."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ) -> None:
        """Initialize the trainer and remove SupCon settings before YOLO parsing.

        Args:
            cfg: Ultralytics default configuration or compatible overrides.
            overrides: Training overrides containing a required ``supcon`` mapping.
            _callbacks: Ultralytics callback registry.

        Raises:
            ValueError: SupCon is disabled or an explicit batch size conflicts with
                the configured P-K batch.
        """
        parsed_overrides = dict(overrides or {})
        self.supcon_configuration = SupConTrainerConfiguration.from_mapping(
            parsed_overrides.pop("supcon", None)
        )
        if not self.supcon_configuration.enabled:
            raise ValueError(
                "YoloSupConClassificationTrainer requires supcon.enabled=true"
            )

        configured_batch = parsed_overrides.get("batch")
        if (
            configured_batch is not None
            and configured_batch != self.supcon_configuration.batch_size
        ):
            raise ValueError(
                "train batch must equal supcon classes_per_batch * samples_per_class "
                f"({self.supcon_configuration.batch_size})"
            )
        parsed_overrides["batch"] = self.supcon_configuration.batch_size
        super().__init__(cfg, parsed_overrides, _callbacks)

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: nn.Module | None = None,
        verbose: bool = True,
    ) -> nn.Module:
        """Build the standard classifier and add its SupCon projection branch.

        Args:
            cfg: YOLO model configuration.
            weights: Optional pretrained classifier whose parameters are loaded.
            verbose: Whether Ultralytics prints the model summary.

        Returns:
            A YOLO classification model with a dual-output training head.
        """
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.model[-1] = SupConClassificationHead.from_head(
            model.model[-1],
            self.supcon_configuration,
        )
        return model

    def set_class_weights(self) -> None:
        """Compute original-split class weights and install the joint criterion."""
        super().set_class_weights()
        model = unwrap_model(self.model)
        criterion_model = (
            model.student_model if hasattr(model, "student_model") else model
        )
        class_weights = getattr(criterion_model, "class_weights", None)
        criterion_model.criterion = WeightedSupConLoss(
            self.supcon_configuration,
            class_weights=class_weights,
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> InfiniteDataLoader:
        """Build an exact P-K training loader or the stock evaluation loader.

        Args:
            dataset_path: Directory containing class-specific image folders.
            batch_size: Batch size requested by the Ultralytics training loop.
            rank: Distributed process rank, or ``-1`` for single-device execution.
            mode: Dataset mode: ``train``, ``val``, or ``test``.

        Returns:
            An Ultralytics infinite dataloader.

        Raises:
            RuntimeError: Distributed P-K sampling is requested.
            ValueError: The requested batch size or represented class count is
                incompatible with the SupCon configuration.
        """
        if mode == "train":
            return self._build_pk_dataloader(dataset_path, batch_size, rank)
        if mode in {"val", "test"}:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        raise ValueError(f"Unsupported classification dataloader mode: {mode}")

    def _build_pk_dataloader(
        self,
        dataset_path: str,
        batch_size: int,
        rank: int,
    ) -> InfiniteDataLoader:
        """Build the single-device training loader backed by MPerClassSampler."""
        if rank != -1:
            raise RuntimeError(
                "P-K sampling is currently supported only for single-device training"
            )
        if batch_size != self.supcon_configuration.batch_size:
            raise ValueError(
                "Ultralytics requested batch size "
                f"{batch_size}, expected P-K batch size "
                f"{self.supcon_configuration.batch_size}"
            )

        dataset = self.build_dataset(dataset_path, mode="train")
        if not dataset.samples:
            raise FileNotFoundError(
                f"No images found in 'train' split of {dataset_path}. "
                "See https://docs.ultralytics.com/datasets/classify for the "
                "classification dataset format."
            )

        labels = [sample[1] for sample in dataset.samples]
        represented_classes = len(set(labels))
        if represented_classes < self.supcon_configuration.classes_per_batch:
            raise ValueError(
                "Training split contains "
                f"{represented_classes} represented classes, but P-K sampling "
                f"requires {self.supcon_configuration.classes_per_batch}"
            )

        epoch_size = self.supcon_configuration.epoch_size
        if epoch_size is None:
            epoch_size = math.ceil(len(dataset) / batch_size) * batch_size
        sampler = PKSamplerFactory().build(
            labels=labels,
            configuration=self.supcon_configuration,
            epoch_size=epoch_size,
            seed=self.args.seed,
        )

        batch_count = epoch_size // batch_size
        available_cpus = os.cpu_count() or 1
        worker_count = min(
            available_cpus,
            self.args.workers,
            0 if batch_count <= 1 else batch_count,
        )
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        pin_memory = self.device.type not in {"cpu", "mps"}
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            sampler=sampler,
            prefetch_factor=2 if worker_count > 0 else None,
            pin_memory=pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )


def build_supcon_trainer(
    configuration: SupConTrainerConfiguration,
) -> type[YoloSupConClassificationTrainer] | None:
    """Select the Ultralytics trainer for a SupCon configuration.

    Args:
        configuration: Validated supervised-contrastive configuration.

    Returns:
        The custom trainer type when SupCon is enabled, otherwise ``None`` so
        Ultralytics retains its stock classification trainer selection.
    """
    if configuration.enabled:
        return YoloSupConClassificationTrainer
    return None
