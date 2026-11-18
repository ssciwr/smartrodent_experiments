"""Initialize portable YOLO CE classifiers with an explicit encoder freeze policy."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from pickle import UnpicklingError

import torch
from ultralytics.nn.modules import Classify, Conv, ConvTranspose, RepConv, RepVGGDW
from ultralytics.nn.tasks import ClassificationModel, load_checkpoint

from smartrodent.supcon_model import YoloSupConModel


class YoloClassifierModel(torch.nn.Module):
    """Wrap a YOLO classifier, returning raw logits in training and evaluation.

    Retained native layers must be unfused so the embedded architecture can
    reconstruct them. Fusion in discarded source or donor layers is harmless.

    Args:
        network: Native YOLO classification network with its initialized head.
        label_names: Alphabetically assigned class-name-to-ID mapping.
        head_replace: ``linear`` or ``full`` classifier boundary.
        freeze_encoder: Whether to disable encoder parameter updates.
        encoder_batchnorm: ``eval`` for stored statistics or ``adapt`` for updates.
    """

    def __init__(
        self,
        network: ClassificationModel,
        label_names: Mapping[str, int],
        *,
        head_replace: str = "linear",
        freeze_encoder: bool = True,
        encoder_batchnorm: str = "eval",
    ):
        """Own the network without registering duplicate encoder/head aliases."""
        super().__init__()
        self._validate_policy(head_replace, freeze_encoder, encoder_batchnorm)
        head = self._classification_head(network)
        expected_labels = self._label_mapping(tuple(label_names))
        if dict(label_names) != expected_labels:
            raise ValueError("Classifier label_names must use alphabetical contiguous IDs")
        if head.linear.out_features != len(expected_labels):
            raise ValueError("Classifier head output shape does not match class labels")
        self._reject_fused(network)
        for module in self._convolutions(network).values():
            self._activation_configuration(module.act)
        self.network = network
        self.label_names = expected_labels
        self.head_replace = head_replace
        self.freeze_encoder = freeze_encoder
        self.encoder_batchnorm = encoder_batchnorm
        self.network.names = {index: name for name, index in self.label_names.items()}
        self.network.yaml["nc"] = len(self.label_names)
        # Export-mode Classify returns only probabilities. Keep the normal
        # (probabilities, logits) evaluation contract without enabling head BN.
        head.export = False
        self.network.requires_grad_(not freeze_encoder)
        self._classifier.requires_grad_(True)
        self.eval()

    @classmethod
    def initialize(
        cls,
        checkpoint: str | Path,
        *,
        source_format: str,
        class_names: Sequence[str],
        head_replace: str = "linear",
        head_source: str = "random",
        head_checkpoint: str | Path | None = None,
        freeze_encoder: bool = True,
        encoder_batchnorm: str = "eval",
        device: str | torch.device = "cpu",
    ) -> "YoloClassifierModel":
        """Start a new classifier from an explicitly identified encoder checkpoint.

        Args:
            checkpoint: Local SupCon, Ultralytics, or standalone CE checkpoint.
            source_format: ``supcon``, ``ultralytics``, or ``classifier``.
            class_names: Target classes, sorted to assign dataset-compatible IDs.
            head_replace: Replace ``linear`` only or the ``full`` Classify module.
            head_source: Fresh ``random`` weights or a native ``yolo_checkpoint``.
            head_checkpoint: Native head donor, required only for checkpoint weights.
            freeze_encoder: Keep all parameters before the selected boundary fixed.
            encoder_batchnorm: Use stored statistics (``eval``) or adapt them.
            device: Device on which the initialized classifier should run.

        Returns:
            A newly initialized classifier in evaluation mode. No optimizer or
            scheduler state is resumed from either source checkpoint.

        Raises:
            ValueError: Options, checkpoint format, head dimensions, or labels
                are incompatible. Native checkpoints must be trusted files.
            FileNotFoundError: A requested local source does not exist.
        """
        cls._validate_policy(head_replace, freeze_encoder, encoder_batchnorm)
        labels = cls._label_mapping(class_names)
        if source_format not in ("supcon", "ultralytics", "classifier"):
            raise ValueError("source_format must be 'supcon', 'ultralytics', or 'classifier'")
        if head_source not in ("random", "yolo_checkpoint"):
            raise ValueError("head_source must be 'random' or 'yolo_checkpoint'")
        if head_source == "yolo_checkpoint" and head_checkpoint is None:
            raise ValueError("head_checkpoint is required for head_source='yolo_checkpoint'")
        if head_source == "random" and head_checkpoint is not None:
            raise ValueError("head_checkpoint is not allowed for head_source='random'")
        path = cls._local_checkpoint(checkpoint)
        if source_format == "supcon":
            with cls._preserve_activation_default():
                try:
                    source = YoloSupConModel.from_checkpoint(path)
                except UnpicklingError as error:
                    raise ValueError("Expected a standalone SupCon checkpoint format") from error
            # Removing the pooling hook releases its closure over the SupCon
            # wrapper. Retain only the native network, not the projector/optimizer.
            network = source.finalize().model
        elif source_format == "ultralytics":
            network = cls._load_native(path)
        else:
            network = cls.from_checkpoint(path).network
        original = cls._classification_head(network)
        donor = None
        if head_source == "yolo_checkpoint":
            donor_network = cls._load_native(cls._local_checkpoint(head_checkpoint))
            expected_names = {index: name for name, index in labels.items()}
            if donor_network.names != expected_names:
                raise ValueError("Head checkpoint class identities and label order must match")
            donor = cls._classification_head(donor_network)
            cls._validate_donor(original, donor, head_replace, len(labels))
        if head_replace == "linear":
            # reshape_outputs() deliberately preserves same-size layers; a new
            # CE experiment must reset this layer even when the class count matches.
            original.linear = (
                torch.nn.Linear(original.linear.in_features, len(labels))
                if donor is None else deepcopy(donor.linear)
            )
        elif head_replace == "full":
            replacement = (
                Classify(original.conv.conv.in_channels, len(labels))
                if donor is None else deepcopy(donor)
            )
            # Prediction follows the retained model's graph, not the donor graph.
            replacement.i = original.i
            replacement.f = original.f
            replacement.type = original.type
            replacement.np = sum(parameter.numel() for parameter in replacement.parameters())
            network.model[-1] = replacement
        return cls(
            network, labels, head_replace=head_replace,
            freeze_encoder=freeze_encoder, encoder_batchnorm=encoder_batchnorm,
        ).to(device)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "YoloClassifierModel":
        """Restore a complete standalone classifier without replacing trained weights.

        Args:
            path: Checkpoint containing ``checkpoint_state()`` model metadata.
            device: Device for the restored classifier.

        Returns:
            A classifier in evaluation mode with its stored labels and freeze policy.

        Raises:
            ValueError: The payload is not a standalone classifier checkpoint.
            FileNotFoundError: The local checkpoint does not exist.
        """
        try:
            payload = torch.load(cls._local_checkpoint(path), map_location="cpu", weights_only=True)
        except UnpicklingError as error:
            raise ValueError("Expected a standalone classifier checkpoint") from error
        required = {"architecture", "model_configuration", "model_state_dict", "label_names"}
        if (
            not isinstance(payload, Mapping)
            or payload.get("checkpoint_type") != "classifier"
            or not required.issubset(payload)
        ):
            raise ValueError("Expected a standalone classifier checkpoint")
        configuration = dict(payload["model_configuration"])
        activations = configuration.pop("conv_activations")
        batchnorm = configuration.pop("batchnorm")
        dropout = configuration.pop("dropout")
        architecture = deepcopy(payload["architecture"])
        # Passing the embedded dictionary avoids reopening yaml_file.
        with cls._preserve_activation_default():
            network = ClassificationModel(architecture, nc=len(payload["label_names"]), verbose=False)
        cls._restore_behavior(network, activations, batchnorm, dropout)
        model = cls(network, payload["label_names"], **configuration)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model.to(device).eval()

    def checkpoint_state(self) -> dict:
        """Export portable model state for a trainer's checkpoint payload.

        Returns:
            Model tensors, embedded architecture, labels, and freeze/head behavior.
            Like ``state_dict()``, tensors share model storage; serialize promptly
            or deepcopy the payload if retaining an independent in-memory snapshot.

        Raises:
            ValueError: The exposed network was fused after initialization.
        """
        self._reject_fused(self.network)
        architecture = deepcopy(self.network.yaml)
        head = self.network.model[-1]
        convolution = head.conv.conv
        # A full donor head may differ from the source head's geometry. Encode
        # the actual Classify arguments rather than retaining stale source YAML.
        architecture["head"][-1][3] = [
            len(self.label_names), list(convolution.kernel_size), list(convolution.stride),
            list(convolution.padding), convolution.groups,
        ]
        return {
            "checkpoint_type": "classifier",
            "architecture": architecture,
            "model_configuration": {
                "head_replace": self.head_replace,
                "freeze_encoder": self.freeze_encoder,
                "encoder_batchnorm": self.encoder_batchnorm,
                **self._behavior_configuration(),
            },
            "model_state_dict": self.state_dict(),
            "label_names": dict(self.label_names),
        }

    def train(self, mode: bool = True) -> "YoloClassifierModel":
        """Set mode while preserving the configured encoder and BatchNorm policy.

        Args:
            mode: Whether the trainable classifier should use training behavior.

        Returns:
            This model. Evaluation puts every module in eval mode; training enables
            only the selected head plus permitted encoder behavior.
        """
        super().train(mode)
        if not mode:
            return self
        if self.freeze_encoder:
            self.network.eval()
        classifier = self._classifier
        classifier.train(True)
        classifier_modules = {id(module) for module in classifier.modules()}
        for module in self._batchnorm_modules(self.network).values():
            if id(module) not in classifier_modules:
                module.train(self.encoder_batchnorm == "adapt")
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits without changing any training modes.

        Args:
            images: Image batch on the same device as the model.

        Returns:
            A batch-by-class tensor of unnormalized logits suitable for CE loss.
        """
        output = self.network.predict(images)
        return output[1] if isinstance(output, tuple) else output

    @property
    def _classifier(self) -> torch.nn.Module:
        """Locate the selected trainable component without registering another alias."""
        head = self.network.model[-1]
        return head.linear if self.head_replace == "linear" else head

    def _behavior_configuration(self) -> dict:
        """Capture known runtime options omitted by YAML and state_dict, not arbitrary modules."""
        return {
            "conv_activations": {
                name: self._activation_configuration(module.act)
                for name, module in self._convolutions(self.network).items()
            },
            "batchnorm": {
                name: {"eps": module.eps, "momentum": module.momentum}
                for name, module in self._batchnorm_modules(self.network).items()
            },
            "dropout": {
                name: {"p": module.p, "inplace": module.inplace}
                for name, module in self._dropout_modules(self.network).items()
            },
        }

    @staticmethod
    @contextmanager
    def _preserve_activation_default() -> Iterator[None]:
        """Prevent Ultralytics YAML parsing from changing the next model's activation default."""
        default_activation = Conv.default_act
        try:
            yield
        finally:
            Conv.default_act = default_activation

    @staticmethod
    def _convolutions(network: ClassificationModel) -> dict[str, torch.nn.Module]:
        """Identify the native YOLO layers whose activations YAML may not describe."""
        return {
            name: module for name, module in network.named_modules()
            if isinstance(module, (Conv, ConvTranspose))
        }

    @staticmethod
    def _batchnorm_modules(network: ClassificationModel) -> dict[str, torch.nn.Module]:
        """Identify BatchNorm behavior separately from its parameters and buffers."""
        types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
        return {name: module for name, module in network.named_modules() if isinstance(module, types)}

    @staticmethod
    def _dropout_modules(network: ClassificationModel) -> dict[str, torch.nn.Module]:
        """Identify dropout options that training may change independently of YAML."""
        types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)
        return {name: module for name, module in network.named_modules() if isinstance(module, types)}

    @staticmethod
    def _activation_specs() -> dict:
        """Allow only known stateless activations and their primitive constructor options."""
        return {
            "SiLU": (torch.nn.SiLU, ("inplace",)),
            "ReLU": (torch.nn.ReLU, ("inplace",)),
            "ReLU6": (torch.nn.ReLU6, ("inplace",)),
            "LeakyReLU": (torch.nn.LeakyReLU, ("negative_slope", "inplace")),
            "Hardswish": (torch.nn.Hardswish, ("inplace",)),
            "GELU": (torch.nn.GELU, ("approximate",)),
            "Identity": (torch.nn.Identity, ()),
        }

    @classmethod
    def _activation_configuration(cls, activation: torch.nn.Module) -> dict:
        """Export safe activation metadata, refusing unknown classes or subclasses."""
        name = type(activation).__name__
        activation_type, options = cls._activation_specs().get(name, (None, ()))
        if type(activation) is not activation_type:
            raise ValueError(f"Unsupported YOLO convolution activation: {name}")
        return {"name": name, "parameters": {option: getattr(activation, option) for option in options}}

    @classmethod
    def _restore_activation(cls, configuration: Mapping) -> torch.nn.Module:
        """Construct only an allowlisted activation; never import/eval a checkpoint class name."""
        if not isinstance(configuration, Mapping) or set(configuration) != {"name", "parameters"}:
            raise ValueError("Invalid activation configuration")
        specification = cls._activation_specs().get(configuration["name"])
        if specification is None:
            raise ValueError("Unsupported activation configuration")
        activation_type, options = specification
        parameters = configuration["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != set(options):
            raise ValueError("Invalid activation parameters")
        return activation_type(**parameters)

    @classmethod
    def _restore_behavior(cls, network, activations, batchnorm, dropout) -> None:
        """Restore known non-tensor behavior on the already reconstructed layer graph."""
        convolutions = cls._convolutions(network)
        normalizations = cls._batchnorm_modules(network)
        dropouts = cls._dropout_modules(network)
        if (
            set(activations) != set(convolutions)
            or set(batchnorm) != set(normalizations)
            or set(dropout) != set(dropouts)
        ):
            raise ValueError("Classifier behavior metadata does not match the architecture")
        for name, module in convolutions.items():
            module.act = cls._restore_activation(activations[name])
        for name, module in normalizations.items():
            module.eps = batchnorm[name]["eps"]
            module.momentum = batchnorm[name]["momentum"]
        for name, module in dropouts.items():
            module.p = dropout[name]["p"]
            module.inplace = dropout[name]["inplace"]

    @staticmethod
    def _validate_policy(head_replace: str, freeze_encoder: bool, encoder_batchnorm: str) -> None:
        """Reject ambiguous replacement boundaries or freeze behavior before loading."""
        if head_replace not in ("linear", "full"):
            raise ValueError("head_replace must be 'linear' or 'full'")
        if not isinstance(freeze_encoder, bool):
            raise ValueError("freeze_encoder must be a boolean")
        if encoder_batchnorm not in ("eval", "adapt"):
            raise ValueError("encoder_batchnorm must be 'eval' or 'adapt'")

    @staticmethod
    def _label_mapping(class_names: Sequence[str]) -> dict[str, int]:
        """Match ImageDataset's alphabetical IDs without silently normalizing identities."""
        if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence) or not class_names:
            raise ValueError("class_names must be a nonempty sequence")
        if any(not isinstance(name, str) or not name for name in class_names):
            raise ValueError("class_names must contain nonempty strings")
        if len(set(class_names)) != len(class_names):
            raise ValueError("class_names must be unique")
        return {name: index for index, name in enumerate(sorted(class_names))}

    @staticmethod
    def _local_checkpoint(path: str | Path) -> Path:
        """Require local files before calling loaders that may otherwise download weights."""
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local checkpoint does not exist: {checkpoint}")
        # Ultralytics treats a corrupt bare official asset name as a cache entry
        # to delete/re-download. An absolute path keeps local inputs read-only.
        return checkpoint.resolve()

    @staticmethod
    def _reject_fused(network: ClassificationModel) -> None:
        """Reject fused retained layers that embedded unfused YAML cannot reconstruct."""
        for module in network.modules():
            fused_conv = isinstance(module, (Conv, ConvTranspose)) and not hasattr(module, "bn")
            fused_rep = isinstance(module, (RepConv, RepVGGDW)) and not hasattr(module, "conv1")
            if fused_conv or fused_rep:
                raise ValueError("Fused layers are unsupported in portable CE classifiers; use unfused weights")

    @staticmethod
    def _classification_head(network: ClassificationModel) -> Classify:
        """Reject detection networks and other non-YOLO-classifier structures early."""
        if not isinstance(network, ClassificationModel) or not isinstance(network.model[-1], Classify):
            raise ValueError("Expected a YOLO classification model with a Classify head")
        return network.model[-1]

    @classmethod
    def _load_native(cls, path: Path) -> ClassificationModel:
        """Load only an explicitly requested native checkpoint, without fusing BatchNorm."""
        try:
            network, _ = load_checkpoint(str(path), device="cpu", fuse=False)
        except TypeError as error:
            raise ValueError("Expected an Ultralytics classification checkpoint format") from error
        cls._classification_head(network)
        return network

    @staticmethod
    def _validate_donor(original: Classify, donor: Classify, boundary: str, num_classes: int) -> None:
        """Validate the transferred component's interface, not unrelated donor backbone layers."""
        if donor.linear.out_features != num_classes:
            raise ValueError("Head checkpoint output shape is incompatible with target classes")
        if boundary == "linear" and donor.linear.in_features != original.linear.in_features:
            raise ValueError("Head checkpoint linear input features are incompatible")
        if boundary == "full" and donor.conv.conv.in_channels != original.conv.conv.in_channels:
            raise ValueError("Head checkpoint Classify input channels are incompatible")
