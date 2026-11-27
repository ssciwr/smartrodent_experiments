"""Build encoder modules and configure policies on their named parameters."""

import sys
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal, Protocol

import torch
from pydantic import AfterValidator, ConfigDict
from pydantic.dataclasses import dataclass
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from smartrodent.components import ComponentSpec
from smartrodent.training_validation import Nonblank, unique
from smartrodent.yolo_encoder import YoloEncoder

# YoloEncoder is re-exported on purpose: build_encoder resolves short encoder
# names such as "YoloEncoder" against this module's namespace.
__all__ = [
    "Encoder",
    "FreezePolicy",
    "ParameterPrefixes",
    "Reinitialization",
    "YoloEncoder",
    "build_encoder",
    "encoder_checkpoint_spec",
    "preflight_encoders",
]

type ParameterPrefixes = Annotated[
    Sequence[Nonblank], AfterValidator(unique), AfterValidator(tuple)
]


class Encoder(Protocol):
    """Feature interface implemented by a plain ``torch.nn.Module``.

    Attributes:
        feature_dim: Width of the feature vector for one image.
    """

    feature_dim: int

    def forward(self, images: Tensor) -> Tensor:
        """Encode RGB images in [0, 1], applying any normalization internally.

        Args:
            images: Float RGB tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tensor of shape ``(B, feature_dim)``.
        """
        ...


def build_encoder(spec: ComponentSpec) -> nn.Module:
    """Build a configured encoder from its class or factory.

    Args:
        spec: Encoder class or factory with its constructor arguments.

    Returns:
        PyTorch module implementing the encoder interface.

    Raises:
        ValueError: The spec fails to build or produces an invalid encoder.
    """
    encoder = spec.build(sys.modules[__name__])
    feature_dim = getattr(encoder, "feature_dim", None)
    if (
        not isinstance(encoder, nn.Module)
        or type(feature_dim) is not int
        or feature_dim <= 0
    ):
        raise ValueError(
            f"Encoder {spec.name!r} must be an nn.Module with a positive feature_dim"
        )
    return encoder


def encoder_checkpoint_spec(spec: ComponentSpec, encoder: nn.Module) -> ComponentSpec:
    """Describe how to rebuild an encoder before loading its checkpoint state.

    Args:
        spec: Encoder spec used to construct the module.
        encoder: Built module. Encoders initialized from external weights can
            provide ``checkpoint_parameters()`` to rebuild from architecture alone.

    Returns:
        Component spec whose parameters rebuild the encoder for restoration.
    """
    parameters = getattr(encoder, "checkpoint_parameters", None)
    if callable(parameters):
        return ComponentSpec(name=spec.name, parameters=parameters())
    return spec


def _has_prefix(name: str, prefix: str) -> bool:
    """Match a complete module path or parameter name."""
    return name == prefix or name.startswith(f"{prefix}.")


def _matching_parameters(
    encoder: nn.Module, prefixes: ParameterPrefixes
) -> dict[str, nn.Parameter]:
    """Resolve every prefix before the caller changes any parameters."""
    parameters = dict(encoder.named_parameters())
    matched = {}
    for prefix in prefixes:
        selected = {
            name: parameter
            for name, parameter in parameters.items()
            if _has_prefix(name, prefix)
        }
        if not selected:
            raise ValueError(
                f"Encoder parameter prefix {prefix!r} matches no parameters"
            )
        matched.update(selected)
    return matched


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FreezePolicy:
    """Select encoder parameters to freeze and control frozen BatchNorm.

    Args:
        prefixes: Dotted module paths or complete parameter names.
        batchnorm: Whether frozen BatchNorm statistics stay fixed or adapt.
    """

    prefixes: ParameterPrefixes = ()
    batchnorm: Literal["eval", "adapt"] = "eval"

    def validate(self, encoder: nn.Module) -> None:
        """Check that every prefix selects at least one encoder parameter.

        Args:
            encoder: Candidate module to check before training.

        Raises:
            ValueError: A prefix matches no parameter.
        """
        _matching_parameters(encoder, self.prefixes)

    def apply(self, encoder: nn.Module) -> None:
        """Disable gradients for the selected encoder parameters.

        Args:
            encoder: Module whose selected parameters will be frozen.

        Raises:
            ValueError: A prefix matches no parameter.
        """
        for parameter in _matching_parameters(encoder, self.prefixes).values():
            parameter.requires_grad_(False)

    def apply_batchnorm(self, encoder: nn.Module) -> None:
        """Keep frozen BatchNorm statistics fixed when training in ``eval`` mode.

        Args:
            encoder: Module after its training mode has been set.
        """
        if not self.prefixes or not encoder.training or self.batchnorm == "adapt":
            return
        for name, module in encoder.named_modules():
            if isinstance(module, _BatchNorm) and any(
                _has_prefix(name, prefix) for prefix in self.prefixes
            ):
                module.eval()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Reinitialization:
    """Select encoder modules to redraw randomly before classifier training.

    Args:
        prefixes: Dotted module paths or complete parameter names.
        source: Only ``random`` exists: the matched modules reset themselves.
    """

    prefixes: ParameterPrefixes
    source: Literal["random"]

    def validate(self, encoder: nn.Module) -> None:
        """Check that every prefix selects at least one encoder parameter.

        Args:
            encoder: Candidate module to check before training.

        Raises:
            ValueError: A prefix matches no parameter.
        """
        _matching_parameters(encoder, self.prefixes)

    def apply(self, encoder: nn.Module) -> None:
        """Reset the matched modules; parameters outside the prefixes stay intact.

        Args:
            encoder: Module whose matched modules are redrawn.

        Raises:
            ValueError: A prefix matches no parameter, or a matched module has no
                ``reset_parameters``.
        """
        matched = _matching_parameters(encoder, self.prefixes)
        # A lone parameter has no initializer, so whole owning modules are reset.
        owners = {name.rpartition(".")[0] for name in matched}
        for name, module in encoder.named_modules():
            if name not in owners:
                continue
            reset = getattr(module, "reset_parameters", None)
            if not callable(reset):
                raise ValueError(f"Module {name!r} cannot be reinitialized randomly")
            # The reset redraws every parameter of the owner, including siblings of a
            # requested parameter name (``fc.weight`` must not touch ``fc.bias``).
            spared = {
                key: parameter.detach().clone()
                for key, parameter in module.named_parameters()
                if f"{name}.{key}".removeprefix(".") not in matched
            }
            reset()
            with torch.no_grad():
                for key, parameter in module.named_parameters():
                    if key in spared:
                        parameter.copy_(spared[key])


def preflight_encoders(
    encoders: Iterable[nn.Module],
    *,
    freeze: FreezePolicy,
    reinitialization: Reinitialization | None,
    native: bool,
    full_head: bool = False,
) -> None:
    """Check a classifier's encoder policies against every candidate encoder.

    Args:
        encoders: Every encoder the classifier stage may receive.
        freeze: Freeze policy of the classifier stage.
        reinitialization: Optional reinitialization of the classifier stage.
        native: Whether the classifier stage requests a native export.
        full_head: Whether the classifier stage renews the encoder's post-pooling part.

    Raises:
        ValueError: A policy does not fit one of the encoders.
    """
    for encoder in encoders:
        freeze.validate(encoder)
        if reinitialization is not None:
            reinitialization.validate(encoder)
        if full_head and not callable(
            getattr(encoder, "reinitialize_post_pooling", None)
        ):
            raise ValueError(
                f"{type(encoder).__name__} cannot renew its post-pooling part: "
                "it has no reinitialize_post_pooling method"
            )
        if native and not callable(getattr(encoder, "to_native", None)):
            raise ValueError(
                f"{type(encoder).__name__} cannot export a native checkpoint: "
                "it has no to_native method"
            )
