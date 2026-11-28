"""Expose spatially pooled features from an Ultralytics classification model."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from ultralytics import YOLO
from ultralytics.nn.modules import Classify
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils.torch_utils import initialize_weights


class _PooledFeatures(nn.Module):
    """Reuse the classification head's convolution and pool in its graph position."""

    def __init__(self, head: Classify):
        """Keep only the modules and routing metadata needed to emit features."""
        super().__init__()
        self.conv = head.conv
        self.pool = head.pool
        self.f = head.f
        self.i = head.i

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the pre-classifier convolution and flatten the pooled features."""
        return self.pool(self.conv(inputs)).flatten(1)


class YoloEncoder(nn.Module):
    """Extract features from a YOLO classification backbone without hooks.

    Args:
        model_name: Ultralytics classification checkpoint or architecture YAML.
        architecture: Parsed architecture for restoration without the source file.
    """

    def __init__(
        self,
        model_name: str | Path | None = None,
        *,
        architecture: dict[str, Any] | None = None,
    ):
        super().__init__()
        if (model_name is None) == (architecture is None):
            raise ValueError("Provide exactly one YOLO model_name or architecture")
        if architecture is not None:
            model = ClassificationModel(cfg=deepcopy(architecture), verbose=False)
        else:
            model = YOLO(str(model_name), task="classify").model
        if not isinstance(model, ClassificationModel) or not isinstance(
            model.model[-1], Classify
        ):
            raise ValueError(
                "Expected a YOLO classification model with a Classify head"
            )

        head = model.model[-1]
        self.feature_dim = head.linear.in_features
        # Reusing Ultralytics' graph routing preserves models with skip connections.
        model.model[-1] = _PooledFeatures(head)
        # Saved Ultralytics checkpoints may come with every gradient disabled.
        self.network = model.requires_grad_(True)

    def checkpoint_parameters(self) -> dict[str, Any]:
        """Get the constructor arguments needed to reload encoder state.

        Returns:
            Parsed architecture, including YOLO's selected model scale.
        """
        return {"architecture": deepcopy(self.network.yaml)}

    def reinitialize_post_pooling(self) -> None:
        """Renew the part after pooling: the convolution of the ``Classify`` block.

        Everything before it keeps the encoder's weights. This is the optional
        encoder interface behind ``head: full``.
        """
        pooled = self.network.model[-1]
        conv = pooled.conv.conv
        # Only the convolution is kept; the throwaway linear layer is the cost of
        # reusing Ultralytics' own construction and initialization.
        fresh = Classify(
            conv.in_channels,
            1,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.groups,
        )
        initialize_weights(fresh)
        pooled.conv = fresh.conv

    def to_native(self, head: nn.Linear, label_names: Mapping[str, int]) -> nn.Module:
        """Rebuild an Ultralytics classification model from the encoder and a head.

        Args:
            head: Trained linear classifier over the encoder features.
            label_names: Class name to class ID mapping.

        Returns:
            A ``ClassificationModel`` with ``names`` that Ultralytics can load.
        """
        architecture = deepcopy(self.network.yaml)
        layer = architecture["head"][-1]
        if layer[2] != "Classify":
            raise ValueError(f"Expected a Classify head, found {layer[2]!r}")
        if not layer[3]:
            raise ValueError("The Classify head lists no arguments to set the classes")
        # The class count lives in the Classify layer's arguments, not only in `nc`.
        architecture["nc"] = len(label_names)
        layer[3][0] = len(label_names)
        native = ClassificationModel(cfg=architecture, verbose=False)
        for layer, trained in zip(native.model[:-1], self.network.model[:-1]):
            layer.load_state_dict(trained.state_dict())
        classify = native.model[-1]
        pooled = self.network.model[-1]
        classify.conv.load_state_dict(pooled.conv.state_dict())
        classify.linear.load_state_dict(head.state_dict())
        native.names = {index: name for name, index in label_names.items()}
        return native.eval()

    def forward(self, images: Tensor) -> Tensor:
        """Return one pre-classifier feature vector per RGB image.

        Args:
            images: Float RGB tensor in [0, 1], of shape ``(B, 3, H, W)``.

        Returns:
            Spatially pooled features of shape ``(B, feature_dim)``.
        """
        return self.network.predict(images)
