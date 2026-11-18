"""YOLO classification wrapper exposing contrastive representations."""

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import ultralytics
import yaml
from ultralytics.nn.tasks import ClassificationModel


class YoloSupConModel(torch.nn.Module):
    """Expose classifier logits and normalized projected pooling features.

    Args:
        model_name: Ultralytics classification checkpoint to load.
        dropout: Dropout probability in the projection head.
        output_features: Dimension of the normalized contrastive embeddings.
        num_classes: Number of outputs in the classifier's linear layer.
        freeze: Indices of classification model layers whose gradients to disable.
    """

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.0,
        output_features: int = 128,
        num_classes: int = 10,
        freeze: Iterable[int] = (),
    ):
        super().__init__()
        self.embeddings = None
        self.label_names = {}
        self.training_configuration = {}

        def capture_pool_output(module, input, output):
            """Keep pooled features connected to the encoder's gradient graph."""
            # Clone because the projection head applies dropout in place.
            self.embeddings = output.flatten(1).clone()

        self.full_model = ultralytics.YOLO(model_name, task="classify")
        ClassificationModel.reshape_outputs(self.full_model.model, nc=num_classes)

        for f in freeze:
            self.full_model.model.model[f].requires_grad_(False)

        self.head = self.full_model.model.model[-1]
        self.handle = self.head.pool.register_forward_hook(capture_pool_output)
        self.projector = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout, inplace=True),
            torch.nn.Linear(
                in_features=self.head.linear.in_features,
                out_features=output_features,
                bias=True,
            ),
        )

    def train(self, mode: bool = True):
        """Set the network, projector, and wrapper training mode.

        Args:
            mode: Whether to enable training behavior.

        Returns:
            This wrapper.
        """
        self.full_model.model.train(mode)
        self.projector.train(mode)
        self.training = mode
        return self

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "YoloSupConModel":
        """Restore a standalone experiment checkpoint without pretrained weights.

        Args:
            path: Checkpoint produced by ``scripts/train_supcon.py``.
            device: Device on which the restored model should run.

        Returns:
            Restored wrapper in evaluation mode, with its pooling hook active.

        Raises:
            ValueError: Required model metadata is absent from the checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        required = {"architecture", "model_configuration", "model_state_dict", "label_names"}
        if (
            not isinstance(checkpoint, dict)
            or not required.issubset(checkpoint)
            or checkpoint.get("checkpoint_type") not in (None, "supcon")
        ):
            raise ValueError("Expected a checkpoint produced by scripts/train_supcon.py")
        architecture = deepcopy(checkpoint["architecture"])
        scale = architecture.get("scale")
        if scale and "scales" in architecture:
            # Ultralytics infers scale from the YAML filename. Keeping only the
            # selected scale also preserves custom filenames without a scale suffix.
            architecture["scales"] = {scale: architecture["scales"][scale]}
        filename = Path(architecture.get("yaml_file", "model.yaml")).name
        with TemporaryDirectory(prefix="smartrodent-checkpoint-") as temporary:
            config_path = Path(temporary) / filename
            config_path.write_text(yaml.safe_dump(architecture))
            model = cls(str(config_path), **checkpoint["model_configuration"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.label_names = dict(checkpoint["label_names"])
        model.training_configuration = dict(checkpoint.get("configuration", {}))
        model.full_model.model.names = {index: name for name, index in model.label_names.items()}
        model.to(device)
        model.train(False)
        return model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute classifier outputs and normalized contrastive embeddings.

        Args:
            images: Batch of image tensors on the model's device.

        Returns:
            Classifier outputs and normalized projected pooling features.
        """
        logits = self.full_model.model.predict(images)
        z = torch.nn.functional.normalize(
            self.projector.forward(self.embeddings), p=2, dim=1
        )
        return logits, z

    def finalize(self):
        """Remove the feature hook and return the underlying YOLO wrapper.

        Returns:
            The wrapped Ultralytics YOLO instance.
        """
        self.handle.remove()
        self.embeddings = None
        return self.full_model
