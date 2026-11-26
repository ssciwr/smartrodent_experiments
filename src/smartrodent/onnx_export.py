"""Write shareable ONNX files for trained CE classifiers and SupCon encoders."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import onnx
import onnxslim
import torch
from ultralytics import YOLO
from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

from smartrodent.supcon_model import YoloSupConModel
from smartrodent.training_validation import OnnxExportSettings


class _PooledFeatures(torch.nn.Module):
    """Expose the SupCon wrapper's pooled encoder features as the traced graph's output."""

    def __init__(self, model: YoloSupConModel):
        """Wrap a restored SupCon model whose pool hook supplies the features."""
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reuse the training-time feature hook, so the export cannot drift from it."""
        # The classifier and projector outputs are unused, so the exporter drops them.
        self.model(images)
        return self.model.embeddings


class OnnxExporter:
    """Export best checkpoints next to themselves with one shared set of ONNX options.

    Both exports expect the same input: RGB, float32 in [0, 1], N x 3 x S x S,
    resized to the training image size, without mean/std normalization.

    Args:
        settings: Validated ``onnx_export`` options.
        image_size: Square input size S the network was trained on.
    """

    INPUT_DESCRIPTION = (
        "images: RGB, float32 in [0, 1], N x 3 x S x S (S from imgsz), "
        "resized without cropping, no mean/std normalization"
    )

    def __init__(self, settings: OnnxExportSettings, image_size: int):
        """Resolve omitted options to Ultralytics' export defaults once, for both exports."""
        self.dynamic = settings.get("dynamic", False)
        self.simplify = settings.get("simplify", True)
        self.image_size = image_size

    def classifier(self, checkpoint: Path) -> Path:
        """Export a native classifier with Ultralytics' own exporter (class probabilities).

        Args:
            checkpoint: Native Ultralytics classification checkpoint (best.pt).

        Returns:
            The ONNX file written next to the checkpoint.
        """
        output = checkpoint.with_suffix(".onnx")
        with self._removed_on_failure(output):
            # Export on the CPU: the file is device-independent, and training may
            # still hold the GPU. Simplification runs in _finish instead, because
            # Ultralytics downgrades a simplifier failure to a warning.
            YOLO(str(checkpoint), task="classify").export(
                format="onnx",
                imgsz=self.image_size,
                dynamic=self.dynamic,
                simplify=False,
                device="cpu",
            )
            self._finish(output, {})
        return output

    def encoder(self, checkpoint: Path) -> Path:
        """Export a SupCon checkpoint's pooled encoder features (pre-projection embeddings).

        Args:
            checkpoint: Standalone SupCon checkpoint (best.pt).

        Returns:
            The ONNX file written next to the checkpoint, with ``images`` input and
            ``embeddings`` output.
        """
        output = checkpoint.with_suffix(".onnx")
        with self._removed_on_failure(output):
            embedding_dim = self._trace_encoder(checkpoint, output)
            self._finish(output, self._encoder_metadata(checkpoint, embedding_dim))
        return output

    def _trace_encoder(self, checkpoint: Path, output: Path) -> int:
        """Trace like Ultralytics' native export (same exporter, opset, dynamic axes); return the feature size."""
        dynamic = (
            {
                "images": {0: "batch", 2: "height", 3: "width"},
                "embeddings": {0: "batch"},
            }
            if self.dynamic
            else None
        )
        model = YoloSupConModel.from_checkpoint(checkpoint)
        try:
            torch2onnx(
                _PooledFeatures(model).eval(),
                torch.zeros(1, 3, self.image_size, self.image_size),
                output,
                opset=best_onnx_opset(onnx),
                input_names=["images"],
                output_names=["embeddings"],
                dynamic=dynamic,
            )
            return model.head.linear.in_features
        finally:
            model.finalize()

    def _encoder_metadata(self, checkpoint: Path, embedding_dim: int) -> dict[str, str]:
        """Describe what a recipient needs to run the encoder file and trace it to best.pt."""
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "embedding_dim": str(embedding_dim),
            # Same "[S, S]" format as Ultralytics' native classifier export.
            "imgsz": str([self.image_size, self.image_size]),
            "checkpoint_sha256": checksum,
            "input": self.INPUT_DESCRIPTION,
        }

    def _finish(self, output: Path, metadata: dict[str, str]) -> None:
        """Simplify when requested and add metadata; both exports share this step.

        A simplifier failure propagates rather than silently leaving an unsimplified
        file. onnxslim keeps existing metadata, such as Ultralytics' names and imgsz.
        """
        document = onnx.load(str(output))
        if self.simplify:
            document = onnxslim.slim(document)
        for key, value in metadata.items():
            document.metadata_props.add(key=key, value=value)
        onnx.save(document, str(output))

    @staticmethod
    @contextmanager
    def _removed_on_failure(output: Path) -> Iterator[None]:
        """Never leave a partial file that looks like a finished export."""
        try:
            yield
        except BaseException:
            output.unlink(missing_ok=True)
            raise
