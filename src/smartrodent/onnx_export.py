"""Write shareable ONNX files for trained classifiers and SupCon encoders."""

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import onnx
import onnxslim
import torch
from torch import nn
from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

from smartrodent.classifier_task import ClassifierTask
from smartrodent.supcon_task import SupConTask


class _Probabilities(nn.Module):
    """Turn a classifier's logits into class probabilities, as a recipient expects."""

    def __init__(self, task: ClassifierTask):
        super().__init__()
        self.task = task

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one probability vector per image."""
        return self.task(images).softmax(dim=1)


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

    def __init__(self, settings: Mapping, image_size: int):
        """Resolve omitted options to their defaults once, for both exports."""
        self.dynamic = settings.get("dynamic", False)
        self.simplify = settings.get("simplify", True)
        self.image_size = image_size

    def classifier(self, checkpoint: Path) -> Path:
        """Export a classifier checkpoint as class probabilities.

        Args:
            checkpoint: Lightning checkpoint of a ``ClassifierTask``.

        Returns:
            The ONNX file written next to the checkpoint.
        """
        task = ClassifierTask.load_from_checkpoint(checkpoint, map_location="cpu")
        names = {index: name for name, index in task.label_names.items()}
        return self._export(
            _Probabilities(task),
            checkpoint,
            output_name="probabilities",
            # The keys Ultralytics reads to load the file with ``YOLO(path)``.
            metadata={
                "names": str(names),
                "task": "classify",
                "batch": "1",
                "stride": "1",
                "channels": "3",
            },
        )

    def encoder(self, checkpoint: Path) -> Path:
        """Export a SupCon checkpoint's encoder features (pre-projection embeddings).

        Args:
            checkpoint: Lightning checkpoint of a ``SupConTask``.

        Returns:
            The ONNX file written next to the checkpoint.
        """
        task = SupConTask.load_from_checkpoint(checkpoint, map_location="cpu")
        return self._export(
            task.encoder,
            checkpoint,
            output_name="embeddings",
            metadata={"embedding_dim": str(task.encoder.feature_dim)},
        )

    def _export(
        self,
        module: nn.Module,
        checkpoint: Path,
        *,
        output_name: str,
        metadata: Mapping[str, str],
    ) -> Path:
        """Trace with Ultralytics' exporter and best opset, then simplify and annotate."""
        output = checkpoint.with_suffix(".onnx")
        opset = best_onnx_opset(onnx)
        dynamic = (
            {
                "images": {0: "batch", 2: "height", 3: "width"},
                output_name: {0: "batch"},
            }
            if self.dynamic
            else None
        )
        with self._removed_on_failure(output):
            torch2onnx(
                module.eval(),
                torch.zeros(1, 3, self.image_size, self.image_size),
                output,
                opset=opset,
                input_names=["images"],
                output_names=[output_name],
                dynamic=dynamic,
            )
            self._finish(
                output,
                {
                    **metadata,
                    "opset": str(opset),
                    # Same "[S, S]" format as Ultralytics' native classifier export.
                    "imgsz": str([self.image_size, self.image_size]),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "input": self.INPUT_DESCRIPTION,
                },
            )
        return output

    def _finish(self, output: Path, metadata: Mapping[str, str]) -> None:
        """Simplify when requested and add metadata.

        A simplifier failure propagates rather than silently leaving an unsimplified
        file. onnxslim keeps existing metadata.
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
