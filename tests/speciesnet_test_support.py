"""Shared external-model substitutes for SpeciesNet-related tests."""

from copy import deepcopy


class SpeciesNetDetectorStub:
    """Mirror the SDK's class-level cutoff, used during detector prediction."""

    DETECTION_THRESHOLD = 0.1


class SpeciesNetStub:
    """Provide native SpeciesNet predictions without model downloads or a GPU."""

    def __init__(self, predictions, *, max_batch_size=None, fail=False):
        """Configure model responses, batch capacity, and inference failure.

        Args:
            predictions: Native SpeciesNet prediction records to return.
            max_batch_size: Optional maximum number of images per invocation.
            fail: Whether inference raises a model failure.
        """
        self.predictions = predictions
        self.max_batch_size = max_batch_size
        self.fail = fail
        self.detector = SpeciesNetDetectorStub()

    def detect(self, *, filepaths):
        """Return predictions through the SDK's detector-only API.

        Args:
            filepaths: Source image paths for this invocation.

        Returns:
            Native SpeciesNet prediction dictionary for the requested images.

        Raises:
            ValueError: The requested batch exceeds the configured capacity.
            RuntimeError: An inference failure was configured.
        """
        if self.max_batch_size is not None and len(filepaths) > self.max_batch_size:
            raise ValueError("Detector batch exceeds capacity")
        if self.fail:
            raise RuntimeError("Model failed")

        requested = {str(path) for path in filepaths}
        return {
            "predictions": [
                {
                    **deepcopy(item),
                    "detections": [
                        deepcopy(box)
                        for box in item["detections"]
                        if box["conf"] >= SpeciesNetDetectorStub.DETECTION_THRESHOLD
                    ],
                }
                for item in self.predictions
                if item["filepath"] in requested
            ]
        }


def detection(confidence, *, label="animal", bbox=(0.25, 0.2, 0.5, 0.6)):
    """Build a native SpeciesNet detection with a normalized xywh box."""
    return {"conf": confidence, "label": label, "bbox": list(bbox)}
