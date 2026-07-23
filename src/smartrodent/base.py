from abc import ABC, abstractmethod
from pathlib import Path


class DetectorBase(ABC):
    """Define the common interface and helpers for detector backends.

    Subclasses wrap model-specific APIs behind a common ``detect(path, out)``
    method and convert their native results into the normalized
    ``detections.json`` format.
    """

    def __init__(
        self,
        name: str,
        crop: bool,
        batchsize: int,
        img_outname: str,
        conf: float,
        # padding for crop extraction:
        relpad=0.0,
        project: str | None = None,
    ):
        """Initialize settings shared by detector backends.

        Args:
            name: Name of the detector run.
            crop: Whether the backend should save detection crops.
            batchsize: Maximum number of images processed in one batch.
            img_outname: Name used for image output where supported.
            conf: Minimum detection confidence.
            relpad: Relative padding used when extracting crops.
            project: Optional directory for backend-native output.
        """
        # all
        self.name = name
        self.crop = crop
        self.batchsize = batchsize
        self.img_outname = img_outname
        self.conf = conf
        self.model = None
        self.relpad = relpad
        self.project = project

    @abstractmethod
    def write_detections_json(self, results, json_path: Path | str) -> None:
        """Write backend-native results as normalized detection records.

        Args:
            results: Results in the detector backend's native format.
            json_path: Destination path for the normalized JSON records.
        """
        pass

    def resolve_local_model(self, model_name: str | Path) -> str:
        """Resolve a model path, preferring weights bundled with this module.

        Args:
            model_name: Absolute path, relative path, or model identifier.

        Returns:
            The resolved bundled path when it exists; otherwise, ``model_name``
            converted to a string.
        """
        model_path = Path(model_name)
        if model_path.is_absolute():
            return str(model_path)

        bundled_path = Path(__file__).with_name(str(model_name))
        return str(bundled_path) if bundled_path.exists() else str(model_name)

    def restore_result_paths(self, results, source_paths):
        """Restore source filenames on Ultralytics batch results.

        Ultralytics may replace paths with names such as ``image0.jpg`` when it
        converts a list of paths into in-memory images. Result order still matches
        input order, so list inputs can be restored safely.

        Args:
            results: Ultralytics result objects to update.
            source_paths: A single source path or an ordered collection of paths.

        Returns:
            The original ``results`` collection, with paths updated for collection
            inputs.
        """
        if isinstance(source_paths, str | Path):
            return results

        for result, source_path in zip(results, source_paths, strict=False):
            result.path = str(Path(source_path).resolve())

        return results

    def path_batches(self, path: str | Path | list[Path | str]):
        """Split detector input into bounded inference batches.

        Args:
            path: A single path or an ordered list of image paths.

        Yields:
            A single path unchanged, or lists containing at most ``batchsize``
            paths.

        Raises:
            ValueError: If ``batchsize`` is not positive for a list input.
        """
        if isinstance(path, str | Path):
            yield path
            return

        if self.batchsize <= 0:
            raise ValueError(f"batchsize must be positive, got {self.batchsize}")

        for start in range(0, len(path), self.batchsize):
            yield path[start : start + self.batchsize]

    @abstractmethod
    def detect(
        self, path: str | Path | list[Path | str], out: Path, *args, **kwargs
    ) -> list:
        """Run detection and write normalized output.

        Args:
            path: A single image, a directory, or an ordered list of image paths.
            out: Directory for normalized detector output.
            *args: Additional backend-specific positional arguments.
            **kwargs: Additional backend-specific keyword arguments.

        Returns:
            Results in the detector backend's native format.
        """
        pass
