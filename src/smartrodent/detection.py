from pathlib import Path
from ultralytics import YOLO, YOLOE

from .base import DetectorBase
import importlib
import string
import json


class YOLO_Detector(DetectorBase):
    """Wrap Ultralytics YOLO for boxed object detection.

    The detector saves Ultralytics previews and optional crops while also writing
    a normalized per-image class and confidence summary.
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
        # YOLO specific
        # yoloe specific
        model_name: str | None = None,
        classes: list[str] | None = None,
        task: str = "detect",
        project: str | None = None,
    ):
        """Initialize a YOLO detector run.

        Args:
            name: Ultralytics run name and output subdirectory name.
            crop: Whether to save a crop for each detection.
            batchsize: Maximum number of images passed to one prediction call.
            img_outname: Image output name retained for interface compatibility.
            conf: Confidence threshold passed to Ultralytics.
            relpad: Relative crop padding retained for interface compatibility.
            model_name: Filename, path, or identifier of the YOLO weights.
            classes: Class prompts retained for compatibility with
                ``YOLOE_Detector``.
            task: Ultralytics model task.
            project: Optional Ultralytics output directory.

        Raises:
            ValueError: If ``model_name`` is ``None``.
        """

        super().__init__(name, crop, batchsize, img_outname, conf, relpad, project)
        self.classes = classes
        self.task = task
        if model_name is None:
            raise ValueError("Error, model_name cannot be None")
        self.model_name = model_name

    def write_detections_json(self, results, json_path: Path | str) -> None:
        """Append Ultralytics results to normalized detection records.

        Existing records are preserved. Detections for each image are sorted by
        descending confidence, and images without boxes receive an empty list.

        Args:
            results: Ordered Ultralytics result objects.
            json_path: Destination path for ``detections.json``.
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(json_path.read_text()) if json_path.exists() else {}

        for result in results:
            boxes = result.boxes
            filename = Path(result.path).name
            if not boxes or len(boxes) == 0:
                records[filename] = []
                continue

            records[filename] = [
                {
                    "class": result.names[int(class_id)],
                    "conf": round(float(confidence), 3),
                }
                for class_id, confidence in sorted(
                    zip(boxes.cls, boxes.conf), key=lambda item: -item[1]
                )
            ]

        json_path.write_text(json.dumps(records, indent=2))

    def detect(self, path, out):
        """Run YOLO inference and write normalized detections.

        Args:
            path: A single image, a directory, or an ordered list of image paths.
            out: Directory in which to write ``detections.json``.

        Returns:
            Ultralytics result objects for all processed images, with original
            source paths restored for list inputs.
        """

        self.model = YOLO(self.resolve_local_model(self.model_name), task=self.task)

        all_results = []
        for source_batch in self.path_batches(path):
            # Chunk list inputs ourselves. Ultralytics' list loader treats all list
            # items as one in-memory batch, so passing the full group here can OOM.
            res = self.model.predict(
                source=source_batch,
                batch=len(source_batch),
                save=True,
                project=self.project,
                name=self.name,
                exist_ok=True,
                conf=self.conf,
                save_crop=self.crop,
            )

            res = self.restore_result_paths(res, source_batch)
            all_results.extend(res)
        self.write_detections_json(all_results, out / "detections.json")

        return all_results


class YOLOE_Detector(YOLO_Detector):
    """Wrap Ultralytics YOLOE with optional text-prompt classes.

    YOLOE uses the same normalized output format as ``YOLO_Detector`` and can
    configure natural-language classes before inference.
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
        # yoloe specific
        model_name: str | None = None,
        classes: list[str] | None = None,
        task: str = "detect",
        project: str | None = None,
    ):
        """Initialize a YOLOE detector run.

        Args:
            name: Detector run name retained for interface compatibility.
            crop: Whether to save a crop for each detection.
            batchsize: Maximum number of images passed to one prediction call.
            img_outname: Image output name retained for interface compatibility.
            conf: Confidence threshold passed to Ultralytics.
            relpad: Relative crop padding retained for interface compatibility.
            model_name: Filename, path, or identifier of the YOLOE weights.
            classes: Optional natural-language class prompts.
            task: Ultralytics model task.
            project: Optional Ultralytics output directory.

        Raises:
            ValueError: If ``model_name`` is ``None``.
        """

        super().__init__(
            name,
            crop,
            batchsize,
            img_outname,
            conf,
            relpad=relpad,
            model_name=model_name,
            classes=classes,
            task=task,
            project=project,
        )

    def detect(self, path, out):
        """Run YOLOE inference and write normalized detections.

        Args:
            path: A single image, a directory, or an ordered list of image paths.
            out: Directory in which to write ``detections.json``.

        Returns:
            Ultralytics result objects for all processed images, with original
            source paths restored for list inputs.
        """

        # Load the local YOLOE model. Resolve the weights relative to this file so
        # imports from outside the package still find the bundled model file.
        model = YOLOE(
            self.resolve_local_model(self.model_name),
            task=self.task,
        )

        if self.classes:
            model.set_classes(
                self.classes,
                model.get_text_pe(self.classes),
            )

        # Optional training call kept here as a reminder/example for later experiments.
        # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        all_results = []
        for source_batch in self.path_batches(path):
            if isinstance(source_batch, str | Path):
                # A single image does not need an explicit batch size. Resolve the path so
                # Ultralytics receives an absolute filename regardless of the caller's cwd.
                res = model.predict(
                    source=str(Path(source_batch).resolve()),
                    save=True,
                    project=self.project,
                    name="boxed",
                    exist_ok=True,
                    conf=self.conf,
                    save_crop=self.crop,
                )

            else:
                # Chunk list inputs ourselves. Ultralytics' list loader treats all list
                # items as one in-memory batch, so passing the full group here can OOM.
                res = model.predict(
                    source=source_batch,
                    batch=len(source_batch),
                    save=True,
                    project=self.project,
                    name="boxed",
                    exist_ok=True,
                    conf=self.conf,
                    save_crop=self.crop,
                )

            res = self.restore_result_paths(res, source_batch)
            all_results.extend(res)
        self.write_detections_json(all_results, out / "detections.json")

        return all_results


class DetectionExperiment:
    """Run one detector configuration across grouped dataset images.

    Shared experiment settings are fixed at initialization. A fresh detector is
    created for each group so backend-native output is written to that group's
    project directory.
    """

    def __init__(
        self,
        run_dir: str | Path,
        dataset_name: str,
        experiment_name: str,
        conf: float,
        detector_type: str,
        detector_args: list | None = None,
        detector_kwargs: dict | None = None,
    ):
        """Initialize shared experiment settings.

        Args:
            run_dir: Root directory for experiment output.
            dataset_name: Name of the dataset being processed.
            experiment_name: Name used in the experiment output path.
            conf: Confidence threshold passed to each detector instance.
            detector_type: Name of a ``DetectorBase`` subclass in this module.
            detector_args: Optional positional arguments for the detector.
            detector_kwargs: Optional keyword arguments for the detector. The
                ``batchsize`` value defaults to ``1`` and is handled separately.

        Raises:
            ValueError: If ``detector_type`` does not identify a concrete
                ``DetectorBase`` subclass in this module.
        """
        self.run_dir = Path(run_dir)
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.conf = conf

        module = importlib.import_module(__name__)
        detector_cls = getattr(module, detector_type, None)
        if not (
            isinstance(detector_cls, type)
            and issubclass(detector_cls, DetectorBase)
            and detector_cls is not DetectorBase
        ):
            raise ValueError(
                f"Unknown detector type {detector_type!r}: expected the name of a "
                f"DetectorBase subclass defined in {__name__}."
            )
        self.detector_cls = detector_cls

        self.detector_args = list(detector_args or [])
        self.detector_kwargs = dict(detector_kwargs or {})
        self.batchsize = self.detector_kwargs.pop("batchsize", 1)

    def build_detector(self, project: Path) -> DetectorBase:
        """Build the configured detector for one image group.

        Args:
            project: Backend-native output directory for the group.

        Returns:
            A newly initialized detector instance.
        """
        return self.detector_cls(
            *self.detector_args,
            name="boxed",
            batchsize=self.batchsize,
            img_outname="boxed",
            conf=self.conf,
            project=project,
            **self.detector_kwargs,
        )

    def run_detector_on_group(self, group_name: str, imgs: list[Path]):
        """Run the configured detector on one image group.

        Args:
            group_name: Name used for the group's output directory.
            imgs: Ordered image paths belonging to the group.

        Returns:
            Results returned by the detector's ``detect`` method.
        """
        out = (
            self.run_dir
            / f"detect{self.conf_to_string(self.conf)}"
            / self.experiment_name
            / self.dataset_name
            / group_name
        )
        out.mkdir(parents=True, exist_ok=True)

        detector = self.build_detector(out)
        return detector.detect(imgs, out)

    ## Experimentation driver functions
    def conf_to_string(self, conf: float) -> str:
        """Convert a confidence value to its run-directory suffix.

        Args:
            conf: Detection confidence value.

        Returns:
            The confidence string with punctuation removed. For example, ``0.05``
            becomes ``"005"``.
        """
        return str(conf).translate(str.maketrans("", "", string.punctuation))

    def resolve_lookups(self, value, class_sets: dict[str, list[str]]):
        """Resolve class-set lookup markers recursively.

        Args:
            value: Configuration value that may contain lookup mappings.
            class_sets: Named class lists available to lookup mappings.

        Returns:
            The value with every ``{"lookup": "name"}`` mapping replaced by the
            corresponding class list.

        Raises:
            KeyError: If a lookup refers to an unknown class-set name.
        """
        if isinstance(value, dict):
            if set(value) == {"lookup"}:
                lookup_name = value["lookup"]
                try:
                    return class_sets[lookup_name]
                except KeyError as exc:
                    raise KeyError(
                        f"Unknown class set lookup {lookup_name!r}. "
                        f"Available class sets: {sorted(class_sets)}"
                    ) from exc
            return {
                key: self.resolve_lookups(item, class_sets)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self.resolve_lookups(item, class_sets) for item in value]

        return value

    def load_experiment_config(self, config_path: Path) -> dict:
        """Load an experiment configuration and resolve class-set references.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            The parsed configuration with lookup mappings expanded in its
            ``experiments`` value.
        """
        with config_path.open() as config_file:
            config = json.load(config_file)

        class_sets = config.get("class_sets", {})
        config["experiments"] = self.resolve_lookups(config["experiments"], class_sets)
        return config

    def image_groups(self, root: Path):
        """Yield image groups from immediate subdirectories.

        Args:
            root: Root directory containing one subdirectory per image group.

        Yields:
            Tuples containing the group directory name and its sorted image paths.
            Files, empty directories, and nested groups are ignored.
        """
        for imgpath in sorted(root.iterdir()):
            if not imgpath.is_dir():
                continue
            imgs = sorted(imgpath.iterdir())
            if imgs:
                yield imgpath.name, imgs
