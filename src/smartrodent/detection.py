from pathlib import Path
from ultralytics import YOLO, YOLOE

from .base import DetectorBase
import importlib
import string
import json


class YOLO_Detector(DetectorBase):
    """Ultralytics YOLO26 detector wrapper for boxed object detections.

    This backend uses the bundled YOLO26 weights by default, lets Ultralytics save
    preview images/crops, and writes a compact per-image class/confidence summary.
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
        """Configure a YOLO26 detector run.

        Args:
            name (str): Ultralytics run name; used as the ``name=`` argument to
                ``predict`` and as the subfolder under ``project`` where previews land.
            crop (bool): Whether to save per-detection crop images alongside previews.
            batchsize (int): Maximum number of images sent to ``predict`` per call.
            img_outname (str): Unused by this backend; kept for interface parity with
                other detectors.
            conf (float): Confidence threshold passed to ``predict``.
            relpad (float, optional): Unused by this backend; kept for interface parity
                with other detectors. Defaults to 0.0.
            model_name (str | None, optional): Filename or path of the YOLO weights to
                load, resolved via ``resolve_local_model``. Required. Defaults to None.
            classes (list[str] | None, optional): Unused by this backend; kept for
                interface parity with ``YOLOE_Detector``. Defaults to None.
            task (str, optional): Ultralytics task passed to the ``YOLO`` constructor.
                Defaults to "detect".
            project (str | None, optional): Ultralytics ``project=`` output directory
                for previews/crops. Defaults to None.

        Raises:
            ValueError: If ``model_name`` is not provided.
        """

        super().__init__(name, crop, batchsize, img_outname, conf, relpad, project)
        self.classes = classes
        self.task = task
        if model_name is None:
            raise ValueError("Error, model_name cannot be None")
        self.model_name = model_name

    def write_detections_json(self, results, json_path: Path | str) -> None:
        """Append Ultralytics results to a normalized detections JSON file."""
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
        """Run YOLO26 inference and write a normalized detections summary.

        Args:
            path (str | Path | list[Path | str]): A single image/directory, or a list
                of image paths to run in batches of ``self.batchsize``.
            out (Path): Directory to write ``detections.json`` into.

        Returns:
            list: The Ultralytics ``Results`` objects for all processed images, with
            their ``path`` attributes restored to the original source paths.
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
    """Ultralytics YOLOE detector wrapper with optional text prompt classes.

    YOLOE can be run with natural-language class prompts through ``self.classes``.
    The rest of the output handling mirrors ``YOLO_Detector`` so experiments can
    compare YOLO26 and YOLOE using the same ``detections.json`` format.
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
        """Configure a YOLOE detector run, optionally with text-prompt classes.

        Args:
            name (str): Unused by this backend (the Ultralytics run name is hardcoded
                to "boxed" in ``detect``); kept for interface parity with ``YOLO_Detector``.
            crop (bool): Whether to save per-detection crop images alongside previews.
            batchsize (int): Maximum number of images sent to ``predict`` per call.
            img_outname (str): Unused by this backend; kept for interface parity with
                other detectors.
            conf (float): Confidence threshold passed to ``predict``.
            relpad (float, optional): Unused by this backend; kept for interface parity
                with other detectors. Defaults to 0.0.
            model_name (str | None, optional): Filename or path of the YOLOE weights to
                load, resolved via ``resolve_local_model``. Required. Defaults to None.
            classes (list[str] | None, optional): Natural-language class prompts. When
                given, ``detect`` calls ``model.set_classes`` before running inference.
                Defaults to None.
            task (str, optional): Ultralytics task passed to the ``YOLOE`` constructor.
                Defaults to "detect".
            project (str | None, optional): Ultralytics ``project=`` output directory
                for previews/crops. Defaults to None.

        Raises:
            ValueError: If ``model_name`` is not provided.
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
        """Run YOLOE inference and write a normalized detections summary.

        Args:
            path (str | Path | list[Path | str]): A single image/directory, or a list
                of image paths to run in batches of ``self.batchsize``.
            out (Path): Directory to write ``detections.json`` into.

        Returns:
            list: The Ultralytics ``Results`` objects for all processed images, with
            their ``path`` attributes restored to the original source paths.
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
    """Runs one detector configuration across the groups of a dataset.

    ``run_dir``, ``dataset_name``, ``experiment_name`` and ``conf`` are shared by every
    group processed in the experiment, so they are fixed once here rather than being
    re-passed on each call. The detector itself is also owned by this class: give it the
    detector's class name plus the positional/keyword arguments it needs, and this class
    resolves the name to a ``DetectorBase`` subclass and builds a fresh instance per group
    (each group needs its own ``project`` output directory).
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
        """Store shared experiment settings and resolve the detector to build per group.

        Args:
            run_dir (str | Path): Root directory that all experiment runs are written under.
            dataset_name (str): Name of the dataset being processed.
            experiment_name (str): Name of this experiment, used in the output path.
            conf (float): Confidence threshold passed to every detector instance.
            detector_type (str): Name of a ``DetectorBase`` subclass defined in this
                module, e.g. ``"YOLO_Detector"``.
            detector_args (list | None, optional): Positional args forwarded to the
                detector constructor. Defaults to none.
            detector_kwargs (dict | None, optional): Keyword args forwarded to the
                detector constructor. ``batchsize`` may be included here and defaults
                to 1 if omitted. Defaults to none.

        Raises:
            ValueError: If ``detector_type`` is not a ``DetectorBase`` subclass defined
                in this module.
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
        """Instantiate the configured detector for a single group's output directory."""
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
        """Run the configured detector on one group of images.

        Args:
            group_name (str): Name of the image group, used in the output path.
            imgs (list[Path]): Images belonging to this group.

        Returns:
            list: The detector's per-image results, as returned by ``detect``.
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
        """Convert a confidence value into the run-folder suffix used by experiments.

        For example, ``0.05`` becomes ``"005"`` so outputs land under folders such as
        ``runs/detect005/...``. This preserves the historical directory layout.
        """
        return str(conf).translate(str.maketrans("", "", string.punctuation))

    def resolve_lookups(self, value, class_sets: dict[str, list[str]]):
        """Replace {"lookup": "name"} markers with lists from config.class_sets."""
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
        """Load detector experiment settings and resolve class-set references.

        The JSON file can keep long class lists under ``class_sets`` and refer to them
        from an experiment with ``{"lookup": "class_set_name"}``. This function expands
        those references before the main loop instantiates detector classes.
        """
        with config_path.open() as config_file:
            config = json.load(config_file)

        class_sets = config.get("class_sets", {})
        config["experiments"] = self.resolve_lookups(config["experiments"], class_sets)
        return config

    def image_groups(self, root: Path):
        """Yield named image batches from immediate subdirectories of ``root``.

        The current dataset layout groups images by folder. Each yielded tuple is
        ``(folder_name, sorted_image_paths)`` and empty/non-directory entries are ignored.
        """
        for imgpath in sorted(root.iterdir()):
            if not imgpath.is_dir():
                continue
            imgs = sorted(imgpath.iterdir())
            if imgs:
                yield imgpath.name, imgs
