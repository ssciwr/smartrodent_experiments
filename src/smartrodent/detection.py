from pathlib import Path
from ultralytics import YOLO, YOLOE

from .base import DetectorBase


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
        """_summary_

        Args:
            name (str): _description_
            crop (bool): _description_
            batchsize (int): _description_
            img_outname (str): _description_
            conf (float): _description_
            relpad (float, optional): _description_. Defaults to 0.0.
            project (str | None, optional): _description_. Defaults to None.
        """

        super().__init__(name, crop, batchsize, img_outname, conf, relpad)
        self.project = project
        self.classes = classes
        self.task = task
        if model_name is None:
            raise ValueError("Error, model_name cannot be None")
        self.model_name = model_name

    def detect(self, path, out):
        """_summary_

        Args:
            path (_type_): _description_
            out (_type_): _description_

        Returns:
            _type_: _description_
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


class YOLOE_Detector(DetectorBase):
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
    ):
        """_summary_

        Args:
            name (str): _description_
            crop (bool): _description_
            batchsize (int): _description_
            img_outname (str): _description_
            conf (float): _description_
            relpad (float, optional): _description_. Defaults to 0.0.
            model_name (str | None, optional): _description_. Defaults to None.
            classes (list[str] | None, optional): _description_. Defaults to None.
            task (str, optional): _description_. Defaults to "detect".

        Raises:
            ValueError: _description_
        """

        super().__init__(name, crop, batchsize, img_outname, conf, relpad)

        self.classes = classes
        self.task = task
        if model_name is None:
            raise ValueError("Error, model_name cannot be None")
        self.model_name = model_name

    def detect(self, path, out):
        """_summary_

        Args:
            path (_type_): _description_
            out (_type_): _description_

        Returns:
            _type_: _description_
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


def run_detector_on_group(
    run_dir: str,
    dataset_name: str,
    detector_cls: type[DetectorBase],
    experiment_name: str,
    detector_kwargs: dict,
    conf: float,
    group_name: str,
    imgs: list[Path],
):
    """_summary_

    Args:
        run_dir (str): _description_
        dataset_name (str): _description_
        detector_cls (type[DetectorBase]): _description_
        experiment_name (str): _description_
        detector_kwargs (dict): _description_
        conf (float): _description_
        group_name (str): _description_
        imgs (list[Path]): _description_

    Returns:
        _type_: _description_
    """
    out = (
        Path(run_dir)
        / f"detect{conf_to_string(conf)}"
        / experiment_name
        / dataset_name
        / group_name
    )
    out.mkdir(parents=True, exist_ok=True)

    detector = detector_cls(
        name="boxed",
        batchsize=detector_kwargs.pop("batchsize", BATCHSIZE),
        img_outname="boxed",
        conf=conf,
        project=out,
        **detector_kwargs,
    )
    return detector.detect(imgs, out)
