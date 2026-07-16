from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import json


class DetectorBase(ABC):
    """Common interface and shared helpers for all detector backends.

    Subclasses wrap very different model APIs (Ultralytics YOLO, SpeciesNet, and
    BioTrove-CLIP) behind one ``detect(path, out)`` method. The base class stores
    the experiment settings that are common across backends and provides helpers for
    writing a normalized ``detections.json`` file.
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
    ):
        """Store detector configuration for a single experiment run.

        Parameters are intentionally broad because the experiment JSON feeds several
        detector types through the same constructor. Each subclass only uses the
        fields relevant to its backend; unused fields are kept here so configs can be
        swapped between experiments without changing the orchestration code.
        """
        # all
        self.name = name
        self.crop = crop
        self.batchsize = batchsize
        self.img_outname = img_outname
        self.conf = conf
        self.model = None
        self.relpad = relpad

    def write_detections_json(self, results, json_path: Path | str) -> None:
        """Append detection records to a JSON file from any supported model.

        Accepts an Ultralytics Results list (YOLO26 or YOLOE) or a SpeciesNet predictions
        dict. Existing entries are preserved so the file accumulates across per-image calls.
        Each entry is keyed by filename and contains a list of {class, conf} dicts sorted
        by confidence descending.
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(json_path.read_text()) if json_path.exists() else {}

        if isinstance(results, dict) and "predictions" in results:
            # SpeciesNet: {"predictions": [{filepath, prediction, prediction_score, ...}]}
            for item in results["predictions"]:
                filename = Path(item["filepath"]).name
                label = self.shorten_label(item.get("prediction"))
                score = item.get("prediction_score")
                if label == "unknown" or score is None:
                    records[filename] = []
                else:
                    records[filename] = [
                        {"class": label, "conf": round(float(score), 3)}
                    ]
        else:
            # Ultralytics: list of Results objects (YOLO26 or YOLOE)
            for r in results:
                boxes = r.boxes
                if not boxes or len(boxes) == 0:
                    records[Path(r.path).name] = []
                else:
                    records[Path(r.path).name] = [
                        {"class": r.names[int(cls)], "conf": round(float(conf), 3)}
                        for cls, conf in sorted(
                            zip(boxes.cls, boxes.conf), key=lambda x: -x[1]
                        )
                    ]

        json_path.write_text(json.dumps(records, indent=2))

    def shorten_label(self, item):
        """Return the display label used in ``detections.json``.

        Most detectors already emit short human-readable labels, so the base
        implementation returns the value unchanged. SpeciesNet overrides this because
        it emits semicolon-delimited taxonomy strings.
        """
        return item

    def resolve_local_model(self, model_name: str | Path) -> str:
        """Resolve bundled model weights relative to this file when present."""
        model_path = Path(model_name)
        if model_path.is_absolute():
            return str(model_path)

        bundled_path = Path(__file__).with_name(str(model_name))
        return str(bundled_path) if bundled_path.exists() else str(model_name)

    def restore_result_paths(self, results, source_paths):
        """Restore original filenames on Ultralytics batch results.

        Ultralytics converts list-of-path inputs into PIL images internally. That can
        drop PIL ``filename`` metadata and make result paths fall back to image0.jpg,
        image1.jpg, etc. The result order matches the input order, so put the original
        source paths back before writing detections.json.
        """
        if isinstance(source_paths, str | Path):
            return results

        for result, source_path in zip(results, source_paths, strict=False):
            result.path = str(Path(source_path).resolve())

        return results

    def path_batches(self, path: str | Path | list[Path | str]):
        """Yield single-image input or bounded path batches for model inference.

        Ultralytics treats a Python list of paths as in-memory images and can ignore
        the separate ``batch=`` setting for that loader. Chunking before ``predict``
        keeps the real GPU batch bounded by ``self.batchsize``.
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
        """Run the detector on one image, a directory, or a batch of image paths.

        Subclasses should save their model-native outputs under ``out`` when useful,
        call ``write_detections_json`` to update the normalized summary, and return
        the backend's native result object for ad-hoc inspection.
        """
        pass
