import json
from pathlib import Path

from ultralytics import YOLOE


def detect(
    path: str | Path | list[Path | str],
    batchsize: int,
    project: Path | str = "runs/yolo26",
    crop: bool = False,
    conf=0.1,
    task: str = "detect",
    classes: list[str] | None = ["rodent", "non-rodent"],
):
    """Run YOLO on one image path or a batch of image paths.

    Ultralytics handles saving boxed preview images when ``save=True`` is passed to
    ``model.predict``. Single-image and batch paths are split here only so the batch
    case can use a larger ``batch`` value.
    """
    # Load the local COCO-pretrained YOLO model. Resolve the weights relative to this
    # file so imports from outside the package still find the bundled model file.
    model = YOLOE(
        "yoloe-26m-seg.pt",
        task=task,
    )

    if classes:
        model.set_classes(
            classes,
            model.get_text_pe(classes),
        )

    # Optional training call kept here as a reminder/example for later experiments.
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    if isinstance(path, str | Path):
        # A single image does not need an explicit batch size. Resolve the path so
        # Ultralytics receives an absolute filename regardless of the caller's cwd.
        return model.predict(
            source=str(Path(path).resolve()),
            save=True,
            project=project,
            name="boxed",
            exist_ok=True,
            conf=conf,
            save_crop=crop,
        )

    # For a list of images, pass the caller's batch size through to Ultralytics so
    # larger datasets can be processed more efficiently.
    return model.predict(
        source=path,
        batch=batchsize,
        save=True,
        project=project,
        name="boxed",
        exist_ok=True,
        conf=conf,
        save_crop=crop,
    )


def write_detections_json(results, json_path: Path | str) -> None:
    """Append detection records from a predict() result list to a JSON file.

    Existing entries are preserved so the file accumulates across per-image calls.
    Each entry is keyed by filename and contains a list of {class, conf} dicts.
    """
    json_path = Path(json_path)
    records = json.loads(json_path.read_text()) if json_path.exists() else {}
    for r in results:
        boxes = r.boxes
        if not boxes or len(boxes) == 0:
            records[Path(r.path).name] = []
        else:
            records[Path(r.path).name] = [
                {"class": r.names[int(cls)], "conf": round(float(conf), 3)}
                for cls, conf in sorted(zip(boxes.cls, boxes.conf), key=lambda x: -x[1])
            ]
    json_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    IMAGE_DIR = Path(
        "/home/hmack/Development/rodent_experiments/datasets/biotrove-central-europe/filtered"
    )

    for imgpath in IMAGE_DIR.iterdir():
        name = imgpath.name
        out = Path(f"./runs/yoloe/central-europe/{name}")
        imgs = sorted(imgpath.iterdir())
        out.mkdir(parents=True, exist_ok=True)

        results = []
        for img in imgs:
            res = detect(
                img,
                1,
                crop=True,
                task="segment",
                project=out,
                classes=[
                    "an animal like a mouse, rat, cat, fox, or hamster",
                ],
                conf=0.25,
            )
            results.append(res)

        for res in results:
            write_detections_json(res, out / "detections.json")
