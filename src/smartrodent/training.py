from ultralytics import YOLO
from pathlib import Path
from typing import Any


def train_model(
    data_config: str, model_name: str = "yolo26n.pt", trainer=None, **train_kwargs
) -> Any:
    """Thin wrapper around the yolo26 train functionality. Check out the yolo26 documentation for it.

    Args:
        data_config (str): Config for training dataset. must be coco style.
        model_name (str, optional): Name of the model checkpoint. Defaults to "yolo26n.pt".
        **kwargs (str, any): Training arguments for the
    Returns:
        Any: training results
    """
    if Path(data_config).suffix not in [".yaml", ".yml"]:
        raise ValueError("'data_config' must be a yaml file.")

    if model_name not in [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    ]:
        raise ValueError("yolo model name unknown")

    model = YOLO(model_name)
    results = model.train(trainer=trainer, data=data_config, **train_kwargs)
    return results
