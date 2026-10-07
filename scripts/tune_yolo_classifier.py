from smartrodent import YoloClassificationTrainer
from smartrodent import config_utils
import yaml
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path = Path(
    "./projects/smartrodent_experiments/configs/train_yolo_classifier_config.yaml"
)
with config_path.open("r") as f:
    cfg = yaml.load(f, config_utils.get_loader())

configs = config_utils.ConfigHandler(cfg).run_configs
for config in configs:
    with open(
        Path("/tmp/") / "train_yolo_classifier_config.yaml",
        "w",
    ) as f:
        cfg = yaml.dump(config, f)

    trainer = YoloClassificationTrainer.from_config(
        "/tmp/train_yolo_classifier_config.yaml"
    )

    output_dir = Path(config["tune_kwargs"]["project"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)
    trainer.tune()

    output = trainer.export()
    print("model exported to: ", output)
