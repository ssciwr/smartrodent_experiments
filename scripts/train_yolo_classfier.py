from smartrodent import YoloClassificationTrainer
from smartrodent import config_utils
import yaml
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open(
    "/home/hmack/projects/smartrodent_experiments/configs/train_yolo_classifier_config.yaml",
    "r",
) as f:
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

    print(trainer.model.ckpt["train_args"])

    trainer.train()

    output = trainer.export()
    print("model exported to: ", output)
