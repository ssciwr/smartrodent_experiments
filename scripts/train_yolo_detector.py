from smartrodent import YoloDetectionTrainer
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path = Path(
    "./projects/smartrodent_experiments/configs/train_yolo_detector_config.yaml"
)
trainer = YoloDetectionTrainer.from_config(config_path)

print(trainer.model.ckpt["train_args"])

output_dir = Path(trainer.train_kwargs["project"])
output_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(config_path, output_dir / config_path.name)
trainer.train()

output = trainer.export()
print("model exported to: ", output)
