from smartrodent import YoloDetectionTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainer = YoloDetectionTrainer.from_config(
    "/home/hmack/projects/smartrodent_experiments/configs/train_yolo_detector_config.yaml"
)

print(trainer.model.ckpt["train_args"])

trainer.train()

output = trainer.export()
print("model exported to: ", output)
