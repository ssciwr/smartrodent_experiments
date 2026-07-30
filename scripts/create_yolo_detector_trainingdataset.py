from smartrodent.dataprocessing import YoloDetectorDatasetCreatorFromSpeciesnet
from pathlib import Path
from smartrodent.biotrove_process import (
    load_config,
)

config_path = Path(
    "/home/hmack/projects/smartrodent_experiments/configs/data_config_full.yaml"
)
config = load_config(config_path)
params = config["data"]["biotrove"].get("metadata_processor_info", {})
speciesnames = params["categories"] + [
    "Canis familiaris",
    "Felis catus",
    "car",
    "empty",
]
dataset_output_path = "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/yolo_detector_training_dataset"
path_to_data = (
    "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/filtered_kept"
)
path_to_labels = "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/speciesnet_box_labels_nopad"

dataset_creator = YoloDetectorDatasetCreatorFromSpeciesnet(
    path_to_image_data=path_to_data,
    path_to_labels=path_to_labels,
    dataset_output_path=dataset_output_path,
    labels_to_filter=["animal", "rodent", "vehicle"],
    class_names=speciesnames,
    train_val_test_split=(0.7, 0.15, 0.15),
    IoU_threshold=0.2,
    confidence_threshold=0.15,
    background_image_dir="/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/filtered_kept/empty",
)

dataset = dataset_creator()
