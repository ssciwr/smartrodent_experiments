import pandas as pd
from pathlib import Path
import yaml
from typing import Any
import warnings


def create_dataset_csv(config: dict[str, Any]):
    """
    Build dataset csv file as described here https://agentmorris.github.io/speciesnet-fine-tuning/#running-megadetector-on-your-images
    """
    # Build mapping files
    remappings = {}
    for mapping_name, mappings in config["mappings"].items():
        remapping = {"input": [], "output": []}

        for speciesname, newname in mappings.items():
            remapping["input"].append(speciesname)
            remapping["output"].append(newname)

        remapping = pd.DataFrame.from_dict(remapping)
        remappings[mapping_name] = remapping

    # build main data file
    species = set(config["specieslist"])
    input_path = Path(config["input_data_path"])
    input_path = input_path.resolve()
    if not input_path.exists():
        raise ValueError("Error, input path does not exist")

    cols = {"filename": [], "category": [], "location": []}
    for dir in input_path.iterdir():
        if dir.name in species:
            imgpath = Path(str(dir / config.get("images_path", "")).strip())
            for detection_cat in imgpath.iterdir():
                if detection_cat.stem in config["detection_categories"][dir.name]:
                    for candidate in detection_cat.iterdir():
                        if candidate.suffix in config["image_formats"]:
                            cols["filename"].append(str(candidate.resolve()))
                            cols["category"].append(dir.name)
                            cols["location"].append("unknown")

        else:
            warnings.warn(
                f"Dirctory found that doesn't correspond to a recognized species: {dir.name}"
            )

    dataset = pd.DataFrame.from_dict(cols)

    output_path = Path(config["output_path"]).resolve()
    output_path.mkdir(exist_ok=True, parents=True)

    dataset.to_csv(output_path / "dataset.csv")
    for mappingname, mapping in remappings.items():
        mapping.to_csv(output_path / f"{mappingname}.csv")


def create_dataset_md_results_file(config):
    """
    Use existing results from megadetector/speciesnet detector runs and build a megadetector results file

    """
    # TODO


if __name__ == "__main__":
    config_path = Path(
        "/home/hmack/projects/smartrodent_experiments/configs/speciesnet_trainingdataset_creation_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_output_path = "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/yolo_classifier_training_dataset"
    path_to_data = "/home/hmack/projects/smartrodent_experiments/datasets/full_dataset/speciesnet_box_labels_nopad"
    image_directory = "crops"

    create_dataset_csv(config)
