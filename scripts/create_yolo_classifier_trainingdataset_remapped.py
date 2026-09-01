from pathlib import Path

import yaml

if __name__ == "__main__":
    config_path = Path(
        "/home/hmack/projects/smartrodent_experiments/configs/coco_style_trainingdataset_creation_config.yaml"
    )

    mapping = "reduced_rodenspecies_mapping"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inputpath = Path(config["input_data_path"]).resolve()
    outpath = Path(config["output_path"]).resolve()
    outpath.mkdir(exist_ok=True, parents=True)
    imageformats = config["image_formats"]
    ignore_unmapped_species = config.get("ignore_unmapped_species", False)
    species_mapping = config["mappings"][mapping]

    dataset_yaml = {
        "names": {i: n for i, n in enumerate(set(species_mapping.values()))},
        "path": str(outpath),
        "test": "test",
        "train": "train",
        "val": "val",
    }

    ignored_species = set()
    for split in ["train", "val", "test"]:
        split_path = inputpath / split
        for species_path in split_path.iterdir():
            if not species_path.is_dir():
                continue

            speciesname = species_path.name
            if speciesname not in species_mapping:
                if not ignore_unmapped_species:
                    raise KeyError(f"species {speciesname} not found in the mapping")
                if speciesname not in ignored_species:
                    print(f"species {speciesname} not found in the mapping, ignoring")
                    ignored_species.add(speciesname)
                continue

            mapping_name = species_mapping[speciesname]
            print(speciesname, ": ", mapping_name)
            (outpath / split / mapping_name).mkdir(exist_ok=True, parents=True)

            for input_filepath in species_path.iterdir():
                if input_filepath.suffix in imageformats:
                    (outpath / split / mapping_name / input_filepath.name).symlink_to(
                        input_filepath
                    )

    with open(outpath / "data.yaml", "w") as f:
        yaml.safe_dump(dataset_yaml, f)
