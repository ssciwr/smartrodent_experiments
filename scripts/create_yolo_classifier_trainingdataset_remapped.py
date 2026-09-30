"""Remap classes in an existing YOLO classification dataset."""

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "coco_style_trainingdataset_creation_config.yaml"
)
DEFAULT_MAPPING = "reduced_rodenspecies_mapping"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load remapping configuration and resolve its input and output paths."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    for key in ("input_data_path", "output_path"):
        path = Path(config[key])
        config[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
    return config


def remap_dataset(
    config_path: str | Path = DEFAULT_CONFIG,
    mapping: str = DEFAULT_MAPPING,
) -> Path:
    """Create a symlink-based classification dataset using a class mapping.

    Args:
        config_path: YAML file describing source, destination, and mappings.
        mapping: Name of the mapping to apply.

    Returns:
        Root of the remapped dataset.

    Raises:
        KeyError: If the mapping or a required source class is unknown.
    """
    config = load_config(config_path)
    input_root = Path(config["input_data_path"])
    output_root = Path(config["output_path"])
    image_formats = {suffix.lower() for suffix in config["image_formats"]}
    ignore_unmapped = bool(config.get("ignore_unmapped_species", False))
    try:
        species_mapping = config["mappings"][mapping]
    except KeyError as error:
        raise KeyError(f"Unknown species mapping: {mapping}") from error

    output_root.mkdir(parents=True, exist_ok=True)
    class_names = sorted(set(species_mapping.values()))
    metadata = {
        "names": dict(enumerate(class_names)),
        "path": str(output_root),
        "test": "test",
        "train": "train",
        "val": "val",
    }

    ignored_species: set[str] = set()
    for split in ("train", "val", "test"):
        split_root = input_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {split_root}")
        for species_path in sorted(path for path in split_root.iterdir() if path.is_dir()):
            species = species_path.name
            if species not in species_mapping:
                if not ignore_unmapped:
                    raise KeyError(f"Species {species} not found in mapping {mapping}")
                ignored_species.add(species)
                continue

            destination = output_root / split / species_mapping[species]
            destination.mkdir(parents=True, exist_ok=True)
            for source in species_path.iterdir():
                if source.is_file() and source.suffix.lower() in image_formats:
                    (destination / source.name).symlink_to(source.resolve())

    if ignored_species:
        print("Ignored unmapped species: " + ", ".join(sorted(ignored_species)))
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False)
    )
    return output_root


def main(argv: Sequence[str] | None = None) -> None:
    """Run classifier dataset remapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    arguments = parser.parse_args(argv)
    output = remap_dataset(arguments.config, arguments.mapping)
    print(f"Remapped YOLO classification dataset written to {output}")


if __name__ == "__main__":
    main()
