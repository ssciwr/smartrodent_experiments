"""Entry-point behavior of scripts/filter_data_vlm.py."""

from pathlib import Path

import pandas as pd
import pytest
from bundle_test_support import write_stage_config, write_stage_records

from scripts import (
    filter_data_vlm,
)
from smartrodent.filter import VLMFilter


@pytest.mark.parametrize("absolute_input", [False, True])
def test_filter_reads_from_shared_root_and_preserves_paths_for_all_species(
    bundle, tmp_path, monkeypatch, absolute_input
):
    """Inference can read originals without persisting machine-specific paths."""
    dataset_root, records_by_species = bundle
    input_root = dataset_root / "inaturalist"
    output_root = dataset_root / "stage1_filtered"
    input_records = {
        species: records.copy() for species, records in records_by_species.items()
    }
    if absolute_input:
        for records in input_records.values():
            records["image_path"] = records["image_path"].map(
                lambda path: str(dataset_root / path)
            )
    write_stage_records(input_root, input_records)
    snapshots = {
        species: (input_root / species / "records.csv").read_bytes()
        for species in records_by_species
    }
    backend = VLMFilter(
        prompt="animal?",
        system_prompt="",
        labels=["kept", "rejected"],
        taskname="animal",
    )
    seen_images = []

    def classify(image_paths):
        results = []
        for path in image_paths:
            assert Path(path).read_bytes()
            seen_images.append(Path(path).resolve())
            results.append({"label": "kept", "parse_error": False})
        return results

    monkeypatch.setattr(backend, "_classify_paths", classify)
    monkeypatch.setattr(VLMFilter, "from_config", lambda _path: backend)
    config = write_stage_config(
        tmp_path / "filter.yaml",
        {
            "paths": {"input_root": str(input_root), "output_root": str(output_root)},
            "species": ["Mus musculus"],
            "taskname": "animal",
            "labels": ["kept", "rejected"],
        },
    )

    filter_data_vlm.main(config)

    assert seen_images == [
        dataset_root / path for path in records_by_species["Mus musculus"]["image_path"]
    ]
    for species, original in records_by_species.items():
        actual = pd.read_csv(output_root / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert (input_root / species / "records.csv").read_bytes() == snapshots[species]
    assert pd.read_csv(output_root / "Mus musculus" / "records.csv")[
        "kept_animal"
    ].all()
    assert not pd.read_csv(output_root / "Sorex araneus" / "records.csv")[
        "kept_animal"
    ].any()
