"""Shared-root CSV paths and symlinks survive relocation of the dataset bundle."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st
from PIL import Image

from scripts import (
    filter_data_vlm,
    materialize_filtered_data,
    reconcile_filtered_data,
    sample_training_dataset,
    split_dataset,
)
from smartrodent.filter import VLMFilter
from smartrodent.inaturalist import InaturalistDataset
from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)
from speciesnet_test_support import SpeciesNetStub, detection


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Create originals under one stage and run from an unrelated directory."""
    dataset_root = tmp_path / "dataset"
    records_by_species = {}
    for species in ("Mus musculus", "Sorex araneus"):
        rows = []
        for index in range(6):
            image = (
                dataset_root / "inaturalist" / species / "imgs" / f"{100 + index}_0.jpg"
            )
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 10), "red").save(image)
            rows.append(
                {
                    "id": 100 + index,
                    "photo.id": 1000 + index,
                    "photo_index": 0,
                    "photo.url": f"https://example.test/{index}/square.jpg",
                    "photo.license_code": "cc-by",
                    "image_path": image.relative_to(dataset_root).as_posix(),
                }
            )
        records_by_species[species] = pd.DataFrame(rows)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    return dataset_root, records_by_species


def _write_config(path, configuration):
    """Write a temporary stage config without introducing new path settings."""
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


def _write_records(stage_root, records_by_species):
    """Write stage CSVs without rebasing their shared-root image references."""
    for species, records in records_by_species.items():
        records_path = stage_root / species / "records.csv"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(records_path, index=False)


def _move_bundle(dataset_root):
    """Relocate the entire tree so the original location cannot mask broken links."""
    destination = dataset_root.parent / "transferred" / "renamed_dataset"
    destination.parent.mkdir()
    dataset_root.rename(destination)
    return destination


def test_inaturalist_writes_shared_root_relative_paths_and_can_resume_after_move(
    bundle, tmp_path, monkeypatch
):
    """Original-image CSV references are portable and need no prefix migration."""
    dataset_root, records_by_species = bundle
    _write_records(dataset_root / "inaturalist", records_by_species)
    downloader = InaturalistDataset(
        output_path=dataset_root / "inaturalist",
        species=list(records_by_species),
        years=[2024],
    )
    unexpected_download = Mock(side_effect=AssertionError("Images already exist"))
    monkeypatch.setattr(downloader, "_download_photo", unexpected_download)

    try:
        downloader.download_images()
    finally:
        downloader.session.close()

    for species, original in records_by_species.items():
        actual = pd.read_csv(dataset_root / "inaturalist" / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all(not Path(path).is_absolute() for path in actual["image_path"])
    unexpected_download.assert_not_called()

    moved_root = _move_bundle(dataset_root)
    resumed = InaturalistDataset(
        output_path=moved_root / "inaturalist",
        species=list(records_by_species),
        years=[2024],
    )
    monkeypatch.setattr(resumed, "_download_photo", unexpected_download)
    try:
        resumed.download_images()
    finally:
        resumed.session.close()
    for species, original in records_by_species.items():
        actual = pd.read_csv(moved_root / "inaturalist" / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all((moved_root / path).is_file() for path in actual["image_path"])
    unexpected_download.assert_not_called()


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
    _write_records(input_root, input_records)
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
    config = _write_config(
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


@pytest.mark.parametrize("absolute_input", [False, True])
def test_materialized_csvs_and_symlinks_remain_portable_after_bundle_move(
    bundle, tmp_path, absolute_input
):
    """Both partitions reference originals through relative CSV entries and links."""
    dataset_root, records_by_species = bundle
    input_root = dataset_root / "stage3_filtered"
    for records in records_by_species.values():
        records["rejected_animal"] = [False] * 5 + [True]
    input_records = {
        species: frame.copy(deep=True) for species, frame in records_by_species.items()
    }
    if absolute_input:
        for records in input_records.values():
            records["image_path"] = records["image_path"].map(
                lambda path: str(dataset_root / path)
            )
    _write_records(input_root, input_records)
    output_root = dataset_root / "stage4_materialized"
    config = _write_config(
        tmp_path / "materialize.yaml",
        {"input_dir": str(input_root), "output_directory": str(output_root)},
    )

    materialize_filtered_data.main(config)

    for partition in ("kept", "rejected"):
        links = list((output_root / partition).glob("*/imgs/*"))
        assert links
        assert all(
            link.is_symlink() and not link.readlink().is_absolute() for link in links
        )
    moved_root = _move_bundle(dataset_root)
    for partition, selection in (("kept", slice(0, 5)), ("rejected", slice(5, 6))):
        for species, original in records_by_species.items():
            directory = moved_root / output_root.name / partition / species
            actual = pd.read_csv(directory / "records.csv")
            expected_paths = original.iloc[selection]["image_path"].tolist()
            assert actual["image_path"].tolist() == expected_paths
            for path in expected_paths:
                link = directory / "imgs" / Path(path).name
                assert link.resolve(strict=True) == moved_root / path
                assert link.read_bytes() == (moved_root / path).read_bytes()
            pd.testing.assert_frame_equal(
                pd.read_csv(moved_root / input_root.name / species / "records.csv"),
                input_records[species],
            )


def test_reconciliation_preserves_shared_root_paths_after_bundle_move(bundle, tmp_path):
    """Human decisions compare original identities but keep portable CSV strings."""
    dataset_root, records_by_species = bundle
    for partition, selection in (("kept", slice(0, 5)), ("rejected", slice(5, 6))):
        selected = {
            species: frame.iloc[selection].copy()
            for species, frame in records_by_species.items()
        }
        _write_records(dataset_root / "stage4_materialized" / partition, selected)
        for species, records in selected.items():
            image_dir = dataset_root / "stage5_corrected" / partition / species / "imgs"
            image_dir.mkdir(parents=True)
            for path in records["image_path"]:
                (image_dir / Path(path).name).symlink_to(
                    os.path.relpath(dataset_root / path, image_dir)
                )
    moved_root = _move_bundle(dataset_root)
    config = _write_config(
        tmp_path / "reconcile.yaml",
        {
            "input_directory": str(moved_root / "stage4_materialized"),
            "output_directory": str(moved_root / "stage5_corrected"),
            "relevant_subdirs": ["kept", "rejected"],
            "output_columns": ["kept_human", "rejected_human"],
            "allowed_suffixes": [".jpg"],
        },
    )

    reconcile_filtered_data.main(config)

    for partition, selection in (("kept", slice(0, 5)), ("rejected", slice(5, 6))):
        for species, original in records_by_species.items():
            corrected = pd.read_csv(
                moved_root
                / "stage5_corrected"
                / partition
                / species
                / "records_corrected.csv"
            )
            association_columns = ["id", "photo.id", "image_path"]
            pd.testing.assert_frame_equal(
                corrected[association_columns],
                original.iloc[selection][association_columns].reset_index(drop=True),
            )
            assert corrected[f"{partition}_human"].all()
            assert all(
                (moved_root / path).is_file() for path in corrected["image_path"]
            )


@pytest.mark.parametrize("absolute_input", [False, True])
def test_splitting_and_sampling_preserve_shared_root_paths_without_rebasing(
    bundle, tmp_path, absolute_input
):
    """Metadata-only stages keep the same original-image references verbatim."""
    dataset_root, records_by_species = bundle
    corrected_root = dataset_root / "stage5_corrected"
    for species, records in records_by_species.items():
        directory = corrected_root / "kept" / species
        directory.mkdir(parents=True)
        input_records = records.copy()
        if absolute_input:
            input_records["image_path"] = input_records["image_path"].map(
                lambda path: str(dataset_root / path)
            )
        input_records.to_csv(directory / "records_corrected.csv", index=False)
    split_root = dataset_root / "stage6_split"
    split_config = _write_config(
        tmp_path / "split.yaml",
        {
            "paths": {
                "dataset_splitter_input": str(corrected_root),
                "dataset_splitter_output": str(split_root),
                "records_glob": "kept/*/records_corrected.csv",
            },
            "data": {
                "dataset_splitter": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "train_val_test_split": [0.7, 0.15, 0.15],
                    "rng_seed": 42,
                }
            },
        },
    )
    sampled_root = dataset_root / "stage7_sampled"
    sample_config = _write_config(
        tmp_path / "sample.yaml",
        {
            "paths": {
                "training_dataset_sampler_input": str(split_root),
                "training_dataset_sampler_output": str(sampled_root),
            },
            "data": {
                "training_dataset_sampler": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "oversampling_rule": "scripts.sample_training_dataset:compute_oversampling",
                    "rng_seed": 42,
                }
            },
        },
    )

    split_dataset.main(split_config)
    sample_training_dataset.main(sample_config)

    moved_root = _move_bundle(dataset_root)
    for stage in (split_root.name, sampled_root.name):
        for species, original in records_by_species.items():
            actual = pd.read_csv(moved_root / stage / species / "records.csv")
            assert actual["image_path"].tolist() == original["image_path"].tolist()
            assert set(actual["dataset_split"]) == {"train", "validation", "test"}
            assert all((moved_root / path).is_file() for path in actual["image_path"])


def test_sampling_writes_portable_paths_from_legacy_absolute_records(bundle, tmp_path):
    """Sampling also normalizes absolute inputs without altering their source CSVs."""
    dataset_root, records_by_species = bundle
    split_root = dataset_root / "stage6_split"
    input_records = {}
    for species, records in records_by_species.items():
        frame = records.copy()
        frame["dataset_split"] = ["train", "validation", "test"] * 2
        frame["image_path"] = frame["image_path"].map(
            lambda path: str(dataset_root / path)
        )
        input_records[species] = frame
    _write_records(split_root, input_records)
    sampled_root = dataset_root / "stage7_sampled"
    config = _write_config(
        tmp_path / "sample_legacy.yaml",
        {
            "paths": {
                "training_dataset_sampler_input": str(split_root),
                "training_dataset_sampler_output": str(sampled_root),
            },
            "data": {
                "training_dataset_sampler": {
                    "group_columns": ["id"],
                    "stratify_columns": [],
                    "oversampling_rule": "scripts.sample_training_dataset:compute_oversampling",
                }
            },
        },
    )

    sample_training_dataset.main(config)

    moved_root = _move_bundle(dataset_root)
    for species, original in records_by_species.items():
        actual = pd.read_csv(moved_root / sampled_root.name / species / "records.csv")
        assert actual["image_path"].tolist() == original["image_path"].tolist()
        assert all((moved_root / path).is_file() for path in actual["image_path"])
        pd.testing.assert_frame_equal(
            pd.read_csv(moved_root / split_root.name / species / "records.csv"),
            input_records[species],
        )


@pytest.mark.parametrize(
    "creator_type",
    [
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ],
)
def test_yolo_creators_read_shared_root_paths_after_bundle_move(bundle, creator_type):
    """Creators find originals without consulting the old location or the cwd."""
    dataset_root, records_by_species = bundle
    for records in records_by_species.values():
        records["dataset_split"] = ["train", "validation", "test"] * 2
    _write_records(dataset_root / "stage7_sampled", records_by_species)
    moved_root = _move_bundle(dataset_root)

    creator = creator_type(moved_root / "stage7_sampled", moved_root / "stage8_yolo")

    for species, original in records_by_species.items():
        resolved = creator.records_by_species[species]["image_path"]
        assert [Path(path).resolve() for path in resolved] == [
            moved_root / path for path in original["image_path"]
        ]
        stored = pd.read_csv(moved_root / "stage7_sampled" / species / "records.csv")
        assert stored["image_path"].tolist() == original["image_path"].tolist()
    assert not (moved_root / "stage8_yolo").exists()


def test_yolo_detector_symlinks_survive_bundle_move(bundle):
    """Every generated detector split links relatively to the original stage."""
    dataset_root, records_by_species = bundle
    predictions = []
    for records in records_by_species.values():
        records["dataset_split"] = ["train", "validation", "test"] * 2
        records["image_path"] = records["image_path"].map(
            lambda path: str(dataset_root / path)
        )
        predictions.extend(
            {"filepath": path, "detections": [detection(0.9)]}
            for path in records["image_path"]
        )
    _write_records(dataset_root / "stage7_sampled", records_by_species)
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(
        dataset_root / "stage7_sampled",
        dataset_root / "stage8_detector",
        model=SpeciesNetStub(predictions),
    )

    output = creator.create()

    links = list((output / "images").glob("*/*"))
    assert len(links) == 12
    expected_targets = {}
    for link in links:
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        expected_targets[link.relative_to(dataset_root)] = link.resolve().relative_to(
            dataset_root
        )
    moved_root = _move_bundle(dataset_root)
    for relative_link, relative_source in expected_targets.items():
        link = moved_root / relative_link
        source = moved_root / relative_source
        assert link.resolve(strict=True) == source
        assert link.read_bytes() == source.read_bytes()


@given(species=st.from_regex(r"[a-z][a-z0-9 ]{0,12}", fullmatch=True))
def test_generated_species_names_keep_materialized_links_portable(species):
    """Shared-root resolution and relative links also work with species-name spaces."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset_root = root / "dataset"
        source = dataset_root / "inaturalist" / species / "imgs" / "photo.jpg"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original image")
        records = pd.DataFrame(
            {"image_path": [source.relative_to(dataset_root).as_posix()]}
        )
        _write_records(dataset_root / "stage3_filtered", {species: records})
        config = _write_config(
            root / "materialize.yaml",
            {
                "input_dir": str(dataset_root / "stage3_filtered"),
                "output_directory": str(dataset_root / "stage4_materialized"),
            },
        )

        materialize_filtered_data.main(config)

        moved_root = _move_bundle(dataset_root)
        link = (
            moved_root / "stage4_materialized" / "kept" / species / "imgs" / source.name
        )
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        assert link.read_bytes() == b"original image"
        actual = pd.read_csv(link.parent.parent / "records.csv")
        assert actual["image_path"].tolist() == records["image_path"].tolist()
