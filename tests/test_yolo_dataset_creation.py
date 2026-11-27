"""Input, inference wiring, and output contracts for record-based YOLO dataset creation."""

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from bundle_test_support import move_bundle, write_stage_records
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from speciesnet_test_support import SpeciesNetDetectorStub, SpeciesNetStub, detection

from smartrodent import yolo_dataset_creation
from smartrodent.yolo_dataset_creation import (
    ClassifierLayout,
    DatasetCreationPlan,
    DetectorLayout,
    SpeciesNetDetector,
    YoloDatasetCreator,
    load_split_records,
)


@pytest.fixture(params=[DetectorLayout, ClassifierLayout])
def layout(request):
    """Exercise both output layouts with the same input contract."""
    return request.param()


def make_creator(
    input_root,
    output,
    *,
    model=None,
    layout=None,
    class_names=None,
    batch_size=32,
    **detector_settings,
):
    """A creator with an injected detector model and the detector layout by default."""
    return YoloDatasetCreator(
        input_root,
        output,
        layout=layout if layout is not None else DetectorLayout(),
        detector=SpeciesNetDetector(model=model, **detector_settings),
        class_names=class_names,
        batch_size=batch_size,
    )


def stub_for(paths, *detections, **kwargs):
    """A SpeciesNet substitute that returns the same detections for every path."""
    return SpeciesNetStub(
        [
            {"filepath": str(path), "detections": list(detections or [detection(0.9)])}
            for path in paths
        ],
        **kwargs,
    )


def write_species(root, species, splits):
    """Write temporary per-species metadata with real source files."""
    directory = root / species
    directory.mkdir(parents=True)
    rows = []
    for index, split in enumerate(splits):
        image = directory / f"photo_{index}.jpg"
        image.touch()
        rows.append(
            {
                "image_path": str(image.resolve()),
                "dataset_split": split,
                "photo_id": index,
            }
        )
    pd.DataFrame(rows).to_csv(directory / "records.csv", index=False)
    return directory


# --- record loading ---------------------------------------------------------------


def test_classes_are_discovered_in_sorted_order_with_existing_splits(tmp_path):
    """Species folders define sorted classes and existing splits are not resampled."""
    root = tmp_path / "input"
    write_species(root, "shrew", ["test"])
    write_species(root, "mouse", ["train", "validation", "test"])

    records_by_species = load_split_records(root)

    assert list(records_by_species) == ["mouse", "shrew"]
    records = records_by_species["mouse"]
    assert records["dataset_split"].tolist() == ["train", "val", "test"]
    assert records["photo_id"].tolist() == [0, 1, 2]
    assert records["image_path"].tolist() == [
        str((root / "mouse" / f"photo_{index}.jpg").resolve()) for index in range(3)
    ]


def test_explicit_classes_select_directories_and_define_index_order(tmp_path):
    """Unselected directories need no valid metadata, and supplied order is retained."""
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])
    write_species(root, "shrew", ["test"])
    (root / "ignored").mkdir()

    assert list(load_split_records(root, ["shrew", "mouse"])) == ["shrew", "mouse"]


def test_unknown_class_directories_are_rejected(tmp_path):
    """Selections must name immediate species directories, never paths outside them."""
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])

    with pytest.raises(ValueError, match="class"):
        load_split_records(root, ["../mouse"])


@pytest.mark.parametrize("split", ["invalid", None])
def test_invalid_split_assignments_are_rejected(tmp_path, split):
    """No row receives a silently defaulted or recomputed split."""
    root = tmp_path / "input"
    write_species(root, "mouse", [split])

    with pytest.raises(ValueError, match="dataset_split"):
        load_split_records(root)


@pytest.mark.parametrize("column", ["image_path", "dataset_split"])
def test_missing_required_columns_are_reported(tmp_path, column):
    """Every selected species requires image-path and split columns."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv").drop(columns=[column])
    records.to_csv(directory / "records.csv", index=False)

    with pytest.raises(ValueError, match=column):
        load_split_records(root)


def test_relative_image_paths_resolve_against_shared_dataset_root(tmp_path):
    """Relative paths use the stage's parent directory, not its species CSV."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv")
    records["image_path"] = [str((directory / "photo_0.jpg").relative_to(root.parent))]
    records.to_csv(directory / "records.csv", index=False)

    assert load_split_records(root)["mouse"]["image_path"].tolist() == [
        str(directory / "photo_0.jpg")
    ]


@pytest.mark.parametrize("path", [None, "missing.jpg"])
def test_invalid_image_paths_are_reported(tmp_path, path):
    """Blank paths and nonexistent images fail explicitly during input validation."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv")
    records["image_path"] = [path]
    records.to_csv(directory / "records.csv", index=False)
    error = FileNotFoundError if path == "missing.jpg" else ValueError

    with pytest.raises(error, match="image_path"):
        load_split_records(root)


@pytest.mark.parametrize("across_species", [False, True])
def test_same_image_cannot_belong_to_different_splits(tmp_path, across_species):
    """Reject existing split leakage without resampling or dropping records."""
    root = tmp_path / "input"
    mouse = write_species(root, "mouse", ["train"])
    records = pd.read_csv(mouse / "records.csv")
    conflicting = records.copy()
    conflicting["dataset_split"] = "test"
    if across_species:
        shrew = write_species(root, "shrew", ["test"])
        conflicting.to_csv(shrew / "records.csv", index=False)
    else:
        pd.concat([records, conflicting]).to_csv(mouse / "records.csv", index=False)

    with pytest.raises(ValueError, match="multiple splits"):
        load_split_records(root)


@given(indices=st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=20))
def test_record_order_and_oversampled_multiplicity_are_preserved(indices):
    """Every selected source row survives, including repeated oversampled rows."""
    with TemporaryDirectory() as directory:
        root = Path(directory) / "input"
        species = write_species(root, "mouse", ["train", "validation", "test"])
        original = pd.read_csv(species / "records.csv")
        selected = original.iloc[indices].reset_index(drop=True)
        selected.to_csv(species / "records.csv", index=False)
        before = (species / "records.csv").read_bytes()

        loaded = load_split_records(root)["mouse"]

        assert loaded["photo_id"].tolist() == indices
        assert loaded["dataset_split"].tolist() == [
            ["train", "val", "test"][index] for index in indices
        ]
        assert loaded["image_path"].tolist() == selected["image_path"].tolist()
        assert (species / "records.csv").read_bytes() == before


@pytest.mark.parametrize("invalid_root", ["missing", "empty"])
def test_input_requires_a_directory_with_species(tmp_path, invalid_root):
    """Missing and classless inputs have explicit errors."""
    root = tmp_path / "input"
    if invalid_root == "empty":
        root.mkdir()

    with pytest.raises(ValueError):
        load_split_records(root)


def test_selected_species_requires_records_csv(tmp_path):
    """Missing metadata in a selected species is not silently ignored."""
    root = tmp_path / "input"
    (root / "mouse").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="records.csv"):
        load_split_records(root)


def test_header_only_species_records_remain_an_explicit_empty_class(tmp_path):
    """A valid class with no selected rows remains present with its CSV columns."""
    root = tmp_path / "input"
    species = root / "mouse"
    species.mkdir(parents=True)
    (species / "records.csv").write_text("image_path,dataset_split,photo_id\n")

    records = load_split_records(root)["mouse"]

    assert records.empty
    assert records.columns.tolist() == ["image_path", "dataset_split", "photo_id"]


def test_records_without_headers_are_rejected(tmp_path):
    """An empty file is malformed metadata rather than an empty class."""
    root = tmp_path / "input"
    species = root / "mouse"
    species.mkdir(parents=True)
    (species / "records.csv").touch()

    with pytest.raises(ValueError, match="header"):
        load_split_records(root)


def test_records_read_shared_root_paths_after_bundle_move(bundle):
    """Records find originals without consulting the old location or the cwd."""
    dataset_root, records_by_species = bundle
    for records in records_by_species.values():
        records["dataset_split"] = ["train", "validation", "test"] * 2
    write_stage_records(dataset_root / "stage7_sampled", records_by_species)
    moved_root = move_bundle(dataset_root)

    loaded = load_split_records(moved_root / "stage7_sampled")

    for species, original in records_by_species.items():
        assert [Path(path).resolve() for path in loaded[species]["image_path"]] == [
            moved_root / path for path in original["image_path"]
        ]
        stored = pd.read_csv(moved_root / "stage7_sampled" / species / "records.csv")
        assert stored["image_path"].tolist() == original["image_path"].tolist()


# --- detection ----------------------------------------------------------------------


def detect(paths, predictions, *, threshold=0.5):
    """Run the detector with an external model substitute."""
    detector = SpeciesNetDetector(
        model=SpeciesNetStub(predictions),
        confidence_threshold=threshold,
        allowed_classes=("animal",),
        iou_threshold=0.45,
    )
    return detector.detect(paths)


def test_results_preserve_full_paths_and_extract_independent_crops(images):
    """Same-named source images retain distinct metadata and correct crop pixels."""
    result = detect(images, stub_for(images).predictions)

    assert set(result) == {str(path) for path in images}
    for path, color in zip(images, [(255, 0, 0), (0, 0, 255)], strict=True):
        (record,) = result[str(path)]
        assert record["bbox"] == [0.25, 0.2, 0.5, 0.6]
        assert record["confidence"] == 0.9
        assert record["label"] == "animal"
        assert record["detection_index"] == 0
        assert isinstance(record["crop"], Image.Image)
        assert record["crop"].size == (10, 6)
        assert all(
            record["crop"].getpixel((x, y)) == color
            for x in range(10)
            for y in range(6)
        )


def test_confidence_boundary_and_allowed_labels_are_respected(images):
    """The confidence boundary is inclusive and non-animal boxes are excluded."""
    predictions = [
        {
            "filepath": str(images[0]),
            "detections": [
                detection(0.49),
                detection(0.5),
                detection(0.99, label="human"),
            ],
        }
    ]

    (record,) = detect(images[:1], predictions)[str(images[0])]

    assert record["confidence"] == 0.5
    assert record["detection_index"] == 1


def test_nms_keeps_strongest_overlap_and_separate_detection(images):
    """Overlapping boxes are suppressed without losing separate animals."""
    predictions = [
        {
            "filepath": str(images[0]),
            "detections": [
                detection(0.7),
                detection(0.95),
                detection(0.8, bbox=(0.0, 0.0, 0.1, 0.1)),
            ],
        }
    ]

    records = detect(images[:1], predictions)[str(images[0])]

    assert [record["detection_index"] for record in records] == [1, 2]
    assert [record["confidence"] for record in records] == [0.95, 0.8]


@pytest.mark.parametrize("detections", [[], [detection(0.1)]])
def test_images_without_accepted_detections_remain_in_results(images, detections):
    """An empty result is explicit rather than silently dropping its source image."""
    predictions = [{"filepath": str(images[0]), "detections": detections}]

    assert detect(images[:1], predictions) == {str(images[0]): []}


@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_confidence_filter_matches_inclusive_threshold(confidence, threshold):
    """Every valid confidence/threshold pair obeys the same acceptance rule."""
    # Each generated example owns its files; no function-scoped fixture state leaks
    # between Hypothesis examples.
    with TemporaryDirectory() as directory:
        path = (Path(directory) / "source.png").resolve()
        Image.new("RGB", (20, 10), "red").save(path)
        predictions = [{"filepath": str(path), "detections": [detection(confidence)]}]

        result = detect([path], predictions, threshold=threshold)

        assert len(result[str(path)]) == int(confidence >= threshold)


@pytest.mark.parametrize("model_name", [None, "custom-model"])
def test_model_is_loaded_lazily_once_and_reused(monkeypatch, images, model_name):
    """Successive calls reuse one detector loaded with the configured identifier."""
    constructor = Mock(return_value=stub_for(images))
    # Patch the external SDK symbols where the library imports them.
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    monkeypatch.setattr(yolo_dataset_creation, "DEFAULT_MODEL", "test-default")

    detector = SpeciesNetDetector(model_name=model_name)
    assert detector.detect([]) == {}
    constructor.assert_not_called()

    for path in images:
        assert detector.detect([path])[str(path)][0]["confidence"] == 0.9

    constructor.assert_called_once_with(
        "test-default" if model_name is None else model_name,
        components="detector",
    )


def test_injected_model_is_used_instead_of_loading_one(monkeypatch, images):
    constructor = Mock(side_effect=AssertionError("Injected model must be reused"))
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    detector = SpeciesNetDetector(model=stub_for(images))

    for _ in range(2):
        assert set(detector.detect(images)) == {str(path) for path in images}

    constructor.assert_not_called()


def test_conflicting_model_settings_are_rejected():
    """A supplied model and a model identifier cannot silently override each other."""
    with pytest.raises(ValueError):
        SpeciesNetDetector(model=SpeciesNetStub([]), model_name="other-model")


def test_missing_model_result_is_not_silently_dropped(images):
    """A missing prediction is an inference failure, not a negative detection."""
    with pytest.raises(ValueError, match="prediction"):
        detect(images, [])


@pytest.mark.parametrize("fail", [False, True])
def test_external_detector_cutoff_is_restored_after_inference(images, fail):
    """The caller's detector setting survives successful and failed inference."""
    previous_threshold = SpeciesNetDetectorStub.DETECTION_THRESHOLD
    detector = SpeciesNetDetector(model=stub_for(images[:1], fail=fail))

    if fail:
        with pytest.raises(RuntimeError, match="Model failed"):
            detector.detect(images[:1])
    else:
        detector.detect(images[:1])

    assert SpeciesNetDetectorStub.DETECTION_THRESHOLD == previous_threshold


@pytest.mark.speciesnet_integration
def test_real_speciesnet_api_returns_detection_metadata_and_crops(tmp_path):
    """Infer real boxes/crops on a bundled photograph, without an SDK substitute.

    Run with RUN_SPECIESNET_INTEGRATION=1. SPECIESNET_MODEL may point to a local
    weights directory to avoid downloads; otherwise the SDK uses its default
    model and may download weights into its normal cache.
    """
    if os.environ.get("RUN_SPECIESNET_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPECIESNET_INTEGRATION=1 to run real model inference")

    from ultralytics.utils import ASSETS

    # This installed sample has people and a bus, so use those detector labels
    # rather than asserting animal detection on an artificial test image.
    paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in paths:
        shutil.copy2(ASSETS / "bus.jpg", path)

    detector = SpeciesNetDetector(
        model_name=os.environ.get("SPECIESNET_MODEL"),
        confidence_threshold=0.5,
        allowed_classes=("human", "vehicle"),
        iou_threshold=0.45,
    )
    result = detector.detect(paths)

    assert set(result) == {str(path) for path in paths}
    for path in paths:
        records = result[str(path)]
        assert records, "Real inference must produce at least one accepted box"
        with Image.open(path) as source:
            source = source.convert("RGB")
            for record in records:
                assert 0.5 <= record["confidence"] <= 1.0
                assert record["label"] in {"human", "vehicle"}
                assert len(record["bbox"]) == 4
                assert record["detection_index"] >= 0
                crop = record["crop"]
                assert crop.mode == "RGB"
                x, y, width, height = record["bbox"]
                expected = source.crop(
                    (
                        x * source.width,
                        y * source.height,
                        (x + width) * source.width,
                        (y + height) * source.height,
                    )
                )
                assert crop.size == expected.size
                assert crop.tobytes() == expected.tobytes()


# --- dataset creation ---------------------------------------------------------------


@pytest.fixture
def creator_input(images):
    """Attach split metadata, including oversampling, to real temporary images."""
    for path, split, repeats in zip(images, ["train", "test"], [2, 1], strict=True):
        pd.DataFrame(
            [
                {"image_path": str(path), "dataset_split": split, "photo_id": 42}
                for _ in range(repeats)
            ]
        ).to_csv(path.parent / "records.csv", index=False)
    return images[0].parent.parent


def test_preparing_a_creator_loads_no_model_and_writes_nothing(
    creator_input, tmp_path, monkeypatch
):
    """Input setup reads records only; inference and output wait for ``create``."""
    monkeypatch.setattr(
        yolo_dataset_creation, "SpeciesNet", Mock(side_effect=AssertionError("load"))
    )
    output = tmp_path / "output"

    creator = YoloDatasetCreator(
        creator_input,
        output,
        layout=DetectorLayout(),
        detector=SpeciesNetDetector(model_name="custom-speciesnet"),
        class_names=["shrew", "mouse"],
    )

    assert creator.class_names == ["shrew", "mouse"]
    assert len(creator.records_by_species["mouse"]) == 2
    assert not output.exists()


def test_detector_layout_symlinks_sources_and_writes_yolo_labels(
    creator_input, images, tmp_path
):
    """Detector output preserves supplied splits and each oversampled occurrence."""
    output = tmp_path / "detector"
    creator = make_creator(
        creator_input, output, class_names=["shrew", "mouse"], model=stub_for(images)
    )

    assert creator.create() == output

    for split in ["train", "val", "test"]:
        assert (output / "images" / split).is_dir()
        assert (output / "labels" / split).is_dir()
    for split, source, count, class_index in [
        ("train", images[0], 2, "1"),
        ("test", images[1], 1, "0"),
    ]:
        links = list((output / "images" / split).iterdir())
        assert len(links) == count
        assert len({path.name for path in links}) == count
        for link in links:
            assert link.is_symlink()
            assert link.resolve() == source
            label = output / "labels" / split / f"{link.stem}.txt"
            assert label.read_text().split() == [
                class_index,
                "0.5",
                "0.5",
                "0.5",
                "0.6",
            ]
    assert not list((output / "images" / "val").iterdir())
    data = yaml.safe_load((output / "data.yaml").read_text())
    assert data == {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "shrew", 1: "mouse"},
    }


def test_classifier_layout_saves_crops_in_assigned_class_and_split(
    creator_input, images, tmp_path
):
    """Classifier outputs are actual crops, with duplicates retained in one split."""
    output = tmp_path / "classifier"
    creator = make_creator(
        creator_input, output, layout=ClassifierLayout(), model=stub_for(images)
    )

    assert creator.create() == output

    for split in ["train", "val", "test"]:
        for species in ["mouse", "shrew"]:
            assert (output / split / species).is_dir()
    for split, species, count, color in [
        ("train", "mouse", 2, (255, 0, 0)),
        ("test", "shrew", 1, (0, 0, 255)),
    ]:
        crops = list((output / split / species).iterdir())
        assert len(crops) == count
        assert len({path.name for path in crops}) == count
        for path in crops:
            assert not path.is_symlink()
            with Image.open(path) as crop:
                assert crop.size == (10, 6)
                assert all(
                    abs(actual - expected) <= 2
                    for actual, expected in zip(
                        crop.getpixel((5, 3)), color, strict=True
                    )
                )
    assert not list((output / "val").rglob("*.png"))
    data = yaml.safe_load((output / "data.yaml").read_text())
    assert data == {
        "path": str(output),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {0: "mouse", 1: "shrew"},
    }


def test_classifier_crops_drop_source_icc_profiles_but_keep_pixels(
    creator_input, images, tmp_path
):
    """Malformed source colour profiles are not copied into the saved crops."""
    # Arbitrary bytes mimic the malformed profiles found in real source images;
    # PIL embeds them unvalidated, exactly like the broken ones in the dataset.
    broken_profile = b"not a valid ICC profile"
    for path in images:
        with Image.open(path) as source:
            pixels = source.copy()
        pixels.save(path, icc_profile=broken_profile)
        with Image.open(path) as reloaded:
            assert reloaded.info["icc_profile"] == broken_profile
    output = tmp_path / "classifier"

    make_creator(
        creator_input, output, layout=ClassifierLayout(), model=stub_for(images)
    ).create()

    crops = sorted(output.rglob("*.png"))
    assert len(crops) == 3
    for path in crops:
        expected = (255, 0, 0) if path.parent.name == "mouse" else (0, 0, 255)
        with Image.open(path) as crop:
            assert "icc_profile" not in crop.info
            assert crop.getpixel((5, 3)) == expected


def test_create_skips_images_without_accepted_boxes_and_records_why(
    creator_input, images, tmp_path
):
    """Empty/filtered detections produce no examples but retain every metadata row."""
    model = SpeciesNetStub(
        [
            {"filepath": str(images[0]), "detections": []},
            {"filepath": str(images[1]), "detections": [detection(0.01)]},
        ]
    )
    output = tmp_path / "output"

    assert make_creator(creator_input, output, model=model).create() == output

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [record["image_path"] for record in metadata] == [
        str(images[0]),
        str(images[0]),
        str(images[1]),
    ]
    assert [record["species"] for record in metadata] == ["mouse", "mouse", "shrew"]
    assert [record["dataset_split"] for record in metadata] == [
        "train",
        "train",
        "test",
    ]
    for record in metadata:
        assert record["photo_id"] == 42
        assert record["status"] == "skipped"
        assert record["skip_reason"] == "no_accepted_detections"
        assert record["outputs"] == []
    assert not list(output.rglob("*.txt"))
    assert not list(output.rglob("*.png"))


@pytest.mark.parametrize("destination", ["input", "inside_input", "parent"])
def test_create_rejects_destinations_overlapping_input(
    creator_input, tmp_path, destination
):
    """A dataset output must not overwrite or contaminate the source input tree."""
    if destination == "input":
        output = creator_input
    elif destination == "inside_input":
        output = creator_input / "generated"
    else:
        output = creator_input.parent
    creator = make_creator(creator_input, output, model=SpeciesNetStub([], fail=True))

    with pytest.raises(ValueError, match="overlap"):
        creator.create()

    assert not (output / "metadata.json").exists()
    assert not (creator_input / "generated").exists()


def test_create_rejects_existing_output_without_overwriting(creator_input, tmp_path):
    """Recreating a dataset requires an explicit new destination, not silent overwrite."""
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep me")
    creator = make_creator(creator_input, output, model=SpeciesNetStub([], fail=True))

    with pytest.raises(FileExistsError):
        creator.create()

    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "keep me"


def test_empty_classes_create_layout_without_inference(layout, tmp_path):
    """An empty but valid input produces configuration and empty provenance."""
    root = tmp_path / "input"
    (root / "mouse").mkdir(parents=True)
    (root / "mouse" / "records.csv").write_text("image_path,dataset_split\n")
    output = tmp_path / "output"
    creator = make_creator(
        root, output, layout=layout, model=SpeciesNetStub([], fail=True)
    )

    assert creator.create() == output

    assert json.loads((output / "metadata.json").read_text()) == {"records": []}
    assert yaml.safe_load((output / "data.yaml").read_text())["names"] == {0: "mouse"}


def test_create_persists_successful_detection_and_output_provenance(
    layout, creator_input, images, tmp_path
):
    """Written examples retain source metadata and JSON-safe detection/output records."""
    records_path = images[0].parent / "records.csv"
    frame = pd.read_csv(records_path)
    frame["extra_value"] = [None, 1.25]
    frame.to_csv(records_path, index=False)
    model = SpeciesNetStub(
        [
            {"filepath": str(images[0]), "detections": [detection(0.9)]},
            {"filepath": str(images[1]), "detections": []},
        ]
    )
    output = tmp_path / "output"
    creator = make_creator(creator_input, output, layout=layout, model=model)
    original = {
        species: records.copy(deep=True)
        for species, records in creator.records_by_species.items()
    }

    creator.create()

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [record["status"] for record in metadata] == [
        "written",
        "written",
        "skipped",
    ]
    assert [record["source_row"] for record in metadata] == [0, 1, 0]
    assert metadata[0]["extra_value"] is None
    assert metadata[1]["extra_value"] == 1.25
    for record in metadata[:2]:
        assert record["image_path"] == str(images[0])
        assert record["species"] == "mouse"
        assert record["dataset_split"] == "train"
        assert record["photo_id"] == 42
        assert record["skip_reason"] is None
        assert record["detections"] == [
            {
                "bbox": [0.25, 0.2, 0.5, 0.6],
                "confidence": 0.9,
                "label": "animal",
                "detection_index": 0,
            }
        ]
        assert len(record["outputs"]) == 1
        for item in record["outputs"]:
            relative_path = Path(item["image_path"])
            assert not relative_path.is_absolute()
            assert (output / relative_path).is_file()
    assert metadata[0]["outputs"] != metadata[1]["outputs"]
    for species, records in creator.records_by_species.items():
        pd.testing.assert_frame_equal(records, original[species])


def test_create_preserves_loaded_numeric_metadata_precision(
    creator_input, images, tmp_path
):
    """Serializing provenance must not further round the metadata loaded from CSV."""
    records_path = images[0].parent / "records.csv"
    frame = pd.read_csv(records_path)
    frame["measurement"] = [0.12345678901234567, 1.2345678901234567]
    frame.to_csv(records_path, index=False)
    output = tmp_path / "output"
    creator = make_creator(
        creator_input,
        output,
        model=SpeciesNetStub(
            [{"filepath": str(path), "detections": []} for path in images]
        ),
    )
    expected = creator.records_by_species["mouse"]["measurement"].tolist()

    creator.create()

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [row["measurement"] for row in metadata[:2]] == expected


def test_create_writes_every_accepted_detection(
    layout, creator_input, images, tmp_path
):
    """Multiple boxes become multiple labels or crops, not a single selected animal."""
    model = SpeciesNetStub(
        [
            {
                "filepath": str(images[0]),
                "detections": [
                    detection(0.9),
                    detection(0.8, bbox=(0.0, 0.0, 0.1, 0.1)),
                ],
            },
            {"filepath": str(images[1]), "detections": []},
        ]
    )
    output = tmp_path / "output"

    make_creator(creator_input, output, layout=layout, model=model).create()

    if isinstance(layout, DetectorLayout):
        labels = list((output / "labels" / "train").glob("*.txt"))
        assert len(labels) == 2
        for label in labels:
            assert label.read_text().splitlines() == [
                "0 0.5 0.5 0.5 0.6",
                "0 0.05 0.05 0.1 0.1",
            ]
    elif isinstance(layout, ClassifierLayout):
        paths = list((output / "train" / "mouse").iterdir())
        assert len(paths) == 4
        sizes = []
        for path in paths:
            with Image.open(path) as crop:
                sizes.append(crop.size)
        assert sizes.count((10, 6)) == 2
        assert sizes.count((2, 1)) == 2
    else:
        pytest.fail("Unexpected layout")


@pytest.mark.parametrize("batch_size", [7, 40])  # uneven batches, last one partial
def test_create_infers_in_bounded_batches(tmp_path, batch_size):
    """A capacity-limited detector processes a dataset larger than one batch."""
    root = tmp_path / "input"
    species = root / "mouse"
    species.mkdir(parents=True)
    paths = [species / f"photo_{index}.png" for index in range(65)]
    for path in paths:
        Image.new("RGB", (20, 10), "red").save(path)
    pd.DataFrame(
        [{"image_path": str(path), "dataset_split": "train"} for path in paths]
    ).to_csv(species / "records.csv", index=False)
    model = stub_for(paths, max_batch_size=batch_size)
    model.detect = Mock(wraps=model.detect)
    output = tmp_path / "output"

    make_creator(root, output, model=model, batch_size=batch_size).create()

    assert [len(call.kwargs["filepaths"]) for call in model.detect.call_args_list] == [
        min(batch_size, 65 - start) for start in range(0, 65, batch_size)
    ]
    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [row["source_row"] for row in metadata] == list(range(65))
    assert all(row["status"] == "written" for row in metadata)
    assert len({row["outputs"][0]["image_path"] for row in metadata}) == 65


@given(indices=st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=8))
def test_create_filenames_are_deterministic_and_preserve_oversampling(indices):
    """Repeated rows and same-named sources remain distinct in every output root."""
    with TemporaryDirectory() as directory:
        root = Path(directory) / "input"
        species = root / "mouse"
        species.mkdir(parents=True)
        paths = []
        for index in range(3):
            source_dir = species / f"source_{index}"
            source_dir.mkdir()
            # Non-first observations must not be filtered by this format transformer.
            path = source_dir / "observation_1.png"
            Image.new("RGB", (20, 10), "red").save(path)
            paths.append(path.resolve())
        pd.DataFrame(
            [
                {
                    "image_path": str(paths[index]),
                    "dataset_split": "train",
                    "photo_id": index,
                }
                for index in indices
            ]
        ).to_csv(species / "records.csv", index=False)
        model = stub_for(paths)
        output_names = []
        for output in [Path(directory) / "first", Path(directory) / "second"]:
            make_creator(root, output, model=model).create()
            metadata = json.loads((output / "metadata.json").read_text())["records"]
            assert [row["photo_id"] for row in metadata] == indices
            names = [row["outputs"][0]["image_path"] for row in metadata]
            assert len(names) == len(set(names)) == len(indices)
            assert all((output / name).is_file() for name in names)
            output_names.append(names)
        assert output_names[0] == output_names[1]


def test_detector_symlinks_survive_bundle_move(bundle):
    """Every generated detector split links relatively to the original stage."""
    dataset_root, records_by_species = bundle
    sources = []
    for records in records_by_species.values():
        records["dataset_split"] = ["train", "validation", "test"] * 2
        records["image_path"] = records["image_path"].map(
            lambda path: str(dataset_root / path)
        )
        sources.extend(records["image_path"])
    write_stage_records(dataset_root / "stage7_sampled", records_by_species)

    output = make_creator(
        dataset_root / "stage7_sampled",
        dataset_root / "stage8_detector",
        model=stub_for(sources),
    ).create()

    links = list((output / "images").glob("*/*"))
    assert len(links) == 12
    expected_targets = {}
    for link in links:
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        expected_targets[link.relative_to(dataset_root)] = link.resolve().relative_to(
            dataset_root
        )
    moved_root = move_bundle(dataset_root)
    for relative_link, relative_source in expected_targets.items():
        link = moved_root / relative_link
        source = moved_root / relative_source
        assert link.resolve(strict=True) == source
        assert link.read_bytes() == source.read_bytes()


@pytest.fixture
def ultralytics_input(tmp_path):
    """Provide valid, nonempty splits for testing real Ultralytics dataset loading."""
    root = tmp_path / "input"
    sources = []
    for species, color in [("mouse", "red"), ("shrew", "blue")]:
        directory = root / species
        directory.mkdir(parents=True)
        rows = []
        for split in ["train", "validation", "test"]:
            image_path = directory / f"{split}.png"
            # Training loaders validate dimensions. Use realistic sizes rather
            # than the small crop fixtures used to test pixel-exact extraction.
            Image.new("RGB", (64, 64), color).save(image_path)
            row = {"image_path": str(image_path), "dataset_split": split}
            rows.append(row)
            if species == "mouse" and split == "train":
                rows.append(dict(row))
            sources.append(image_path)
        pd.DataFrame(rows).to_csv(directory / "records.csv", index=False)
    return root, stub_for(sources)


@pytest.mark.parametrize(
    "split, expected_length", [("train", 3), ("val", 2), ("test", 2)]
)
def test_detector_output_loads_through_real_ultralytics_dataset(
    ultralytics_input, tmp_path, monkeypatch, split, expected_length
):
    """Ultralytics decodes symlinked images and their labels without losing duplicates."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import utils as data_utils
    from ultralytics.data.dataset import YOLODataset

    # Font downloading belongs to plot rendering, not dataset validation. Disable
    # only that network side effect; YAML checks and image/label loading stay real.
    monkeypatch.setattr(data_utils, "check_font", lambda font: None)
    root, model = ultralytics_input
    output = tmp_path / "detector"
    make_creator(root, output, class_names=["shrew", "mouse"], model=model).create()

    data = data_utils.check_det_dataset(output / "data.yaml", autodownload=False)
    dataset = YOLODataset(
        img_path=data[split],
        data=data,
        imgsz=64,
        batch_size=2,
        augment=False,
        cache=False,
        hyp=get_cfg(),
    )

    assert data["names"] == {0: "shrew", 1: "mouse"}
    assert len(dataset) == expected_length
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        assert tuple(sample["img"].shape) == (3, 64, 64)
        source = Path(sample["im_file"])
        assert source.is_symlink()
        species = source.resolve().parent.name
        class_index = int(sample["cls"].item())
        assert class_index == {"mouse": 1, "shrew": 0}[species]
        assert sample["bboxes"].flatten().tolist() == pytest.approx(
            [0.5, 0.5, 0.5, 0.6]
        )
        labels.append(class_index)
    assert sorted(labels) == ([0, 1, 1] if split == "train" else [0, 1])


@pytest.mark.parametrize(
    "split, expected_length", [("train", 3), ("val", 2), ("test", 2)]
)
def test_classifier_output_loads_through_real_ultralytics_dataset(
    ultralytics_input, tmp_path, split, expected_length
):
    """Ultralytics reads saved crops and derives class ids from species folders."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.data.utils import check_cls_dataset

    root, model = ultralytics_input
    output = tmp_path / "classifier"
    make_creator(
        root,
        output,
        layout=ClassifierLayout(),
        class_names=["shrew", "mouse"],
        model=model,
    ).create()

    data = check_cls_dataset(output, split=split)
    dataset = ClassificationDataset(
        root=data[split],
        args=get_cfg(overrides={"imgsz": 64, "cache": False}),
        augment=False,
        names=data["names"],
    )

    # Unlike detection, classification's library checker obtains its class order
    # from the folders, not data.yaml. Exercise that actual training contract.
    assert data["names"] == {0: "mouse", 1: "shrew"}
    assert len(dataset) == expected_length
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        assert tuple(sample["img"].shape) == (3, 64, 64)
        class_index = int(sample["cls"])
        strongest_channel = int(sample["img"].mean(dim=(1, 2)).argmax())
        assert strongest_channel == {0: 0, 1: 2}[class_index]
        labels.append(class_index)
    assert sorted(labels) == ([0, 0, 1] if split == "train" else [0, 1])


# --- configuration ------------------------------------------------------------------


def write_plan(
    config_dir, datasets, *, layout="detector", batch_size=32, **detector_parameters
):
    """Write a dataset-creation configuration and return its path."""
    config_path = config_dir / "creator.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "detector": {
                    "name": "SpeciesNetDetector",
                    "parameters": detector_parameters,
                },
                "layout": layout,
                "class_names": ["shrew", "mouse"],
                "batch_size": batch_size,
                "datasets": datasets,
            }
        )
    )
    return config_path


def test_configured_creators_keep_class_order_and_load_no_model(
    creator_input, tmp_path, monkeypatch
):
    """Preparing configured creators reads records in class order and loads no model."""
    monkeypatch.setattr(
        yolo_dataset_creation, "SpeciesNet", Mock(side_effect=AssertionError("load"))
    )
    output = tmp_path / "stage8_yolo_dataset"
    config_path = write_plan(
        tmp_path,
        [
            {
                "path_to_image_data": str(creator_input),
                "dataset_output_path": str(output),
            }
        ],
        batch_size=7,
        model_name="custom-speciesnet",
    )

    (creator,) = DatasetCreationPlan.from_yaml(config_path).creators()

    # Both layouts are created through a configuration in the script tests.
    assert creator.batch_size == 7
    assert creator.class_names == ["shrew", "mouse"]
    assert list(creator.records_by_species) == ["shrew", "mouse"]
    assert not output.exists()


def test_configured_paths_resolve_against_the_working_directory(
    creator_input, tmp_path, monkeypatch
):
    """Relative paths follow the working directory, not the YAML file's directory."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = write_plan(
        config_dir,
        [{"path_to_image_data": "input", "dataset_output_path": "stage8_yolo_dataset"}],
    )

    (creator,) = DatasetCreationPlan.from_yaml(config_path).creators()

    assert creator.path_to_image_data == creator_input
    assert creator.dataset_output_path == tmp_path / "stage8_yolo_dataset"


def test_configuration_cannot_inject_a_detector_instance(creator_input, tmp_path):
    """A YAML value cannot stand in for an injected detector model."""
    config_path = write_plan(
        tmp_path,
        [{"path_to_image_data": str(creator_input), "dataset_output_path": "out"}],
        model="speciesnet",
    )

    with pytest.raises(ValueError, match="model"):
        DatasetCreationPlan.from_yaml(config_path)


def test_one_detector_is_shared_by_every_configured_dataset(
    creator_input, images, tmp_path, monkeypatch
):
    """Weights load once per plan, not once per dataset."""
    constructor = Mock(return_value=stub_for(images))
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    outputs = [tmp_path / "first", tmp_path / "second"]
    config_path = write_plan(
        tmp_path,
        [
            {"path_to_image_data": str(creator_input), "dataset_output_path": str(path)}
            for path in outputs
        ],
    )

    assert DatasetCreationPlan.from_yaml(config_path).create() == outputs

    constructor.assert_called_once()
    assert all((output / "metadata.json").is_file() for output in outputs)


@pytest.mark.parametrize(
    "confidence, labels, iou, indices",
    [
        (0.7, ["human"], 1.0, [0, 1]),
        (0.8, ["animal"], 0.0, [2]),
        (0.95, ["human"], 0.0, []),
        (0.0, [], 1.0, []),
    ],
)
def test_configured_filtering_controls_created_dataset(
    creator_input, images, tmp_path, monkeypatch, confidence, labels, iou, indices
):
    """YAML filtering settings control retained boxes and actual dataset outputs."""
    model = stub_for(
        images,
        detection(0.9, label="human"),
        detection(0.7, label="human"),
        detection(0.95, label="animal"),
    )
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", lambda *a, **k: model)
    output = tmp_path / "configured_output"
    config_path = write_plan(
        tmp_path,
        [
            {
                "path_to_image_data": str(creator_input),
                "dataset_output_path": str(output),
            }
        ],
        confidence_threshold=confidence,
        allowed_classes=labels,
        iou_threshold=iou,
    )

    DatasetCreationPlan.from_yaml(config_path).create()

    records = json.loads((output / "metadata.json").read_text())["records"]
    assert records
    for record in records:
        assert [item["detection_index"] for item in record["detections"]] == indices
        assert record["status"] == ("written" if indices else "skipped")
        for item in record["outputs"]:
            assert (output / item["image_path"]).is_file()
