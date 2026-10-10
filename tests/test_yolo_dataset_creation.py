"""Input, inference wiring, and output contracts for record-based YOLO creators."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import json

from hypothesis import given, strategies as st
import pandas as pd
import pytest
import yaml
from PIL import Image

from smartrodent import yolo_dataset_creation
from smartrodent.base import Configurable
from speciesnet_test_support import SpeciesNetStub, detection
from smartrodent.yolo_dataset_creation import (
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)


@pytest.fixture(
    params=[
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ]
)
def creator_type(request):
    """Exercise both public creators with the same input contract."""
    return request.param


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


def test_creators_discover_classes_and_load_existing_split_records(
    creator_type, tmp_path
):
    """Species folders define sorted classes and existing splits are not resampled."""
    root = tmp_path / "input"
    write_species(root, "shrew", ["test"])
    write_species(root, "mouse", ["train", "validation", "test"])
    output = tmp_path / "output"

    creator = creator_type(path_to_image_data=root, dataset_output_path=output)

    assert creator.class_names == ["mouse", "shrew"]
    assert creator.classes == {0: "mouse", 1: "shrew"}
    assert list(creator.records_by_species) == ["mouse", "shrew"]
    records = creator.records_by_species["mouse"]
    assert records["dataset_split"].tolist() == ["train", "val", "test"]
    assert records["photo_id"].tolist() == [0, 1, 2]
    assert records["image_path"].tolist() == [
        str((root / "mouse" / f"photo_{index}.jpg").resolve()) for index in range(3)
    ]
    assert not output.exists()


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "32", None])
def test_constructor_rejects_invalid_batch_size(creator_type, tmp_path, batch_size):
    """Invalid batch limits fail before loading records or creating output."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        creator_type(tmp_path / "missing", tmp_path / "output", batch_size=batch_size)
    assert not (tmp_path / "output").exists()


def test_explicit_classes_select_directories_and_define_index_order(
    creator_type, tmp_path
):
    """Unselected directories need no valid metadata, and supplied order is retained."""
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])
    write_species(root, "shrew", ["test"])
    (root / "ignored").mkdir()
    requested = ["shrew", "mouse"]

    creator = creator_type(root, tmp_path / "output", class_names=requested)
    requested.reverse()

    assert creator.class_names == ["shrew", "mouse"]
    assert creator.classes == {0: "shrew", 1: "mouse"}
    assert list(creator.records_by_species) == ["shrew", "mouse"]


@pytest.mark.parametrize(
    "class_names", [[], ["missing"], ["mouse", "mouse"], ["../mouse"]]
)
def test_invalid_class_selection_is_rejected(creator_type, tmp_path, class_names):
    """Selections must be nonempty, unique immediate species-directory names."""
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])

    with pytest.raises(ValueError, match="class"):
        creator_type(root, tmp_path / "output", class_names=class_names)


@pytest.mark.parametrize("split", ["invalid", None, ""])
def test_invalid_split_assignments_are_rejected(creator_type, tmp_path, split):
    """No row receives a silently defaulted or recomputed split."""
    root = tmp_path / "input"
    write_species(root, "mouse", [split])

    with pytest.raises(ValueError, match="dataset_split"):
        creator_type(root, tmp_path / "output")


@pytest.mark.parametrize("column", ["image_path", "dataset_split"])
def test_missing_required_columns_are_reported(creator_type, tmp_path, column):
    """Every selected species requires image-path and split columns."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv").drop(columns=[column])
    records.to_csv(directory / "records.csv", index=False)

    with pytest.raises(ValueError, match=column):
        creator_type(root, tmp_path / "output")


def test_relative_image_paths_resolve_against_shared_dataset_root(
    creator_type, tmp_path
):
    """Relative paths use the stage's parent directory, not its species CSV."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv")
    records["image_path"] = [str((directory / "photo_0.jpg").relative_to(root.parent))]
    records.to_csv(directory / "records.csv", index=False)

    creator = creator_type(root, tmp_path / "output")

    assert creator.records_by_species["mouse"]["image_path"].tolist() == [
        str(directory / "photo_0.jpg")
    ]


@pytest.mark.parametrize("path", [None, "", "missing.jpg"])
def test_invalid_image_paths_are_reported(creator_type, tmp_path, path):
    """Blank paths and nonexistent images fail explicitly during input validation."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv")
    records["image_path"] = [path]
    records.to_csv(directory / "records.csv", index=False)
    error = FileNotFoundError if path == "missing.jpg" else ValueError

    with pytest.raises(error, match="image_path"):
        creator_type(root, tmp_path / "output")


@pytest.mark.parametrize("across_species", [False, True])
def test_same_image_cannot_belong_to_different_splits(
    creator_type, tmp_path, across_species
):
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
        creator_type(root, tmp_path / "output")


@pytest.mark.parametrize(
    "creator_type",
    [
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ],
)
@given(indices=st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=20))
def test_record_order_and_oversampled_multiplicity_are_preserved(creator_type, indices):
    """Every selected source row survives, including repeated oversampled rows."""
    with TemporaryDirectory() as directory:
        root = Path(directory) / "input"
        species = write_species(root, "mouse", ["train", "validation", "test"])
        original = pd.read_csv(species / "records.csv")
        selected = original.iloc[indices].reset_index(drop=True)
        selected.to_csv(species / "records.csv", index=False)
        before = (species / "records.csv").read_bytes()

        creator = creator_type(root, Path(directory) / "output")

        loaded = creator.records_by_species["mouse"]
        assert loaded["photo_id"].tolist() == indices
        assert loaded["dataset_split"].tolist() == [
            ["train", "val", "test"][index] for index in indices
        ]
        assert loaded["image_path"].tolist() == selected["image_path"].tolist()
        assert (species / "records.csv").read_bytes() == before


@pytest.mark.parametrize("invalid_root", ["missing", "file", "empty"])
def test_input_requires_a_directory_with_species(creator_type, tmp_path, invalid_root):
    """Missing, non-directory, and classless inputs have explicit errors."""
    root = tmp_path / "input"
    if invalid_root == "file":
        root.touch()
    elif invalid_root == "empty":
        root.mkdir()

    with pytest.raises(ValueError):
        creator_type(root, tmp_path / "output")


def test_selected_species_requires_records_csv(creator_type, tmp_path):
    """Missing metadata in a selected species is not silently ignored."""
    root = tmp_path / "input"
    (root / "mouse").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="records.csv"):
        creator_type(root, tmp_path / "output")


def test_header_only_species_records_remain_an_explicit_empty_class(
    creator_type, tmp_path
):
    """A valid class with no selected rows remains present with its CSV columns."""
    root = tmp_path / "input"
    species = root / "mouse"
    species.mkdir(parents=True)
    (species / "records.csv").write_text("image_path,dataset_split,photo_id\n")

    creator = creator_type(root, tmp_path / "output")

    assert creator.class_names == ["mouse"]
    assert creator.records_by_species["mouse"].empty
    assert creator.records_by_species["mouse"].columns.tolist() == [
        "image_path",
        "dataset_split",
        "photo_id",
    ]


def test_records_without_headers_are_rejected(creator_type, tmp_path):
    """An empty file is malformed metadata rather than an empty class."""
    root = tmp_path / "input"
    species = root / "mouse"
    species.mkdir(parents=True)
    (species / "records.csv").touch()

    with pytest.raises(ValueError, match="header"):
        creator_type(root, tmp_path / "output")


def test_constructor_does_not_load_speciesnet(creator_type, tmp_path, monkeypatch):
    """Input setup stays separate from baseline-model inference."""
    from smartrodent import yolo_dataset_creation

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("Input setup must not load model weights")

    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", unexpected_model_load)
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])

    creator = creator_type(root, tmp_path / "output")

    assert creator.class_names == ["mouse"]


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


@pytest.mark.parametrize("model_name", [None, "custom-model"])
def test_creators_initialize_and_reuse_a_lazy_detector(
    creator_type, creator_input, images, tmp_path, monkeypatch, model_name
):
    """Creator setup configures inherited inference without loading model weights."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    constructor = Mock(return_value=model)
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    monkeypatch.setattr(yolo_dataset_creation, "DEFAULT_MODEL", "test-default")
    output = tmp_path / "output"
    settings = {} if model_name is None else {"model_name": model_name}

    creator = creator_type(creator_input, output, **settings)

    constructor.assert_not_called()
    assert creator.speciesnet_model is None
    assert creator.infer_batch([]) == {}
    constructor.assert_not_called()
    for records in creator.records_by_species.values():
        source_path = records.iloc[0]["image_path"]
        result = creator.infer_batch([source_path])
        assert result[source_path]["detections"][0]["confidence"] == 0.9
        assert creator.speciesnet_model is model
    constructor.assert_called_once_with(
        "test-default" if model_name is None else model_name,
        components="detector",
    )
    assert not output.exists()


def test_creators_use_injected_detector_without_changing_source_records(
    creator_type, creator_input, images, tmp_path, monkeypatch
):
    """Inference reuses the supplied model without altering splits or duplicate rows."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images],
        max_batch_size=1,
    )
    constructor = Mock(side_effect=AssertionError("Injected model must be reused"))
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    output = tmp_path / "output"
    creator = creator_type(creator_input, output, model=model)
    original = {
        species: records.copy(deep=True)
        for species, records in creator.records_by_species.items()
    }

    assert creator.speciesnet_model is model
    for _ in range(2):
        result = creator.infer_batch(images, batch_size=1)
        assert set(result) == {str(path) for path in images}
        assert all(len(item["detections"]) == 1 for item in result.values())
    assert creator.speciesnet_model is model
    constructor.assert_not_called()
    for species, records in creator.records_by_species.items():
        pd.testing.assert_frame_equal(records, original[species])
    assert len(creator.records_by_species["mouse"]) == 2
    assert not output.exists()


def test_creators_reject_conflicting_model_configuration(
    creator_type, creator_input, tmp_path
):
    """An existing detector and a weights identifier cannot override each other."""
    with pytest.raises(ValueError, match="model"):
        creator_type(
            creator_input,
            tmp_path / "output",
            model=SpeciesNetStub([]),
            model_name="other-model",
        )


def test_detector_create_symlinks_sources_and_writes_yolo_labels(
    creator_input, images, tmp_path
):
    """Detector output preserves supplied splits and each oversampled occurrence."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    output = tmp_path / "detector"
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(
        creator_input, output, class_names=["shrew", "mouse"], model=model
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


def test_classifier_create_saves_crops_in_assigned_class_and_split(
    creator_input, images, tmp_path
):
    """Classifier outputs are actual crops, with duplicates retained in one split."""
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    output = tmp_path / "classifier"
    creator = YoloClassifierDatasetCreatorFromSpeciesnet(
        creator_input, output, model=model
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
    assert not list((output / "val").rglob("*.jpg"))
    data = yaml.safe_load((output / "data.yaml").read_text())
    assert data == {
        "path": str(output),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {0: "mouse", 1: "shrew"},
    }


def test_create_skips_images_without_accepted_boxes_and_records_why(
    creator_type, creator_input, images, tmp_path
):
    """Empty/filtered detections produce no examples but retain every metadata row."""
    model = SpeciesNetStub(
        [
            {"filepath": str(images[0]), "detections": []},
            {"filepath": str(images[1]), "detections": [detection(0.01)]},
        ]
    )
    output = tmp_path / "output"
    creator = creator_type(creator_input, output, model=model)

    assert creator.create() == output

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert len(metadata) == 3
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
    assert not list(output.rglob("*.jpg"))


def test_detector_filenames_do_not_collide_for_same_named_sources_in_one_split(
    creator_input, images, tmp_path
):
    """Distinct sources sharing a basename and split cannot overwrite each other."""
    records_path = images[1].parent / "records.csv"
    records = pd.read_csv(records_path)
    records["dataset_split"] = "train"
    records.to_csv(records_path, index=False)
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in images]
    )
    output = tmp_path / "output"
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(
        creator_input, output, model=model
    )

    creator.create()

    links = list((output / "images" / "train").iterdir())
    assert len(links) == 3
    assert len({link.name for link in links}) == 3
    assert sum(link.resolve() == images[0] for link in links) == 2
    assert sum(link.resolve() == images[1] for link in links) == 1
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 3


@pytest.mark.parametrize("destination", ["input", "inside_input", "parent"])
def test_create_rejects_destinations_overlapping_input(
    creator_type, creator_input, tmp_path, destination
):
    """A dataset output must not overwrite or contaminate the source input tree."""
    if destination == "input":
        output = creator_input
    elif destination == "inside_input":
        output = creator_input / "generated"
    else:
        output = creator_input.parent
    creator = creator_type(creator_input, output, model=SpeciesNetStub([], fail=True))

    with pytest.raises(ValueError, match="overlap"):
        creator.create()

    assert not (output / "metadata.json").exists()
    assert not (creator_input / "generated").exists()


def test_create_rejects_existing_output_without_overwriting(
    creator_type, creator_input, tmp_path
):
    """Recreating a dataset requires an explicit new destination, not silent overwrite."""
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep me")
    creator = creator_type(creator_input, output, model=SpeciesNetStub([], fail=True))

    with pytest.raises(FileExistsError):
        creator.create()

    assert marker.read_text() == "keep me"
    assert list(output.iterdir()) == [marker]


def test_empty_classes_create_layout_without_loading_model(creator_type, tmp_path):
    """An empty but valid input produces configuration and empty provenance."""
    root = tmp_path / "input"
    (root / "mouse").mkdir(parents=True)
    (root / "mouse" / "records.csv").write_text("image_path,dataset_split\n")
    output = tmp_path / "output"
    creator = creator_type(root, output, model=SpeciesNetStub([], fail=True))

    assert creator.create() == output

    assert json.loads((output / "metadata.json").read_text()) == {"records": []}
    assert yaml.safe_load((output / "data.yaml").read_text())["names"] == {0: "mouse"}
    assert creator.metadata_records == []


def test_create_persists_successful_detection_and_output_provenance(
    creator_type, creator_input, images, tmp_path
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
    creator = creator_type(creator_input, output, model=model)
    original = {
        species: records.copy(deep=True)
        for species, records in creator.records_by_species.items()
    }

    creator.create()

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert metadata == creator.metadata_records
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
    creator_type, creator_input, images, tmp_path
):
    """Serializing provenance must not further round the metadata loaded from CSV."""
    records_path = images[0].parent / "records.csv"
    frame = pd.read_csv(records_path)
    frame["measurement"] = [0.12345678901234567, 1.2345678901234567]
    frame.to_csv(records_path, index=False)
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": []} for path in images]
    )
    output = tmp_path / "output"
    creator = creator_type(creator_input, output, model=model)
    expected = creator.records_by_species["mouse"]["measurement"].tolist()

    creator.create()

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [row["measurement"] for row in metadata[:2]] == expected


def test_create_writes_every_accepted_detection(
    creator_type, creator_input, images, tmp_path
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
    creator = creator_type(creator_input, output, model=model)

    creator.create()

    if creator_type is YoloDetectorDatasetCreatorFromSpeciesnet:
        labels = list((output / "labels" / "train").glob("*.txt"))
        assert len(labels) == 2
        for label in labels:
            assert label.read_text().splitlines() == [
                "0 0.5 0.5 0.5 0.6",
                "0 0.05 0.05 0.1 0.1",
            ]
    elif creator_type is YoloClassifierDatasetCreatorFromSpeciesnet:
        paths = list((output / "train" / "mouse").iterdir())
        assert len(paths) == 4
        sizes = []
        for path in paths:
            with Image.open(path) as crop:
                sizes.append(crop.size)
        assert sizes.count((10, 6)) == 2
        assert sizes.count((2, 1)) == 2
    else:
        pytest.fail("Unexpected creator type")


@pytest.mark.parametrize("batch_size", [1, 7, 32, 40])
def test_create_handles_more_than_one_inference_batch(
    creator_type, tmp_path, batch_size
):
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
    model = SpeciesNetStub(
        [{"filepath": str(path), "detections": [detection(0.9)]} for path in paths],
        max_batch_size=batch_size,
    )
    model.detect = Mock(wraps=model.detect)
    output = tmp_path / "output"
    creator = creator_type(root, output, model=model, batch_size=batch_size)

    creator.create()

    assert [len(call.kwargs["filepaths"]) for call in model.detect.call_args_list] == [
        min(batch_size, 65 - start) for start in range(0, 65, batch_size)
    ]

    metadata = json.loads((output / "metadata.json").read_text())["records"]
    assert [row["source_row"] for row in metadata] == list(range(65))
    assert all(row["status"] == "written" for row in metadata)
    assert len({row["outputs"][0]["image_path"] for row in metadata}) == 65


@pytest.mark.parametrize(
    "creator_type",
    [
        YoloDetectorDatasetCreatorFromSpeciesnet,
        YoloClassifierDatasetCreatorFromSpeciesnet,
    ],
)
@given(indices=st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=8))
def test_create_filenames_are_deterministic_and_preserve_oversampling(
    creator_type, indices
):
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
            paths.append(path)
        pd.DataFrame(
            [
                {
                    "image_path": str(paths[index].resolve()),
                    "dataset_split": "train",
                    "photo_id": index,
                }
                for index in indices
            ]
        ).to_csv(species / "records.csv", index=False)
        model = SpeciesNetStub(
            [
                {"filepath": str(path.resolve()), "detections": [detection(0.9)]}
                for path in paths
            ]
        )
        output_names = []
        for output in [Path(directory) / "first", Path(directory) / "second"]:
            creator = creator_type(root, output, model=model)
            creator.create()
            metadata = json.loads((output / "metadata.json").read_text())["records"]
            assert [row["photo_id"] for row in metadata] == indices
            names = [row["outputs"][0]["image_path"] for row in metadata]
            assert len(names) == len(set(names)) == len(indices)
            assert all((output / name).is_file() for name in names)
            output_names.append(names)
        assert output_names[0] == output_names[1]


@pytest.fixture
def ultralytics_input(tmp_path):
    """Provide valid, nonempty splits for testing real Ultralytics dataset loading."""
    root = tmp_path / "input"
    predictions = []
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
            predictions.append(
                {"filepath": str(image_path), "detections": [detection(0.9)]}
            )
        pd.DataFrame(rows).to_csv(directory / "records.csv", index=False)
    return root, SpeciesNetStub(predictions)


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
    creator = YoloDetectorDatasetCreatorFromSpeciesnet(
        root, output, class_names=["shrew", "mouse"], model=model
    )
    creator.create()

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
    creator = YoloClassifierDatasetCreatorFromSpeciesnet(
        root, output, class_names=["shrew", "mouse"], model=model
    )
    creator.create()

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


def test_from_config_loads_selected_classes_and_lazy_model(
    creator_type, creator_input, tmp_path, monkeypatch
):
    """Both configurable creators retain class order and defer model loading."""
    constructor = Mock(
        side_effect=AssertionError("Config loading must not load weights")
    )
    monkeypatch.setattr(yolo_dataset_creation, "SpeciesNet", constructor)
    output = tmp_path / "stage8_yolo_dataset"
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    creator_type.__name__: {
                        "path_to_image_data": str(creator_input),
                        "dataset_output_path": str(output),
                        "class_names": ["shrew", "mouse"],
                        "model_name": "custom-speciesnet",
                        "batch_size": 7,
                        "confidence_threshold": 0.8,
                        "allowed_classes": ["human", "vehicle"],
                        "iou_threshold": 0.2,
                    }
                }
            }
        )
    )

    creator = creator_type.from_config(config_path)

    assert type(creator) is creator_type
    assert isinstance(creator, Configurable)
    assert creator.class_names == ["shrew", "mouse"]
    assert creator.classes == {0: "shrew", 1: "mouse"}
    assert list(creator.records_by_species) == ["shrew", "mouse"]
    assert creator.path_to_image_data == creator_input
    assert creator.dataset_output_path == output
    assert creator.speciesnet_model is None
    assert creator.speciesnet_model_name == "custom-speciesnet"
    assert creator.batch_size == 7
    assert creator.confidence_threshold == 0.8
    assert creator.allowed_classes == ("human", "vehicle")
    assert creator.iou_threshold == 0.2
    constructor.assert_not_called()
    assert not output.exists()


def test_from_config_uses_working_directory_paths_and_optional_defaults(
    creator_type, creator_input, tmp_path, monkeypatch
):
    """Relative paths follow the working directory, not the YAML file's directory."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "creator.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    creator_type.__name__: {
                        "path_to_image_data": "input",
                        "dataset_output_path": "stage8_yolo_dataset",
                    }
                }
            }
        )
    )

    creator = creator_type.from_config(config_path)

    assert isinstance(creator, Configurable)
    assert creator.path_to_image_data == creator_input
    assert creator.dataset_output_path == tmp_path / "stage8_yolo_dataset"
    assert creator.class_names == ["mouse", "shrew"]
    assert creator.speciesnet_model is None
    assert creator.speciesnet_model_name is None
    assert creator.batch_size == 32
    assert creator.confidence_threshold == 0.1
    assert creator.allowed_classes == ("animal",)
    assert creator.iou_threshold == 0.45
    assert not creator.dataset_output_path.exists()


@pytest.mark.parametrize(
    "configuration",
    [
        "null",
        "[]",
        "invalid",
        "data: null",
        "data: []",
        "data: {}",
        "data:\n  CREATOR: null",
        "data:\n  CREATOR: []",
    ],
)
def test_from_config_requires_mapping_sections(creator_type, tmp_path, configuration):
    """Malformed YAML structure has explicit errors, not guessed defaults."""
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(configuration.replace("CREATOR", creator_type.__name__))

    with pytest.raises(TypeError, match="mapping"):
        creator_type.from_config(config_path)


@pytest.mark.parametrize("missing", ["path_to_image_data", "dataset_output_path"])
def test_from_config_requires_source_and_destination(
    creator_type, creator_input, tmp_path, missing
):
    """Missing required paths are reported by their configuration key."""
    settings = {
        "path_to_image_data": str(creator_input),
        "dataset_output_path": str(tmp_path / "stage8_yolo_dataset"),
    }
    del settings[missing]
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(yaml.safe_dump({"data": {creator_type.__name__: settings}}))

    with pytest.raises(ValueError, match=missing):
        creator_type.from_config(config_path)


@pytest.mark.parametrize("extra_setting", ["model", "train_val_test_split", "typo"])
def test_from_config_rejects_unknown_or_python_only_settings(
    creator_type, creator_input, tmp_path, extra_setting
):
    """Model injection is Python-only, and obsolete/unknown options cannot be ignored."""
    settings = {
        "path_to_image_data": str(creator_input),
        "dataset_output_path": str(tmp_path / "stage8_yolo_dataset"),
        extra_setting: None,
    }
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(yaml.safe_dump({"data": {creator_type.__name__: settings}}))

    with pytest.raises(ValueError, match=extra_setting):
        creator_type.from_config(config_path)


@pytest.mark.parametrize(
    "setting, value, error",
    [
        ("path_to_image_data", None, TypeError),
        ("path_to_image_data", True, TypeError),
        ("dataset_output_path", 123, TypeError),
        ("dataset_output_path", "", ValueError),
        ("path_to_image_data", "   ", ValueError),
        ("class_names", "mouse", TypeError),
        ("class_names", {"mouse": True}, TypeError),
        ("class_names", [123], TypeError),
        ("model_name", 123, TypeError),
        ("model_name", "", ValueError),
        ("batch_size", 0, ValueError),
        ("batch_size", -1, ValueError),
        ("batch_size", True, ValueError),
        ("batch_size", 1.5, ValueError),
        ("batch_size", "32", ValueError),
        ("batch_size", None, ValueError),
        ("confidence_threshold", -0.1, ValueError),
        ("confidence_threshold", 1.1, ValueError),
        ("confidence_threshold", True, TypeError),
        ("confidence_threshold", "0.5", TypeError),
        ("confidence_threshold", None, TypeError),
        ("iou_threshold", -0.1, ValueError),
        ("iou_threshold", 1.1, ValueError),
        ("iou_threshold", True, TypeError),
        ("iou_threshold", "0.5", TypeError),
        ("iou_threshold", None, TypeError),
        ("allowed_classes", "animal", TypeError),
        ("allowed_classes", None, TypeError),
        ("allowed_classes", {"animal": True}, TypeError),
        ("allowed_classes", [123], TypeError),
        ("allowed_classes", ["  "], ValueError),
    ],
)
def test_from_config_rejects_invalid_setting_types(
    creator_type, creator_input, tmp_path, setting, value, error
):
    """Configuration values cannot silently reinterpret paths, class lists, or models."""
    settings = {
        "path_to_image_data": str(creator_input),
        "dataset_output_path": str(tmp_path / "stage8_yolo_dataset"),
        setting: value,
    }
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(yaml.safe_dump({"data": {creator_type.__name__: settings}}))

    with pytest.raises(error, match=setting):
        creator_type.from_config(config_path)


def test_from_config_reports_missing_config_file(creator_type, tmp_path):
    """A missing configuration file does not produce a partially configured creator."""
    with pytest.raises(FileNotFoundError):
        creator_type.from_config(tmp_path / "missing.yaml")


def test_from_config_requires_its_own_class_named_section(creator_type, tmp_path):
    """Creators cannot accidentally consume another creator's configuration."""
    other_type = (
        YoloClassifierDatasetCreatorFromSpeciesnet
        if creator_type is YoloDetectorDatasetCreatorFromSpeciesnet
        else YoloDetectorDatasetCreatorFromSpeciesnet
    )
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(yaml.safe_dump({"data": {other_type.__name__: {}}}))

    with pytest.raises(TypeError, match=creator_type.__name__):
        creator_type.from_config(config_path)


@pytest.mark.parametrize(
    "confidence, labels, iou, indices",
    [
        (0.8, ["human"], 0.0, [0]),
        (0.7, ["human"], 1.0, [0, 1]),
        (0.8, ["animal"], 0.0, [2]),
        (0.95, ["human"], 0.0, []),
        (0.0, [], 1.0, []),
    ],
)
def test_configured_filtering_controls_created_dataset(
    creator_type, creator_input, images, tmp_path, confidence, labels, iou, indices
):
    """YAML filtering settings control retained boxes and actual dataset outputs."""
    output = tmp_path / "configured_output"
    config_path = tmp_path / "creator.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    creator_type.__name__: {
                        "path_to_image_data": str(creator_input),
                        "dataset_output_path": str(output),
                        "confidence_threshold": confidence,
                        "allowed_classes": labels,
                        "iou_threshold": iou,
                    }
                }
            }
        )
    )
    creator = creator_type.from_config(config_path)
    creator.initialize_speciesnet(
        model=SpeciesNetStub(
            [
                {
                    "filepath": str(path),
                    "detections": [
                        detection(0.9, label="human"),
                        detection(0.7, label="human"),
                        detection(0.95, label="animal"),
                    ],
                }
                for path in images
            ]
        )
    )

    creator.create()

    records = json.loads((output / "metadata.json").read_text())["records"]
    assert records
    for record in records:
        assert [item["detection_index"] for item in record["detections"]] == indices
        assert record["status"] == ("written" if indices else "skipped")
        for item in record["outputs"]:
            assert (output / item["image_path"]).is_file()


@given(
    confidence=st.floats(min_value=0, max_value=1),
    iou=st.floats(min_value=0, max_value=1),
)
def test_constructor_preserves_valid_filter_thresholds(confidence, iou):
    """Every finite threshold in the supported interval remains configurable."""
    with TemporaryDirectory() as directory:
        root = Path(directory) / "input"
        write_species(root, "mouse", ["train"])
        labels = ["animal"]
        creator = YoloDetectorDatasetCreatorFromSpeciesnet(
            root,
            Path(directory) / "output",
            confidence_threshold=confidence,
            allowed_classes=labels,
            iou_threshold=iou,
        )
        labels.append("human")
        assert creator.confidence_threshold == confidence
        assert creator.iou_threshold == iou
        assert creator.allowed_classes == ("animal",)
