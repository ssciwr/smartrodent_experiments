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
                "image_path": str(image),
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
        str(root / "mouse" / f"photo_{index}.jpg") for index in range(3)
    ]
    assert not output.exists()


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


def test_relative_image_paths_resolve_against_species_records_directory(
    creator_type, tmp_path
):
    """Relative paths are interpreted relative to their records.csv, not the cwd."""
    root = tmp_path / "input"
    directory = write_species(root, "mouse", ["train"])
    records = pd.read_csv(directory / "records.csv")
    records["image_path"] = ["photo_0.jpg"]
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
