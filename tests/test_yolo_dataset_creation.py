"""Constructor contract for YOLO creators consuming already-split records."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, strategies as st
import pandas as pd
import pytest

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


def test_create_explicitly_reports_pending_integration(creator_type, tmp_path):
    """The setup-only creator cannot silently run the obsolete sampling workflow."""
    root = tmp_path / "input"
    write_species(root, "mouse", ["train"])
    output = tmp_path / "output"
    creator = creator_type(root, output)

    with pytest.raises(NotImplementedError, match="integration pending"):
        creator.create()

    assert not output.exists()
