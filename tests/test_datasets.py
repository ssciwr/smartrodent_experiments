"""Behavioral tests for the labeled image datasets and multiview batching."""

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from smartrodent.datasets import ImageDataset, MultiViewImageDataset, collate_views

TO_TENSOR = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])


class NumberedView:
    """A picklable transform whose n-th call returns a tensor filled with n."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return torch.full((1, 2, 2), float(self.calls))


def write_classes(root: Path, counts: dict[str, int], mode: str = "RGB") -> None:
    """Write ``counts[name]`` small images into one directory per class."""
    for name, count in counts.items():
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new(mode, (8, 8)).save(directory / f"{index}.png")
        (directory / "notes.txt").write_text("not an image")


def test_configured_names_select_directories_with_alphabetical_labels(tmp_path):
    """Only configured classes are indexed; their IDs follow alphabetical order."""
    write_classes(tmp_path, {"rat": 2, "mouse": 1, "shrew": 3})
    # Only a class directory's own files count, not images in nested folders.
    (tmp_path / "rat" / "nested").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "rat" / "nested" / "2.png")

    dataset = ImageDataset(tmp_path, ("rat", "mouse"), TO_TENSOR)

    assert dataset.label_names == {"mouse": 0, "rat": 1}
    assert dataset.class_counts == {"mouse": 1, "rat": 2}
    assert dataset.labels == [0, 1, 1]
    assert dataset.images == [
        tmp_path / "mouse" / "0.png",
        tmp_path / "rat" / "0.png",
        tmp_path / "rat" / "1.png",
    ]


def test_existing_mapping_is_kept(tmp_path):
    """Validation reuses the training mapping instead of deriving its own."""
    write_classes(tmp_path, {"mouse": 1, "rat": 1})

    dataset = ImageDataset(tmp_path, {"rat": 0, "mouse": 1}, TO_TENSOR)

    assert dataset.label_names == {"rat": 0, "mouse": 1}
    assert {
        dataset.images[index].parent.name: dataset[index][0]
        for index in range(len(dataset))
    } == {"rat": 0, "mouse": 1}


def test_items_are_label_then_rgb_image(tmp_path):
    """Items keep the (label, image) order; any image mode loads as RGB."""
    write_classes(tmp_path, {"mouse": 1}, mode="L")

    label, image = ImageDataset(tmp_path, ["mouse"], TO_TENSOR)[0]
    _, untransformed = ImageDataset(tmp_path, ["mouse"])[0]

    assert label == 0
    assert image.shape == (3, 8, 8)
    assert untransformed.mode == "RGB"


def test_missing_class_directory_is_reported(tmp_path):
    write_classes(tmp_path, {"mouse": 1})

    with pytest.raises(FileNotFoundError, match="rat"):
        ImageDataset(tmp_path, ["mouse", "rat"], TO_TENSOR)


def test_empty_class_is_indexed_with_zero_count(tmp_path):
    """Callers decide whether an empty class is acceptable (SupCon allows it)."""
    write_classes(tmp_path, {"mouse": 1, "rat": 0})

    dataset = ImageDataset(tmp_path, ["mouse", "rat"], TO_TENSOR)

    assert dataset.label_names == {"mouse": 0, "rat": 1}
    assert dataset.class_counts == {"mouse": 1, "rat": 0}
    assert len(dataset) == 1


@settings(max_examples=25, deadline=None)
@given(
    counts=st.dictionaries(
        st.text("abcdefgh", min_size=1, max_size=6), st.integers(0, 3), min_size=1
    )
)
def test_labels_match_each_image_directory(counts):
    """Every image's label is the alphabetical ID of the directory it is in."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        write_classes(root, counts)

        dataset = ImageDataset(root, list(counts), TO_TENSOR)

        assert dataset.label_names == {
            name: index for index, name in enumerate(sorted(counts))
        }
        assert dataset.class_counts == counts
        assert [dataset.label_names[image.parent.name] for image in dataset.images] == (
            dataset.labels
        )


def test_multiview_items_have_two_independent_views_and_collate_by_view(tmp_path):
    """Each item holds two transform calls; a batch lists all first views first."""
    write_classes(tmp_path, {"mouse": 1, "rat": 1})
    dataset = MultiViewImageDataset(tmp_path, ["mouse", "rat"], NumberedView())
    # DataLoader workers receive the dataset pickled.
    restored = pickle.loads(pickle.dumps(dataset))

    first, second = restored[0], restored[1]
    labels, images = collate_views([first, second])

    assert first[0] == 0 and second[0] == 1
    assert len(first) == len(second) == 3
    assert labels.tolist() == [0, 1, 0, 1]
    # Views 1 and 2 come from the first item, 3 and 4 from the second.
    assert images[:, 0, 0, 0].tolist() == [1.0, 3.0, 2.0, 4.0]
