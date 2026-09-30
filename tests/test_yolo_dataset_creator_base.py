"""Behavioral tests for the shared YOLO dataset creator base."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, strategies as st
import pytest

from smartrodent.base import YoloDatasetCreatorBase


class ConcreteDatasetCreator(YoloDatasetCreatorBase):
    """Minimal concrete creator used to exercise shared validation."""

    def __call__(self):
        """Return the configured output root."""
        return self.dataset_output_path


def test_base_validates_inputs_and_creates_only_dataset_root(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    images.mkdir()
    labels.mkdir()

    creator = ConcreteDatasetCreator(
        path_to_image_data=images,
        path_to_labels=labels,
        dataset_output_path=output,
        class_names=["mouse"],
    )

    assert creator.dataset_output_path == output
    assert output.is_dir()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("missing_argument", ["path_to_image_data", "path_to_labels"])
def test_base_rejects_missing_input_paths(tmp_path, missing_argument):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    arguments = {
        "path_to_image_data": images,
        "path_to_labels": labels,
        "dataset_output_path": tmp_path / "output",
        "class_names": ["mouse"],
    }
    arguments[missing_argument] = tmp_path / "missing"

    with pytest.raises(ValueError, match="does not exist"):
        ConcreteDatasetCreator(**arguments)


@given(
    train=st.integers(min_value=0, max_value=100),
    validation=st.integers(min_value=0, max_value=100),
    test=st.integers(min_value=0, max_value=100),
)
def test_base_rejects_split_fractions_that_do_not_sum_to_one(
    train, validation, test
):
    if train + validation + test == 100:
        return
    with TemporaryDirectory() as directory:
        root = Path(directory)
        images = root / "images"
        labels = root / "labels"
        images.mkdir()
        labels.mkdir()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConcreteDatasetCreator(
                path_to_image_data=images,
                path_to_labels=labels,
                dataset_output_path=root / "output",
                class_names=["mouse"],
                train_val_test_split=(train / 100, validation / 100, test / 100),
            )
