import json
import sys
from collections import OrderedDict
from pathlib import Path

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np
import pytest
import torch

from smartrodent import resolve_data_path
import smartrodent.utils as utils


def test_resolve_data_path_returns_absolute_path_unchanged(tmp_path):
    absolute = tmp_path / "images"

    assert resolve_data_path(absolute) == absolute


def test_resolve_data_path_resolves_relative_path_from_repo_root():
    expected_root = Path(utils.__file__).resolve().parents[2]

    assert (
        resolve_data_path("configs/data_config_full.yaml")
        == (expected_root / "configs/data_config_full.yaml").resolve()
    )


def test_resolve_data_path_accepts_path_objects():
    expected_root = Path(utils.__file__).resolve().parents[2]

    assert (
        resolve_data_path(Path("README.md")) == (expected_root / "README.md").resolve()
    )


def test_resolve_data_path_honors_parent_override(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "PARENT", tmp_path, raising=False)

    assert resolve_data_path("relative/data") == (tmp_path / "relative/data").resolve()


def test_import_class_returns_the_named_attribute():
    assert utils.import_class("collections.OrderedDict") is OrderedDict


@pytest.mark.parametrize(
    ("class_path", "message"),
    [
        ("OrderedDict", "full module.Class"),
        ("not_a_module.Thing", "Cannot import"),
        ("collections.NoSuchThing", "no attribute"),
    ],
)
def test_import_class_rejects_unresolvable_paths(class_path, message):
    with pytest.raises(ValueError, match=message):
        utils.import_class(class_path)


tensor_arrays = hnp.arrays(
    dtype=st.sampled_from([np.float32, np.float64, np.int64, np.bool_]),
    shape=hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=4),
    elements={"allow_nan": False, "allow_infinity": False},
)


@given(array=tensor_arrays)
def test_to_json_value_survives_json_round_trip(array):
    tensor = torch.from_numpy(array)

    value = utils.to_json_value(tensor)

    assert json.loads(json.dumps(value)) == value
    # Nested lists cannot encode the shape of empty tensors, e.g. (0, 0) -> [],
    # so the shape is only checked from the nesting when there are elements.
    restored = np.asarray(value, dtype=array.dtype)
    if array.size > 0:
        assert restored.shape == array.shape
    else:
        assert restored.size == 0
    assert np.array_equal(restored.reshape(array.shape), array)


@given(array=tensor_arrays)
def test_to_json_value_returns_scalars_only_for_zero_dim_tensors(array):
    value = utils.to_json_value(torch.from_numpy(array))

    assert isinstance(value, list) == (array.ndim > 0)
