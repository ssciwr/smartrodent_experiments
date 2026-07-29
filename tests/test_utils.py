import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
