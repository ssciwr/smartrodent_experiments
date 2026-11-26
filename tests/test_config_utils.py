"""Behavioral tests for configuration loading and sweep expansion."""

from pathlib import Path

import pytest
import yaml
from study_test_support import load_tagged_yaml

from smartrodent import config_utils


def test_custom_yaml_tags_and_config_expansion():
    config = yaml.load(
        """
model: !sweep
  values: [small, large]
batch: !coupled-sweep
  target: model
  values: [8, 4]
optimizer:
  momentum: !sweep
    values: [0.8, 0.9]
reference: !reference
  target: nested.values[1]
object: !pyobject pathlib.Path
""",
        Loader=config_utils.get_loader(),
    )

    runs = config_utils.ConfigHandler(config).run_configs

    assert len(runs) == 4
    assert {(run["model"], run["batch"]) for run in runs} == {
        ("small", 8),
        ("large", 4),
    }
    assert {run["optimizer"]["momentum"] for run in runs} == {0.8, 0.9}
    assert config["reference"]["target"] == ["nested", "values", 1]
    assert config["object"] is Path


def test_configuration_helpers_handle_edge_cases():
    unchanged = {"plain": 1}
    assert config_utils.ConfigHandler(unchanged).run_configs == [unchanged]
    converted = config_utils.convert_to_pyobject_tags(
        {"type": Path, "nested": [str, "plain"]}
    )
    assert converted["type"].endswith(".Path")
    assert converted["nested"] == ["!pyobject builtins.str", "plain"]


def test_coupled_sweep_requires_matching_lengths():
    config = {
        "model": {"type": "sweep", "values": ["small", "large"]},
        "batch": {
            "type": "coupled-sweep",
            "target": ["model"],
            "values": [8],
        },
    }

    with pytest.raises(ValueError, match="Incompatible lengths"):
        config_utils.ConfigHandler(config)


@pytest.mark.parametrize(
    "text",
    [
        "lr: !suggest-float {low: 0.1, high: 0.9}\n",
        "optimizer:\n  lr: !suggest-float {low: 0.1, high: 0.9}\n",
        "scale: [!suggest-float {low: 0.5, high: 0.9}, 1.0]\n",
        "seed: !sweep {values: [1, 2]}\nhead: !suggest-categorical {choices: [a, b]}\n",
    ],
)
def test_grid_sweep_expansion_rejects_suggest_tags_without_a_study(text):
    """Decision: tags outside a study fail clearly instead of leaking into a run config."""
    with pytest.raises(ValueError, match="tuning"):
        config_utils.ConfigHandler(load_tagged_yaml(text))


@pytest.mark.parametrize("content", ["[]\n", "data: []\n", "data:\n  lila: 3\n"])
def test_load_section_names_the_missing_mapping(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)

    with pytest.raises(ValueError, match="data.lila"):
        config_utils.load_section(path, "data", "lila")
