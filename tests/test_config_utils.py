"""Behavioral tests for configuration loading and sweep expansion."""

from pathlib import Path

import pytest
import yaml

import smartrodent.config_utils as config_utils


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


@pytest.mark.parametrize("tag", ["!range", "!random_uniform"])
def test_removed_pre_optuna_tags_are_rejected_as_unknown(tag):
    """The retired grid/random tags must not silently load as unexpanded mappings."""
    with pytest.raises(yaml.YAMLError, match=tag):
        yaml.load(
            f"lr: {tag}\n  start: 0.1\n  stop: 0.9\n",
            Loader=config_utils.get_loader(),
        )


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
