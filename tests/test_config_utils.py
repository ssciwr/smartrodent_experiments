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
  lr: !range
    start: 1
    stop: 2
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
    assert config["optimizer"]["lr"]["values"].tolist() == [1, 2]
    assert config["reference"]["target"] == ["nested", "values", 1]
    assert config["object"] is Path


def test_range_and_random_uniform_yaml_tags(monkeypatch):
    monkeypatch.setattr(
        config_utils.np.random,
        "uniform",
        lambda start, stop, size: config_utils.np.full(size, (start + stop) / 2),
    )
    config = yaml.load(
        """
explicit: !range
  start: 0.0
  stop: 0.4
  step: 0.2
linear: !random_uniform
  start: 0.1
  stop: 0.9
  log: false
  size: 2
logarithmic: !random_uniform
  start: 0.01
  stop: 1.0
  log: true
""",
        Loader=config_utils.get_loader(),
    )

    assert config["explicit"]["values"].tolist() == pytest.approx([0.0, 0.2, 0.4])
    assert len(config["linear"]["values"]) == 2
    assert len(config["logarithmic"]["values"]) == 5


def test_configuration_helpers_handle_edge_cases():
    with pytest.raises(ValueError, match="step must not be zero"):
        config_utils.range_inclusive(0, 1, 0)

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
