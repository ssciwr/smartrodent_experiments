"""Behavioral tests for YAML configuration loading with the study tags."""

import pytest
import yaml

from smartrodent import config_loading


def test_tagged_files_load_like_tagged_text(tmp_path):
    """Both entry points read the same tags into the same document."""
    text = """
lr: !suggest-float {low: 0.1, high: 0.9}
head: !suggest-categorical {choices: [linear, full]}
freeze: !coupled {target: head, values: [[0], [0, 1]]}
"""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    assert config_loading.load_yaml(path) == config_loading.parse_yaml(text)


@pytest.mark.parametrize(
    "text",
    [
        "seed: !sweep {values: [1, 2]}\n",
        "batch: !coupled-sweep {target: seed, values: [8, 4]}\n",
        "other: !reference {target: seed}\n",
        "kind: !pyobject pathlib.Path\n",
    ],
)
def test_tags_other_than_the_study_tags_are_rejected(text):
    """Only the study tags are configuration syntax; any other tag fails to load."""
    with pytest.raises(yaml.constructor.ConstructorError):
        config_loading.parse_yaml(text)


def test_study_tags_do_not_leak_into_plain_yaml_loading():
    """Loading a tagged configuration leaves PyYAML's shared safe loader unchanged."""
    text = "lr: !suggest-float {low: 0.1, high: 0.9}\n"
    config_loading.parse_yaml(text)
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(text)


@pytest.mark.parametrize("content", ["[]\n", "data: []\n", "data:\n  lila: 3\n"])
def test_load_section_names_the_missing_mapping(tmp_path, content):
    """A missing or non-mapping section is reported by its dotted location."""
    path = tmp_path / "config.yaml"
    path.write_text(content)

    with pytest.raises(ValueError, match="data.lila"):
        config_loading.load_section(path, "data", "lila")
