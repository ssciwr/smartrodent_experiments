"""Verify a uniform required positional configuration interface for scripts."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "parser_name"),
    [
        ("scripts.download_gbif_data", "_parse_args"),
        ("scripts.filter_data_vlm", "parse_args"),
        ("scripts.materialize_filtered_data", "parse_args"),
        ("scripts.migrate_records", "parse_args"),
        ("scripts.sample_training_dataset", "_parse_args"),
        ("scripts.split_dataset", "_parse_args"),
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [["settings.yaml"], [], ["-c", "settings.yaml"], ["--config", "settings.yaml"]],
)
def test_scripts_require_positional_config(
    module_name, parser_name, arguments, monkeypatch
):
    parser = getattr(importlib.import_module(module_name), parser_name)
    monkeypatch.setattr("sys.argv", [module_name, *arguments])
    if arguments == ["settings.yaml"]:
        assert parser().config == Path("settings.yaml")
    else:
        with pytest.raises(SystemExit) as error:
            parser()
        assert error.value.code == 2
