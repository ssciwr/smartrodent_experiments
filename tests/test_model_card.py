"""Focused checks for card rendering and the trainer's MLflow integration."""

from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from smartrodent.model_card import (
    load_card_template,
    render_model_card,
    write_model_card,
)


def test_yaml_and_markdown_files(tmp_path):
    """Both formats preserve populated values and authored Markdown."""
    card = load_card_template(Path("configs/model_card_template.yaml"))
    card["model"].update(name="rodents", registry_version="3")
    card["classes"] = {0: "mouse", 1: "rat"}
    card["validation"]["metrics"] = {"accuracy": 0.8}
    card["intended_use"] = "Use **only** on camera traps."
    card["notes"] = ["First note", "Second note"]
    yaml_path, markdown_path = write_model_card(card, tmp_path / "best.pt")
    assert yaml.safe_load(yaml_path.read_text()) == card
    markdown = markdown_path.read_text()
    assert markdown == render_model_card(card)
    assert "**Registry Version:** 3" in markdown
    assert "**0:** mouse" in markdown
    assert "**Accuracy:** 0.8" in markdown
    assert "Use **only** on camera traps." in markdown
    assert "- First note" in markdown
    assert "Not available" in markdown


def test_template_requires_mapping(tmp_path):
    """Invalid YAML structure fails when the template is loaded."""
    path = tmp_path / "template.yaml"
    path.write_text("- not a mapping")
    with pytest.raises(ValueError, match="mapping"):
        load_card_template(path)


@given(
    st.dictionaries(
        st.text(min_size=1), st.floats(allow_nan=False, allow_infinity=False)
    )
)
def test_render_survives_yaml_round_trip(metrics):
    """Metric names and numeric values render consistently after YAML storage."""
    card = {"validation": {"metrics": metrics}}
    assert render_model_card(
        yaml.safe_load(yaml.safe_dump(card, sort_keys=False))
    ) == render_model_card(card)
