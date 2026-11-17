"""Focused checks for card rendering and the trainer's MLflow integration."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from hypothesis import given, strategies as st
from mlflow import MlflowClient
import pytest
import torch
import yaml

from scripts import train_supcon
from smartrodent.model_card import load_card_template, render_model_card, write_model_card


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


@given(st.dictionaries(st.text(min_size=1), st.floats(allow_nan=False, allow_infinity=False)))
def test_render_survives_yaml_round_trip(metrics):
    """Metric names and numeric values render consistently after YAML storage."""
    card = {"validation": {"metrics": metrics}}
    assert render_model_card(yaml.safe_load(yaml.safe_dump(card, sort_keys=False))) == render_model_card(card)


@pytest.mark.parametrize("upload_fails", [False, True])
def test_trainer_publishes_selected_checkpoint(tmp_path, monkeypatch, upload_fails):
    """Real MLflow stores the card and description before champion promotion."""
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = client.create_experiment("cards", artifact_location=(tmp_path / "artifacts").as_uri())
    run = client.create_run(experiment_id)
    logged = client.create_logged_model(experiment_id, source_run_id=run.info.run_id)
    client.create_registered_model("rodents")
    version = client.create_model_version("rodents", logged.artifact_location, run_id=run.info.run_id)
    model_info = SimpleNamespace(model_id=logged.model_id, run_id=run.info.run_id, registered_model_version=version.version)
    monkeypatch.setattr(train_supcon.mlflow.pytorch, "log_model", lambda **kwargs: model_info)
    config = SimpleNamespace(
        model_name="base.pt", output_directory=tmp_path,
        model_card_template="configs/model_card_template.yaml",
        validation_metrics=None, validation_dataset="images/val",
        mlflow=SimpleNamespace(registered_model_name="rodents", model_artifact_name="classifier", champion_alias="champion", dataset_revision="data-v1"),
    )
    checkpoint = tmp_path / "epoch_0002.pt"
    torch.save({
        "model_state_dict": {}, "epoch": 2, "label_names": {"rat": 1, "mouse": 0},
        "history": [{"epoch": 1, "validation": {"accuracy": 0.1}},
                    {"epoch": 2, "validation": {"accuracy": 0.8}}],
    }, checkpoint)
    trainer = train_supcon.SupConExperiment(config)
    if upload_fails:
        monkeypatch.setattr(client, "log_model_artifact", Mock(side_effect=RuntimeError("upload failed")))
        with pytest.raises(RuntimeError, match="upload failed"):
            trainer._register_best_model(Mock(), checkpoint, client, {}, 2)
        assert client.get_registered_model("rodents").aliases == {}
        assert checkpoint.with_suffix(".model_card.md").exists()
    else:
        trainer._register_best_model(Mock(), checkpoint, client, {"git_commit": "training-commit"}, 2)
        yaml_path = checkpoint.with_suffix(".model_card.yaml")
        markdown_path = checkpoint.with_suffix(".model_card.md")
        card = yaml.safe_load(yaml_path.read_text())
        assert card["model"]["registry_version"] == str(version.version)
        assert card["classes"] == {0: "mouse", 1: "rat"}
        assert card["validation"]["metrics"] == {"accuracy": 0.8}
        assert card["provenance"] == {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epoch": 2, "run_id": run.info.run_id, "model_id": logged.model_id,
            "git_commit": "training-commit", "dataset_revision": "data-v1",
        }
        assert "# Provenance" in markdown_path.read_text()
        assert "training-commit" in markdown_path.read_text()
        assert client.get_model_version("rodents", str(version.version)).description == markdown_path.read_text()
        assert {item.path for item in client.list_logged_model_artifacts(logged.model_id)} == {yaml_path.name, markdown_path.name}
        assert client.get_model_version_by_alias("rodents", "champion").version == version.version
