"""Read model-card YAML, render Markdown, and save both formats."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml


def load_card_template(path: Path) -> dict:
    """Read a YAML model-card template.

    Args:
        path: Template file.

    Returns:
        Template content as a dictionary.
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Model card template must be a YAML mapping")
    return document


def load_training_template(path: Path) -> dict:
    """Load and verify the sections populated by training publication.

    Args:
        path: Model-card template filename.

    Returns:
        Template with mutable model, validation, and provenance sections.

    Raises:
        ValueError: A required training section is missing or is not a mapping.
    """
    template = load_card_template(path)
    for section in ("model", "validation", "provenance"):
        if not isinstance(template.get(section), dict):
            raise ValueError(f"model_card_template.{section} must be a mapping")
    return template


def render_model_card(document: dict, level: int = 1) -> str:
    """Render YAML fields as Markdown headings, lists, and text.

    Args:
        document: Populated model-card content.
        level: Heading level for nested sections.

    Returns:
        Markdown text preserving the YAML field order.
    """
    lines = []
    for key, value in document.items():
        title = str(key).replace("_", " ").title()
        if isinstance(value, dict):
            lines.extend(
                [f"{'#' * level} {title}", "", render_model_card(value, level + 1)]
            )
        elif isinstance(value, list):
            lines.extend(
                [f"{'#' * level} {title}", "", *(f"- {item}" for item in value), ""]
            )
        else:
            text = "Not available" if value is None else str(value)
            lines.extend([f"**{title}:** {text}", ""])
    return "\n".join(lines)


def write_training_card(
    template: dict,
    checkpoint: Path,
    snapshot: Mapping,
    *,
    model_details: Mapping,
    validation_dataset: Path | None,
    provenance: Mapping,
) -> tuple[Path, Path]:
    """Populate and write a verified template from a selected training snapshot.

    Args:
        template: Prepared training template, retaining authored narrative fields.
        checkpoint: Selected checkpoint described by the card.
        snapshot: Its loaded class mapping, epoch, and history.
        model_details: Stage-specific model name, version, source, and task.
        validation_dataset: Evaluated split, or None for training-only runs.
        provenance: Captured source, checkpoint, run, and logged-model identity.

    Returns:
        Written YAML and Markdown card paths.
    """
    card = deepcopy(template)
    card["model"].update(model_details)
    # Preserve dataset IDs; sorting changes only their display order.
    card["classes"] = dict(
        sorted((index, name) for name, index in snapshot["label_names"].items())
    )
    metrics = snapshot["history"][-1].get("validation", {})
    card["validation"].update(
        status="evaluated" if metrics else "not_validated",
        dataset=str(validation_dataset) if validation_dataset is not None else None,
        epoch=snapshot["epoch"],
        metrics=metrics,
    )
    card["provenance"].update(
        checkpoint=checkpoint.name,
        epoch=snapshot["epoch"],
        run_id=None,
        model_id=None,
        git_commit=None,
        dataset_revision=None,
    )
    card["provenance"].update(provenance)
    return write_model_card(card, checkpoint)


def write_model_card(document: dict, checkpoint: Path) -> tuple[Path, Path]:
    """Save populated YAML and Markdown beside a checkpoint.

    Args:
        document: Populated model-card content.
        checkpoint: Model snapshot whose filename supplies the card filenames.

    Returns:
        Paths to the YAML and Markdown files.
    """
    yaml_path = checkpoint.with_suffix(".model_card.yaml")
    markdown_path = checkpoint.with_suffix(".model_card.md")
    yaml_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    markdown_path.write_text(render_model_card(document), encoding="utf-8")
    return yaml_path, markdown_path
