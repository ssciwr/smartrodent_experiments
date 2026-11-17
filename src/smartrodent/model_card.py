"""Read model-card YAML, render Markdown, and save both formats."""

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
            lines.extend([f"{'#' * level} {title}", "", render_model_card(value, level + 1)])
        elif isinstance(value, list):
            lines.extend([f"{'#' * level} {title}", "", *(f"- {item}" for item in value), ""])
        else:
            text = "Not available" if value is None else str(value)
            lines.extend([f"**{title}:** {text}", ""])
    return "\n".join(lines)


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
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    markdown_path.write_text(render_model_card(document), encoding="utf-8")
    return yaml_path, markdown_path
