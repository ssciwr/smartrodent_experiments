"""Load YAML configuration files, including the Optuna study tags."""

from pathlib import Path
from typing import Any, ClassVar

import yaml


def _record_tag(loader: yaml.SafeLoader, node: yaml.Node) -> dict[str, Any]:
    """Record a study tag's settings as ``{type: <tag>, <settings>...}``.

    In a YAML file the tags look like this::

        lr: !suggest-float {low: 1.0e-5, high: 1.0e-3, log: true}
        batch: !suggest-int {low: 4, high: 12, step: 2}
        head: !suggest-categorical {choices: [linear, [0, 1], {kind: full}]}
        freeze: !coupled {target: head, values: [11, 10, 9]}

    Only the settings are recorded here; ``search_space.SearchSpace`` validates them.

    Raises:
        ValueError: The tag is not applied to a mapping or uses the reserved key 'type'.
    """
    if not isinstance(node, yaml.MappingNode):
        raise ValueError(f"{node.tag} requires a mapping of settings")
    settings = loader.construct_mapping(node, deep=True)
    if "type" in settings:
        raise ValueError(f"{node.tag} does not accept the reserved key 'type'")
    return {"type": node.tag.removeprefix("!"), **settings}


class _TaggedLoader(yaml.SafeLoader):
    """A safe loader of its own, so the tags never leak into ``yaml.safe_load``."""

    # Declared on the subclass: SafeLoader.add_constructor would register the tags on
    # PyYAML's shared SafeLoader instead.
    yaml_constructors: ClassVar[dict] = {
        **yaml.SafeLoader.yaml_constructors,
        "!suggest-float": _record_tag,
        "!suggest-int": _record_tag,
        "!suggest-categorical": _record_tag,
        "!coupled": _record_tag,
    }


def parse_yaml(text: str) -> Any:
    """Parse YAML text with the study tags.

    Args:
        text: YAML document.

    Returns:
        The loaded document.
    """
    return yaml.load(text, Loader=_TaggedLoader)


def load_yaml(path: str | Path) -> Any:
    """Load a YAML configuration file with the study tags.

    Args:
        path: Configuration file.

    Returns:
        The loaded document.
    """
    return parse_yaml(Path(path).read_text(encoding="utf-8"))


def load_section(config_path: str | Path, *keys: str) -> dict[str, Any]:
    """Return the settings mapping found at ``keys`` in a plain YAML file.

    Only locating the section happens here; each component validates its own
    settings declaratively (pydantic) when it is constructed from them.

    Args:
        config_path: YAML configuration file.
        *keys: Path to the section, e.g. ``("data", "lila")``; none for the root.

    Returns:
        A shallow copy of the section mapping.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If there is no mapping at ``keys``.
    """
    section = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8"))
    for key in keys:
        section = section.get(key) if isinstance(section, dict) else None
    if not isinstance(section, dict):
        location = ".".join(keys) or "the document root"
        raise ValueError(f"{config_path}: expected a mapping at {location}")
    return dict(section)
