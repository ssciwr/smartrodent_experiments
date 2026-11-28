"""A scripted VLM backend and answer schema shared by the VLM filter tests."""

import json
from collections.abc import Callable
from pathlib import Path

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["kept", "rejected"]},
        "needs_human_review": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "needs_human_review"],
    "additionalProperties": False,
}


def answer(label: str, *, review: bool = False, **extra) -> dict:
    """A complete answer to ``ANSWER_SCHEMA``."""
    return {"label": label, "needs_human_review": review, **extra}


class ScriptedBackend:
    """Answer each image through ``respond(path)`` and record every request.

    ``respond`` returns an answer mapping (sent as JSON), a raw string, or None for
    a failed generation. The default keeps every image without review.
    """

    def __init__(self, respond: Callable[[Path], object] | None = None):
        self.respond = respond or (lambda path: answer("kept"))
        self.requests: list[dict] = []
        self.closed = 0

    def generate(self, image_paths, *, system_prompt, prompt, schema):
        """Return one raw response per image, in order."""
        self.requests.append(
            {
                "image_paths": list(image_paths),
                "system_prompt": system_prompt,
                "prompt": prompt,
                "schema": schema,
            }
        )
        responses = [self.respond(Path(path)) for path in image_paths]
        return [
            json.dumps(response) if isinstance(response, dict) else response
            for response in responses
        ]

    def close(self):
        """Count releases so tests can check that the filter released the backend."""
        self.closed += 1

    @property
    def image_paths(self) -> list[Path]:
        """Every image requested so far, in request order."""
        return [
            Path(path) for request in self.requests for path in request["image_paths"]
        ]
