"""Opt-in integration tests of the VLM filter against a real Ollama or vLLM backend."""

import os
from pathlib import Path

import pandas as pd
import pytest

from smartrodent.vlm_filter import OllamaBackend, VllmBackend, VLMFilter

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SYSTEM_PROMPT = (
    "You are a careful wildlife camera-trap image classifier. "
    "Return only a valid JSON object."
)
PROMPT = """
Classify this image for a rodent filtering workflow.

Use label "kept" when a rodent or rodent-like small mammal is visibly present.
Use label "rejected" when no animal is visibly present.

Return JSON matching this schema:
{answer_schema}
""".strip()
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["kept", "rejected"]},
        "visible_animal": {"type": "boolean"},
        "needs_human_review": {"type": "boolean"},
    },
    "required": ["label", "visible_animal", "needs_human_review"],
    "additionalProperties": False,
}

pytestmark = [
    pytest.mark.vlm_integration,
    pytest.mark.skipif(
        os.getenv("SMARTRODENT_RUN_VLM_INTEGRATION") != "1",
        reason=(
            "Real VLM integration tests require an Ollama server or a vLLM-capable "
            "GPU; set SMARTRODENT_RUN_VLM_INTEGRATION=1 to run them."
        ),
    ),
]


def integration_image_path(env_var: str, default_name: str) -> Path:
    default_path = Path(__file__).resolve().parents[1] / "resources" / default_name
    path = Path(os.getenv(env_var, default_path))
    if not path.exists():
        pytest.skip(
            f"Missing integration image {path}. Set {env_var} or add {default_path}."
        )
    return path


def make_ollama_backend():
    return OllamaBackend(
        model=os.getenv("SMARTRODENT_OLLAMA_MODEL", "qwen2.5vl:3b"),
        options={"temperature": 0},
    )


def make_vllm_backend():
    pytest.importorskip("vllm")
    return VllmBackend(
        model=os.getenv("SMARTRODENT_VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        batch_size=2,
        engine={
            "gpu_memory_utilization": float(
                os.getenv("SMARTRODENT_VLLM_GPU_MEMORY_UTILIZATION", "0.8")
            ),
            "max_model_len": int(os.getenv("SMARTRODENT_VLLM_MAX_MODEL_LEN", "16384")),
        },
        sampling={
            "temperature": 0,
            "max_tokens": int(os.getenv("SMARTRODENT_VLLM_MAX_NEW_TOKENS", "1024")),
        },
    )


@pytest.mark.parametrize(
    "make_backend", [make_ollama_backend, make_vllm_backend], ids=["ollama", "vllm"]
)
def test_real_backend_classifies_rodent_and_empty_images(make_backend):
    rodent = integration_image_path(
        "SMARTRODENT_INTEGRATION_RODENT_IMAGE", "rodent.jpg"
    )
    empty = integration_image_path("SMARTRODENT_INTEGRATION_EMPTY_IMAGE", "empty.jpg")
    vlm_filter = VLMFilter(
        prompt=PROMPT,
        system_prompt=SYSTEM_PROMPT,
        answer_schema=ANSWER_SCHEMA,
        taskname="rodent",
        backend=make_backend(),
    )
    records = pd.DataFrame(
        {"photo.id": [1, 2], "image_path": [str(rodent), str(empty)]}
    )

    result = vlm_filter.filter_data({"rodent": records})["rodent"]

    assert result["kept_rodent"].tolist() == [True, False]
    assert result["vlm_rodent_visible_animal"].tolist() == [True, False]
    # needs_human_review is not checked: it depends on the model.
    assert result["vlm_rodent_needs_human_review"].notna().all()
