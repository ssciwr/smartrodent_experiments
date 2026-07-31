import os
from pathlib import Path

import pytest
from smartrodent.filter import FilterVLLM, FilterOllama

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
Use label "unsure" only when the image is ambiguous.

Return exactly this JSON schema without ommiting anything:
{
  "label": "kept" | "rejected" | "unsure",
  "visible_animal": true | false,
  "evidence_kept": ["short reason"],
  "evidence_rejected": ["short reason"],
  "image_quality": "clear" | "poor" | "unusable",
  "needs_human_review": true | false
}

Make sure that you return valid JSON.
""".strip()

pytestmark = pytest.mark.skipif(
    os.getenv("SMARTRODENT_SKIP_VLM_INTEGRATION") == "1",
    reason="Set SMARTRODENT_SKIP_VLM_INTEGRATION=1 to skip real VLM integration tests.",
)


def integration_image_path(env_var: str, default_name: str) -> Path:
    default_path = Path(__file__).resolve().parents[1] / "resources" / default_name
    path = Path(os.getenv(env_var, default_path))
    if not path.exists():
        pytest.skip(
            f"Missing integration image {path}. Set {env_var} or add {default_path}."
        )
    return path


@pytest.fixture
def rodent_image() -> Path:
    return integration_image_path("SMARTRODENT_INTEGRATION_RODENT_IMAGE", "rodent.jpg")


@pytest.fixture
def empty_image() -> Path:
    return integration_image_path("SMARTRODENT_INTEGRATION_EMPTY_IMAGE", "empty.jpg")


def make_ollama_filter(tmp_path):
    return FilterOllama(
        prompt=PROMPT,
        system_prompt=SYSTEM_PROMPT,
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".jpeg", ".png", ".webp"},
        model=os.getenv(
            "SMARTRODENT_OLLAMA_MODEL",
            "qwen2.5vl:3b",
        ),
    )


def make_vllm_filter(tmp_path):
    pytest.importorskip("vllm")

    return FilterVLLM(
        prompt=PROMPT,
        system_prompt=SYSTEM_PROMPT,
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".jpeg", ".png", ".webp"},
        model_name=os.getenv(
            "SMARTRODENT_VLLM_MODEL",
            "Qwen/Qwen2.5-VL-3B-Instruct",  # "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        ),
        gpu_memory_utilization=float(
            os.getenv("SMARTRODENT_VLLM_GPU_MEMORY_UTILIZATION", "0.8")
        ),
        max_model_len=int(os.getenv("SMARTRODENT_VLLM_MAX_MODEL_LEN", "16384")),
        max_new_tokens=int(os.getenv("SMARTRODENT_VLLM_MAX_NEW_TOKENS", "1024")),
        batch_size=2,
    )


@pytest.mark.parametrize(
    "make_filter", [make_ollama_filter, make_vllm_filter], ids=["ollama", "vllm"]
)
def test_ollama_classifies_rodent_and_empty_images(
    make_filter, tmp_path, rodent_image, empty_image
):
    with make_filter(tmp_path) as vlm_filter:
        rodent_result = vlm_filter.classify(rodent_image)
        empty_result = vlm_filter.classify(empty_image)
        assert rodent_result["parse_error"] is False
        assert rodent_result["label"] == "kept"
        assert rodent_result["visible_animal"] is True

        assert empty_result["parse_error"] is False
        assert empty_result["label"] == "rejected"
        assert empty_result["visible_animal"] is False

        # needs_human_review is not tested here b/c it's model dependent
