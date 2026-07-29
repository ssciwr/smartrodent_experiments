import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest
import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SYSTEM_PROMPT = (
    "You are a careful wildlife camera-trap image classifier. "
    "Return only a valid JSON object."
)
PROMPT = """
Classify this image for a rodent filtering workflow.

Use label "kept" when a rodent or rodent-like small mammal is visibly present.
Use label "rejected" when no animal is visibly present.
Use label "unsure" only when the image is ambiguous.

Return exactly this JSON schema:
{
  "label": "kept" | "rejected" | "unsure",
  "visible_animal": true | false,
  "evidence_kept": ["short reason"],
  "evidence_rejected": ["short reason"],
  "image_quality": "clear" | "poor" | "unusable",
  "needs_human_review": true | false
}
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
    from smartrodent.filter import FilterOllama

    url = os.getenv("SMARTRODENT_OLLAMA_URL", "http://localhost:11434/api/generate")
    assert_ollama_available(url)

    return FilterOllama(
        prompt=PROMPT,
        system_prompt=SYSTEM_PROMPT,
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".jpeg", ".png", ".webp"},
        url=url,
        model=os.getenv(
            "SMARTRODENT_OLLAMA_MODEL",
            "hf.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M",
        ),
    )


def assert_ollama_available(generate_url: str) -> None:
    parsed = urlparse(generate_url)
    tags_url = urlunparse(parsed._replace(path="/api/tags", params="", query=""))
    try:
        response = requests.get(tags_url, timeout=2)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.fail(
            "Ollama integration test requires a reachable Ollama server at "
            f"{tags_url}. Start Ollama or set SMARTRODENT_OLLAMA_URL. "
            f"Original error: {exc}",
            pytrace=False,
        )


def make_vllm_filter(tmp_path):
    pytest.importorskip("vllm")
    from smartrodent.filter import FilterVLLM

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
            "SMARTRODENT_VLLM_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        ),
        gpu_memory_utilization=float(
            os.getenv("SMARTRODENT_VLLM_GPU_MEMORY_UTILIZATION", "0.7")
        ),
        max_model_len=int(os.getenv("SMARTRODENT_VLLM_MAX_MODEL_LEN", "4096")),
        max_new_tokens=int(os.getenv("SMARTRODENT_VLLM_MAX_NEW_TOKENS", "160")),
        batch_size=2,
    )


def assert_rodent_result(result: dict) -> None:
    assert result["parse_error"] is False
    assert result["label"] == "kept"
    assert result["visible_animal"] is True
    assert result["needs_human_review"] is False


def assert_empty_result(result: dict) -> None:
    assert result["parse_error"] is False
    assert result["label"] == "rejected"
    assert result["visible_animal"] is False
    assert result["needs_human_review"] is False


def test_ollama_smolvlm2_classifies_rodent_and_empty_images(
    tmp_path, rodent_image, empty_image
):
    vlm_filter = make_ollama_filter(tmp_path)

    rodent_result = vlm_filter.classify_image(rodent_image)
    empty_result = vlm_filter.classify_image(empty_image)

    assert_rodent_result(rodent_result)
    assert_empty_result(empty_result)


def test_vllm_smolvlm2_classifies_rodent_and_empty_images(
    tmp_path, rodent_image, empty_image
):
    vlm_filter = make_vllm_filter(tmp_path)

    rodent_result, empty_result = vlm_filter.classify_images_batch(
        [rodent_image, empty_image]
    )

    assert_rodent_result(rodent_result)
    assert_empty_result(empty_result)
