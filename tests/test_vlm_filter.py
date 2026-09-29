import json
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartrodent.filter import FilterOllama, FilterVLLM, VLMFilter


@pytest.fixture(autouse=True)
def fake_ollama_client(monkeypatch):
    """Keep unit tests independent of a locally running Ollama daemon."""

    class FakeOllamaClient:
        def __init__(self, host=None):
            self.host = host
            self.pull_calls = []
            self.generate_calls = []

        def pull(self, model, *, stream):
            self.pull_calls.append({"model": model, "stream": stream})
            return {"status": "success"}

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return types.SimpleNamespace(
                response=json.dumps({"label": "kept", "visible_animal": True})
            )

    monkeypatch.setattr("smartrodent.filter.ollama.Client", FakeOllamaClient)


def make_filter(tmp_path, *, species=None, mode="copy"):
    return VLMFilter(
        prompt="is this useful?",
        system_prompt="classify wildlife",
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".png"},
        species=species,
        mode=mode,
    )


def write_image_tree(root: Path) -> dict[str, Path]:
    paths = {
        "mouse": root / "Mouse" / "a.JPG",
        "mouse_txt": root / "Mouse" / "notes.txt",
        "rat": root / "rat" / "b.png",
        "bat": root / "bat" / "c.jpeg",
        "nested": root / "rat" / "nested" / "ignored.jpg",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image-bytes")
    return paths


def test_parse_response_normalizes_valid_payload_and_defaults_review_flag():
    raw = json.dumps(
        {
            "label": " Unsure ",
            "visible_animal": True,
            "evidence_kept": ["tail"],
            "evidence_rejected": ["blur"],
        }
    )

    parsed = VLMFilter.parse_response(raw)

    assert parsed == {
        "label": "unsure",
        "visible_animal": True,
        "evidence_kept": ["tail"],
        "evidence_rejected": ["blur"],
        "image_quality": "unsure",
        "needs_human_review": True,
        "raw_response": raw,
        "parse_error": False,
    }


@pytest.mark.parametrize("raw", ["not json", "{"])
def test_parse_response_returns_failure_for_invalid_json(raw):
    parsed = VLMFilter.parse_response(raw)

    assert parsed["label"] == "failure"
    assert parsed["visible_animal"] is None
    assert parsed["image_quality"] == "unknown"
    assert parsed["needs_human_review"] is True
    assert parsed["raw_response"] == raw
    assert parsed["parse_error"] is True


def test_parse_response_maps_unknown_labels_to_failure_and_honors_review_flag():
    raw = json.dumps(
        {
            "label": "definitely",
            "visible_animal": False,
            "image_quality": "poor",
            "needs_human_review": False,
        }
    )

    parsed = VLMFilter.parse_response(raw)

    assert parsed["label"] == "failure"
    assert parsed["visible_animal"] is False
    assert parsed["evidence_kept"] == []
    assert parsed["evidence_rejected"] == []
    assert parsed["image_quality"] == "poor"
    assert parsed["needs_human_review"] is False
    assert parsed["parse_error"] is False


def test_init_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be 'move' or 'copy'"):
        make_filter(tmp_path, mode="link")


def test_collect_image_paths_filters_species_case_insensitively_and_suffixes(tmp_path):
    image_paths = write_image_tree(tmp_path / "imgs")
    vlm_filter = make_filter(tmp_path, species=["MOUSE", "rat"])

    assert vlm_filter.collect_image_paths() == [
        image_paths["mouse"],
        image_paths["rat"],
    ]


def test_collect_image_paths_uses_all_species_when_none_requested(tmp_path):
    image_paths = write_image_tree(tmp_path / "imgs")
    vlm_filter = make_filter(tmp_path)

    assert vlm_filter.collect_image_paths() == [
        image_paths["mouse"],
        image_paths["rat"],
    ]


def test_copy_with_structure_preserves_relative_path_and_ignores_same_file(tmp_path):
    source = tmp_path / "imgs" / "Mouse" / "a.JPG"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    vlm_filter = make_filter(tmp_path)

    copied = vlm_filter.copy_with_structure(source, tmp_path / "kept")
    same_file = vlm_filter.copy_with_structure(source, tmp_path / "imgs")

    assert copied == tmp_path / "kept" / "Mouse" / "a.JPG"
    assert copied.read_bytes() == b"image-bytes"
    assert same_file == source


def test_copy_with_structure_moves_when_configured(tmp_path):
    source = tmp_path / "imgs" / "Mouse" / "a.JPG"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    vlm_filter = make_filter(tmp_path, mode="move")

    moved = vlm_filter.copy_with_structure(source, tmp_path / "kept")

    assert moved.read_bytes() == b"image-bytes"
    assert not source.exists()


def test_copy_with_structure_reraises_unexpected_copy_errors(tmp_path):
    source = tmp_path / "imgs" / "Mouse" / "a.JPG"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image-bytes")
    vlm_filter = make_filter(tmp_path)

    def broken_copy(_src, _dst):
        raise OSError("disk full")

    vlm_filter.data_func = broken_copy

    with pytest.raises(OSError, match="disk full"):
        vlm_filter.copy_with_structure(source, tmp_path / "kept")


def test_dest_by_label_maps_every_result_bucket(tmp_path):
    vlm_filter = make_filter(tmp_path)

    assert vlm_filter.dest_by_label == {
        "kept": tmp_path / "kept",
        "rejected": tmp_path / "rejected",
        "unsure": tmp_path / "unsure",
        "failure": tmp_path / "failure",
    }


def test_save_results_writes_csv_next_to_image_root(tmp_path):
    vlm_filter = make_filter(tmp_path)
    vlm_filter.imgs_root.mkdir()
    df = pd.DataFrame([{"label": "kept", "species": "Mouse"}])

    out_path = vlm_filter.save_results(df)

    assert out_path == tmp_path / "filter_results.csv"
    assert "kept" in out_path.read_text()


def test_from_config_creates_ollama_filter_outputs_and_copies_config(tmp_path):
    imgs_root = tmp_path / "imgs"
    imgs_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "ollama",
                "prompt": "prompt",
                "system_prompt": "system",
                "mode": "move",
                "species": ["Mouse"],
                "paths": {"imgs_root": str(imgs_root), "image_suffixes": [".jpg"]},
                "ollama": {"model": "llava"},
            }
        )
    )

    vlm_filter = VLMFilter.from_config(config_path)

    assert isinstance(vlm_filter, FilterOllama)
    assert vlm_filter.imgs_root == imgs_root
    assert vlm_filter.model == "llava"
    assert vlm_filter.client.host is None
    assert vlm_filter.client.pull_calls == [{"model": "llava", "stream": False}]
    assert vlm_filter.species == ["mouse"]
    assert vlm_filter.data_func is shutil.move
    for output_name in (
        "filtered_kept",
        "filtered_undecided",
        "filtered_rejected",
        "filtered_failure",
    ):
        assert (
            tmp_path / output_name / config_path.name
        ).read_text() == config_path.read_text()


def test_from_config_creates_vllm_filter(tmp_path):
    imgs_root = tmp_path / "imgs"
    imgs_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "vllm",
                "prompt": "prompt",
                "system_prompt": "system",
                "paths": {"imgs_root": str(imgs_root), "image_suffixes": [".png"]},
                "vllm": {
                    "model": "org/model",
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 4096,
                    "max_new_tokens": 42,
                    "batch_size": 3,
                },
            }
        )
    )

    vlm_filter = VLMFilter.from_config(config_path)

    assert isinstance(vlm_filter, FilterVLLM)
    assert vlm_filter.model_name == "org/model"
    assert vlm_filter.model_tag == "model"
    assert vlm_filter.gpu_memory_utilization == 0.5
    assert vlm_filter.max_model_len == 4096
    assert vlm_filter.max_new_tokens == 42
    assert vlm_filter.batch_size == 3


def test_from_config_rejects_unknown_backend(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "spaceship",
                "prompt": "prompt",
                "system_prompt": "system",
                "paths": {
                    "imgs_root": str(tmp_path / "imgs"),
                    "image_suffixes": [".jpg"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="Unknown backend 'spaceship'"):
        VLMFilter.from_config(config_path)


def make_ollama_filter(tmp_path):
    return FilterOllama(
        prompt="prompt",
        system_prompt="system",
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".png"},
        model="llava",
    )


def test_ollama_classify_uses_persistent_client(tmp_path):
    image_path = tmp_path / "imgs" / "Mouse" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"abc")
    vlm_filter = make_ollama_filter(tmp_path)
    parsed = vlm_filter.classify(image_path)

    assert parsed["label"] == "kept"
    assert vlm_filter.client.host is None
    assert vlm_filter.client.pull_calls == [{"model": "llava", "stream": False}]
    assert vlm_filter.client.generate_calls == [
        {
            "model": "llava",
            "system": "system",
            "prompt": "prompt",
            "images": ["YWJj"],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
    ]


def test_ollama_can_skip_automatic_model_pull(tmp_path):
    vlm_filter = FilterOllama(
        prompt="prompt",
        system_prompt="system",
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg"},
        model="llava",
        pull_model=False,
    )

    assert vlm_filter.client.pull_calls == []


def test_ollama_model_tag_is_model_name(tmp_path):
    assert make_ollama_filter(tmp_path).model_tag == "llava"


def test_ollama_filter_data_routes_each_classification(monkeypatch, tmp_path):
    image_paths = write_image_tree(tmp_path / "imgs")
    vlm_filter = make_ollama_filter(tmp_path)
    labels = {
        image_paths["mouse"]: "kept",
        image_paths["rat"]: "rejected",
    }

    def fake_classify(path):
        return {"label": labels[path], "parse_error": False}

    monkeypatch.setattr(vlm_filter, "classify", fake_classify)

    with vlm_filter:
        df = vlm_filter.filter_data()

    assert df[["label", "species"]].to_dict("records") == [
        {"label": "kept", "species": "Mouse"},
        {"label": "rejected", "species": "rat"},
    ]
    assert (tmp_path / "kept" / "Mouse" / "a.JPG").read_bytes() == b"image-bytes"
    assert (tmp_path / "rejected" / "rat" / "b.png").read_bytes() == b"image-bytes"
    assert vlm_filter.client.generate_calls == [{"model": "llava", "keep_alive": 0}]


def test_ollama_context_manager_unloads_model_once(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")

    with make_ollama_filter(tmp_path) as vlm_filter:
        vlm_filter.classify(image_path)

    assert vlm_filter.client.generate_calls[-1] == {
        "model": "llava",
        "keep_alive": 0,
    }
    assert len(vlm_filter.client.generate_calls) == 2


def test_ollama_filter_data_unloads_model_when_classification_fails(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "imgs" / "Mouse" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"abc")
    vlm_filter = make_ollama_filter(tmp_path)

    def broken_classify(_path):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(vlm_filter, "classify", broken_classify)

    with pytest.raises(RuntimeError, match="inference failed"):
        vlm_filter.filter_data()

    assert vlm_filter.client.generate_calls == [{"model": "llava", "keep_alive": 0}]


def make_vllm_filter(tmp_path, *, batch_size=2):
    return FilterVLLM(
        prompt="prompt",
        system_prompt="system",
        imgs_root=tmp_path / "imgs",
        kept_root=tmp_path / "kept",
        unsure_root=tmp_path / "unsure",
        rejected_root=tmp_path / "rejected",
        failure_root=tmp_path / "failure",
        image_suffixes={".jpg", ".png"},
        model_name="/models/qwen",
        batch_size=batch_size,
    )


def test_vllm_image_to_data_url_uses_mime_suffix(tmp_path):
    jpg = tmp_path / "a.JPG"
    png = tmp_path / "b.png"
    no_suffix = tmp_path / "image"
    jpg.write_bytes(b"jpg")
    png.write_bytes(b"png")
    no_suffix.write_bytes(b"raw")

    assert FilterVLLM.image_to_data_url(jpg) == "data:image/jpeg;base64,anBn"
    assert FilterVLLM.image_to_data_url(png) == "data:image/png;base64,cG5n"
    assert FilterVLLM.image_to_data_url(no_suffix) == "data:image/jpeg;base64,cmF3"


def test_vllm_build_conversation_embeds_system_prompt_image_and_text(tmp_path):
    image_path = tmp_path / "a.png"
    image_path.write_bytes(b"abc")
    vlm_filter = make_vllm_filter(tmp_path)

    conversation = vlm_filter.build_conversation(image_path)

    assert conversation == [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YWJj"},
                },
                {"type": "text", "text": "prompt"},
            ],
        },
    ]


def test_vllm_ensure_engine_lazily_builds_fake_engine(monkeypatch, tmp_path):
    seen = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            seen["llm_kwargs"] = kwargs

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            seen["structured_kwargs"] = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            seen["sampling_kwargs"] = kwargs
            self.temperature = kwargs["temperature"]
            self.max_tokens = kwargs["max_tokens"]
            self.structured_outputs = kwargs["structured_outputs"]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling_params = types.ModuleType("vllm.sampling_params")
    fake_sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params)

    vlm_filter = make_vllm_filter(tmp_path)
    vlm_filter._ensure_engine()
    first_llm = vlm_filter._llm
    first_sampling_params = vlm_filter._sampling_params
    vlm_filter._ensure_engine()

    assert isinstance(first_llm, FakeLLM)
    assert isinstance(first_sampling_params, FakeSamplingParams)
    assert vlm_filter._llm is first_llm
    assert vlm_filter._sampling_params is first_sampling_params
    assert seen["llm_kwargs"] == {
        "model": "/models/qwen",
        "limit_mm_per_prompt": {"image": 1},
        "gpu_memory_utilization": 0.7,
        "max_model_len": 8192,
    }
    assert seen["structured_kwargs"] == {"json": FilterVLLM.RESPONSE_JSON_SCHEMA}
    assert seen["sampling_kwargs"]["temperature"] == 0
    assert seen["sampling_kwargs"]["max_tokens"] == 300
    assert isinstance(
        seen["sampling_kwargs"]["structured_outputs"], FakeStructuredOutputsParams
    )


def test_vllm_classify_batch_uses_engine_and_parses_outputs(monkeypatch, tmp_path):
    image_paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for image_path in image_paths:
        image_path.write_bytes(b"abc")
    vlm_filter = make_vllm_filter(tmp_path)
    seen = {}

    class FakeGeneratedText:
        def __init__(self, text):
            self.text = text

    class FakeOutput:
        def __init__(self, text):
            self.outputs = [FakeGeneratedText(text)]

    class FakeLLM:
        def chat(self, conversations, *, sampling_params, use_tqdm):
            seen.update(
                {
                    "conversations": conversations,
                    "sampling_params": sampling_params,
                    "use_tqdm": use_tqdm,
                }
            )
            return [
                FakeOutput(json.dumps({"label": "kept"})),
                FakeOutput(json.dumps({"label": "unsure"})),
            ]

    def fake_ensure_engine():
        seen["ensured"] = True
        vlm_filter._llm = FakeLLM()
        vlm_filter._sampling_params = object()

    monkeypatch.setattr(vlm_filter, "_ensure_engine", fake_ensure_engine)

    parsed = vlm_filter._classify_batch(image_paths)

    assert [row["label"] for row in parsed] == ["kept", "unsure"]
    assert seen["ensured"] is True
    assert len(seen["conversations"]) == 2
    assert seen["sampling_params"] is vlm_filter._sampling_params
    assert seen["use_tqdm"] is False


def test_vllm_classify_one_image_uses_batch_interface(monkeypatch, tmp_path):
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"abc")
    vlm_filter = make_vllm_filter(tmp_path)

    monkeypatch.setattr(
        vlm_filter,
        "_classify_batch",
        lambda image_paths: [{"label": "kept", "path": image_paths[0]}],
    )

    assert vlm_filter.classify(image_path) == {"label": "kept", "path": image_path}


def test_vllm_filter_data_batches_and_routes_results(monkeypatch, tmp_path):
    image_paths = write_image_tree(tmp_path / "imgs")
    vlm_filter = make_vllm_filter(tmp_path, batch_size=1)
    monkeypatch.setattr(vlm_filter, "_ensure_engine", lambda: None)
    seen_chunks = []
    labels = {
        image_paths["mouse"]: "unsure",
        image_paths["rat"]: "failure",
    }

    def fake_classify_batch(chunk):
        seen_chunks.append(chunk)
        return [{"label": labels[path], "parse_error": False} for path in chunk]

    monkeypatch.setattr(vlm_filter, "_classify_batch", fake_classify_batch)

    with vlm_filter:
        df = vlm_filter.filter_data()

    assert seen_chunks == [[image_paths["mouse"]], [image_paths["rat"]]]
    assert df[["label", "species"]].to_dict("records") == [
        {"label": "unsure", "species": "Mouse"},
        {"label": "failure", "species": "rat"},
    ]
    assert (tmp_path / "unsure" / "Mouse" / "a.JPG").read_bytes() == b"image-bytes"
    assert (tmp_path / "failure" / "rat" / "b.png").read_bytes() == b"image-bytes"


def test_vllm_context_manager_shuts_down_engine(tmp_path):
    vlm_filter = make_vllm_filter(tmp_path)
    seen = {"shutdowns": 0}

    class FakeEngine:
        def shutdown(self):
            seen["shutdowns"] += 1

    vlm_filter._llm = types.SimpleNamespace(llm_engine=FakeEngine())
    vlm_filter._sampling_params = object()

    with vlm_filter:
        pass

    assert seen["shutdowns"] == 1
    assert vlm_filter._llm is None
    assert vlm_filter._sampling_params is None


def test_vlm_filter_base_methods_are_not_implemented(tmp_path):
    vlm_filter = make_filter(tmp_path)

    with pytest.raises(NotImplementedError):
        _ = vlm_filter.model_tag
    with pytest.raises(NotImplementedError):
        vlm_filter.filter_data()
