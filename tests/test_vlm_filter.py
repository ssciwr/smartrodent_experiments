"""Behavioral tests for the VLM image filter and its Ollama and vLLM backends."""

import base64
import json
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from vlm_test_support import ANSWER_SCHEMA, ScriptedBackend, answer

from smartrodent.vlm_filter import OllamaBackend, VllmBackend, VLMFilter

LABEL_COLUMNS = ["kept_animal", "rejected_animal"]
ANSWER_COLUMNS = ["vlm_animal_needs_human_review", "vlm_animal_evidence"]
OUTPUT_COLUMNS = LABEL_COLUMNS + ANSWER_COLUMNS


def make_filter(backend=None, **kwargs):
    options = {
        "prompt": "Answer with this schema: {answer_schema}",
        "system_prompt": "system",
        "answer_schema": ANSWER_SCHEMA,
        "taskname": "animal",
        "backend": backend if backend is not None else ScriptedBackend(),
    }
    options.update(kwargs)
    return VLMFilter(**options)


def make_records(tmp_path, rows):
    """Create records whose paths point to real temporary image files."""
    records = []
    for photo_id, filename in rows:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image-bytes")
        records.append({"photo.id": photo_id, "image_path": str(path)})
    return pd.DataFrame(records)


def labels_by_name(**labels):
    """Respond with the label configured for each image's file stem."""
    return lambda path: answer(labels[path.stem])


# --- configuration ---------------------------------------------------------------


def test_labels_and_output_columns_come_from_the_answer_schema():
    vlm_filter = make_filter()

    assert vlm_filter.labels == ("kept", "rejected")
    assert vlm_filter.output_columns == OUTPUT_COLUMNS


def test_backend_receives_the_prompt_with_the_schema_filled_in(tmp_path):
    backend = ScriptedBackend()
    records = make_records(tmp_path, [(1, "a.jpg")])

    make_filter(backend).filter_data({"mouse": records})

    (request,) = backend.requests
    prefix = "Answer with this schema: "
    assert request["prompt"].startswith(prefix)
    assert json.loads(request["prompt"].removeprefix(prefix)) == ANSWER_SCHEMA
    assert request["system_prompt"] == "system"
    assert request["schema"] == ANSWER_SCHEMA


def test_prompt_without_schema_placeholder_is_rejected():
    """The schema is written once, in the configuration, so the prompt must cite it."""
    with pytest.raises(ValueError, match="answer_schema"):
        make_filter(prompt="Answer as JSON with a label.")


def _schema_with(change):
    schema = deepcopy(ANSWER_SCHEMA)
    change(schema)
    return schema


@pytest.mark.parametrize(
    "schema",
    [
        _schema_with(lambda s: s["properties"].pop("label")),
        _schema_with(lambda s: s["properties"]["label"].pop("enum")),
        _schema_with(lambda s: s["properties"]["label"].update(enum=[])),
        _schema_with(lambda s: s["properties"]["label"].update(enum=["kept", "kept"])),
        _schema_with(lambda s: s["required"].remove("label")),
        # The label column kept_<task> equals the answer column vlm_<task>_<field>.
        _schema_with(
            lambda s: s["properties"].update(
                {"label": {"type": "string", "enum": ["vlm_animal_x"]}, "x_animal": {}}
            )
        ),
    ],
    ids=[
        "no_label",
        "no_enum",
        "empty_enum",
        "duplicate_enum",
        "optional",
        "collision",
    ],
)
def test_invalid_answer_schema_is_rejected(schema):
    with pytest.raises(ValueError):
        make_filter(answer_schema=schema)


def test_from_config_builds_the_configured_backend(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {},
                "filter": {
                    "prompt": "{answer_schema}",
                    "system_prompt": "system",
                    "answer_schema": ANSWER_SCHEMA,
                    "taskname": "custom",
                    "photo_id_column": "photo_key",
                    "image_path_column": "path",
                    "backend": {"name": "vlm_test_support:ScriptedBackend"},
                },
            }
        )
    )

    vlm_filter = VLMFilter.from_config(config_path)

    assert isinstance(vlm_filter.backend, ScriptedBackend)
    assert vlm_filter.photo_id_column == "photo_key"
    assert vlm_filter.image_path_column == "path"
    assert vlm_filter.output_columns[:2] == ["kept_custom", "rejected_custom"]


@pytest.mark.parametrize(
    ("backend", "backend_type"),
    [
        (
            {
                "name": "OllamaBackend",
                "parameters": {"model": "llava", "pull_model": False},
            },
            OllamaBackend,
        ),
        ({"name": "VllmBackend", "parameters": {"model": "org/model"}}, VllmBackend),
    ],
)
def test_from_config_names_the_bundled_backends(
    tmp_path, ollama_client, backend, backend_type
):
    """Constructing a backend contacts no server and loads no engine."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "filter": {
                    "prompt": "{answer_schema}",
                    "system_prompt": "system",
                    "answer_schema": ANSWER_SCHEMA,
                    "taskname": "animal",
                    "backend": backend,
                }
            }
        )
    )

    assert isinstance(VLMFilter.from_config(config_path).backend, backend_type)
    assert all(
        client.pull_calls == client.generate_calls == [] for client in ollama_client
    )


def test_from_config_rejects_unknown_backend(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "filter": {
                    "prompt": "{answer_schema}",
                    "system_prompt": "system",
                    "answer_schema": ANSWER_SCHEMA,
                    "taskname": "animal",
                    "backend": {"name": "UnknownBackend"},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="UnknownBackend"):
        VLMFilter.from_config(config_path)


# --- filtering ---------------------------------------------------------------------


def test_filter_data_preserves_frames_and_saves_every_answer(tmp_path):
    """Labels become booleans; the other answers become columns of their own (D13)."""
    records = make_records(tmp_path, [(101, "kept.jpg"), (102, "rejected.jpg")])
    records.index = [7, 11]
    original = records.copy(deep=True)
    backend = ScriptedBackend(
        lambda path: (
            answer("kept", evidence=["whiskers", "tail"])
            if path.stem == "kept"
            else answer("rejected", review=True)
        )
    )

    result = make_filter(backend).filter_data({"mouse": records})["mouse"]

    pd.testing.assert_frame_equal(records, original)
    pd.testing.assert_index_equal(result.index, original.index)
    assert result["kept_animal"].tolist() == [True, False]
    assert result["rejected_animal"].tolist() == [False, True]
    assert result["vlm_animal_needs_human_review"].tolist() == [False, True]
    # Lists are stored as JSON text so they survive the CSV round trip.
    assert json.loads(result.loc[7, "vlm_animal_evidence"]) == ["whiskers", "tail"]
    assert pd.isna(result.loc[11, "vlm_animal_evidence"])
    pd.testing.assert_frame_equal(result.drop(columns=OUTPUT_COLUMNS), original)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        None,
        answer("unexpected"),
        {"label": "kept"},  # a required answer is missing
    ],
    ids=["invalid_json", "not_an_object", "no_response", "unknown_label", "incomplete"],
)
def test_unusable_answer_sets_every_output_column_to_missing(tmp_path, response):
    records = make_records(tmp_path, [(55, "image.jpg")])
    backend = ScriptedBackend(lambda path: response)

    result = make_filter(backend).filter_data({"mouse": records})["mouse"]

    assert result[OUTPUT_COLUMNS].isna().all(axis=None)


def test_duplicate_photo_is_classified_once_and_result_is_repeated(tmp_path):
    image_path = tmp_path / "same.jpg"
    image_path.write_bytes(b"image-bytes")
    records = pd.DataFrame(
        {"photo.id": [55, 55, 56], "image_path": [str(image_path)] * 3},
        index=[2, 4, 8],
    )
    backend = ScriptedBackend()

    result = make_filter(backend).filter_data({"mouse": records})["mouse"]

    assert len(backend.image_paths) == 2
    assert result["kept_animal"].tolist() == [True, True, True]
    assert result["rejected_animal"].tolist() == [False, False, False]
    assert result.index.tolist() == [2, 4, 8]


@given(
    photo_ids=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=30)
)
def test_generated_within_species_duplicates_have_one_consistent_result(photo_ids):
    with tempfile.TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        paths = {photo_id: tmp_path / f"{photo_id}.jpg" for photo_id in set(photo_ids)}
        records = pd.DataFrame(
            {
                "photo.id": photo_ids,
                "image_path": [str(paths[photo_id]) for photo_id in photo_ids],
            }
        )
        backend = ScriptedBackend(
            lambda path: answer("kept" if int(path.stem) % 2 else "rejected")
        )

        result = make_filter(backend).filter_data({"mouse": records})["mouse"]

        assert len(backend.image_paths) == len(set(photo_ids))
        for photo_id, row in zip(photo_ids, result.to_dict("records"), strict=True):
            assert row["kept_animal"] is bool(photo_id % 2)
            assert row["rejected_animal"] is not bool(photo_id % 2)


def test_same_photo_id_in_different_species_is_processed_independently(tmp_path):
    mouse = make_records(tmp_path, [(55, "mouse.jpg")])
    rat = make_records(tmp_path, [(55, "rat.jpg")])
    backend = ScriptedBackend(labels_by_name(mouse="kept", rat="rejected"))

    result = make_filter(backend).filter_data({"mouse": mouse, "rat": rat})

    assert len(backend.image_paths) == 2
    assert result["mouse"]["kept_animal"].tolist() == [True]
    assert result["rat"]["rejected_animal"].tolist() == [True]


@pytest.mark.parametrize(
    ("records_by_species", "error_type", "message"),
    [
        ([], TypeError, "dict"),
        ({"mouse": []}, TypeError, "DataFrame"),
        ({"mouse": pd.DataFrame({"photo.id": [1]})}, ValueError, "image_path"),
        ({"mouse": pd.DataFrame({"image_path": ["x.jpg"]})}, ValueError, "photo.id"),
        (
            {"mouse": pd.DataFrame({"photo.id": [None], "image_path": ["x.jpg"]})},
            ValueError,
            "photo.id",
        ),
        (
            {"mouse": pd.DataFrame({"photo.id": [55], "image_path": [None]})},
            ValueError,
            "image_path",
        ),
        (
            {
                "mouse": pd.DataFrame(
                    {"photo.id": [55, 55], "image_path": ["one.jpg", "two.jpg"]}
                )
            },
            ValueError,
            "photo.id.*image_path",
        ),
    ],
    ids=[
        "not_a_dict",
        "not_a_frame",
        "no_path_column",
        "no_id_column",
        "null_id",
        "null_path",
        "duplicate_id_two_paths",
    ],
)
def test_invalid_input_fails_before_inference(records_by_species, error_type, message):
    backend = ScriptedBackend()

    with pytest.raises(error_type, match=message):
        make_filter(backend).filter_data(records_by_species)

    assert backend.requests == []


@pytest.mark.parametrize("existing_column", ["kept_animal", "vlm_animal_evidence"])
def test_existing_output_columns_raise_before_inference(tmp_path, existing_column):
    records = make_records(tmp_path, [(55, "existing.jpg")])
    records[existing_column] = True
    backend = ScriptedBackend()

    with pytest.raises(ValueError, match=existing_column):
        make_filter(backend).filter_data({"mouse": records})

    assert backend.requests == []
    assert records[existing_column].tolist() == [True]


def test_empty_input_mapping_returns_empty_mapping():
    assert make_filter().filter_data({}) == {}


def test_empty_species_frame_keeps_every_output_column_without_inference():
    backend = ScriptedBackend()
    records = pd.DataFrame(columns=["photo.id", "image_path"])

    result = make_filter(backend).filter_data({"empty": records})["empty"]

    assert result.empty
    assert list(result.columns) == ["photo.id", "image_path", *OUTPUT_COLUMNS]
    assert backend.image_paths == []


def test_backend_is_released_after_filtering(tmp_path):
    backend = ScriptedBackend()

    make_filter(backend).filter_data({"mouse": make_records(tmp_path, [(55, "a.jpg")])})

    assert backend.closed == 1


def test_backend_result_count_mismatch_raises_and_releases_backend(tmp_path):
    class MissingResults(ScriptedBackend):
        def generate(self, image_paths, **prompts):
            return []

    backend = MissingResults()

    with pytest.raises(RuntimeError, match="different number of results"):
        make_filter(backend).filter_data(
            {"mouse": make_records(tmp_path, [(55, "a.jpg")])}
        )

    assert backend.closed == 1


def test_inference_exception_propagates_and_releases_backend(tmp_path):
    def fail(path):
        raise RuntimeError("inference failed")

    backend = ScriptedBackend(fail)

    with pytest.raises(RuntimeError, match="inference failed"):
        make_filter(backend).filter_data(
            {"mouse": make_records(tmp_path, [(55, "a.jpg")])}
        )

    assert backend.closed == 1


# --- Ollama backend ------------------------------------------------------------------


@pytest.fixture
def ollama_client(monkeypatch):
    """Keep tests independent of a locally running Ollama daemon."""

    class FakeOllamaClient:
        def __init__(self, host=None):
            self.host = host
            self.pull_calls = []
            self.generate_calls = []
            clients.append(self)

        def pull(self, model, *, stream):
            self.pull_calls.append({"model": model, "stream": stream})
            return {"status": "success"}

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            # Echo the image so tests can check the order of the responses.
            return types.SimpleNamespace(response=kwargs.get("images", [""])[0])

    clients = []
    monkeypatch.setattr("smartrodent.vlm_filter.ollama.Client", FakeOllamaClient)
    return clients


def test_ollama_sends_each_image_with_the_schema_and_configured_options(
    tmp_path, ollama_client
):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")
    options = {"temperature": 0.1, "repeat_penalty": 1.1, "num_gpu": 0}
    backend = OllamaBackend(model="llava", options=options)

    responses = backend.generate(
        [image_path], system_prompt="system", prompt="prompt", schema=ANSWER_SCHEMA
    )

    (client,) = ollama_client
    assert responses == ["YWJj"]
    assert client.pull_calls == [{"model": "llava", "stream": False}]
    assert client.generate_calls == [
        {
            "model": "llava",
            "system": "system",
            "prompt": "prompt",
            "images": ["YWJj"],
            "format": ANSWER_SCHEMA,
            "stream": False,
            "think": False,
            "options": options,
        }
    ]


def test_ollama_parallel_workers_keep_the_input_order(tmp_path, ollama_client):
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(str(index).encode())
        paths.append(path)
    backend = OllamaBackend(model="llava", pull_model=False, num_workers=3)

    responses = backend.generate(paths, system_prompt="", prompt="", schema={})

    assert [base64.b64decode(response) for response in responses] == [
        str(index).encode() for index in range(5)
    ]
    assert ollama_client[0].pull_calls == []


def test_ollama_close_unloads_the_model_once(ollama_client):
    backend = OllamaBackend(model="llava", pull_model=False)

    backend.close()
    backend.close()

    assert ollama_client[0].generate_calls == [{"model": "llava", "keep_alive": 0}]


# --- vLLM backend --------------------------------------------------------------------


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a fake ``vllm`` package that records engine and sampling setup."""
    seen = {"engines": [], "chats": [], "sampling": [], "shutdowns": 0}

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            seen["sampling"].append(kwargs)

    class FakeEngine:
        def shutdown(self):
            seen["shutdowns"] += 1

    class FakeLLM:
        def __init__(self, **kwargs):
            seen["engines"].append(kwargs)
            self.llm_engine = FakeEngine()

        def chat(self, conversations, **options):
            seen["chats"].append(conversations)
            return [
                types.SimpleNamespace(
                    outputs=(types.SimpleNamespace(text=f"answer {index}"),)
                )
                for index, _ in enumerate(conversations)
            ]

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    return seen


def test_vllm_builds_one_engine_and_batches_conversations(tmp_path, fake_vllm):
    """Engine and sampling settings pass through; the schema constrains decoding."""
    paths = [tmp_path / "a.JPG", tmp_path / "b.png", tmp_path / "c.png"]
    for path in paths:
        path.write_bytes(path.stem.encode())
    backend = VllmBackend(
        model="org/model",
        batch_size=2,
        engine={"gpu_memory_utilization": 0.4, "max_model_len": 2048},
        sampling={"temperature": 0, "max_tokens": 17},
    )

    responses = backend.generate(
        paths, system_prompt="system", prompt="prompt", schema=ANSWER_SCHEMA
    )
    backend.generate(paths[:1], system_prompt="system", prompt="prompt", schema={})

    assert responses == ["answer 0", "answer 1", "answer 0"]
    assert fake_vllm["engines"] == [
        {
            "model": "org/model",
            "limit_mm_per_prompt": {"image": 1},
            "gpu_memory_utilization": 0.4,
            "max_model_len": 2048,
        }
    ]
    # Only configured sampling settings pass through; nothing is injected.
    sampling = fake_vllm["sampling"][0]
    structured_outputs = sampling.pop("structured_outputs")
    assert sampling == {"temperature": 0, "max_tokens": 17}
    assert structured_outputs.kwargs == {"json": ANSWER_SCHEMA}
    assert [len(chat) for chat in fake_vllm["chats"][:2]] == [2, 1]
    first = fake_vllm["chats"][0][0]
    assert first[0] == {"role": "system", "content": "system"}
    assert first[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YQ=="}},
        {"type": "text", "text": "prompt"},
    ]
    assert fake_vllm["chats"][0][1][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_vllm_injects_no_sampling_settings(tmp_path, fake_vllm):
    image = tmp_path / "a.png"
    image.write_bytes(b"a")

    VllmBackend(model="org/model").generate(
        [image], system_prompt="", prompt="", schema={}
    )

    assert list(fake_vllm["sampling"][0]) == ["structured_outputs"]


def test_vllm_loads_no_engine_without_images(fake_vllm):
    backend = VllmBackend(model="org/model")

    assert backend.generate([], system_prompt="", prompt="", schema={}) == []
    assert fake_vllm["engines"] == []


@pytest.mark.parametrize(
    "settings",
    [
        {"engine": {"model": "other"}},
        {"engine": {"limit_mm_per_prompt": {"image": 2}}},
        {"sampling": {"structured_outputs": None}},
    ],
    ids=["model", "images_per_prompt", "structured_outputs"],
)
def test_vllm_settings_cannot_override_what_the_backend_supplies(settings):
    with pytest.raises(ValueError):
        VllmBackend(model="org/model", **settings)


def test_vllm_close_shuts_down_the_engine(tmp_path, fake_vllm):
    image = tmp_path / "a.png"
    image.write_bytes(b"a")
    backend = VllmBackend(model="org/model")
    backend.generate([image], system_prompt="", prompt="", schema={})

    backend.close()
    backend.close()

    assert fake_vllm["shutdowns"] == 1
