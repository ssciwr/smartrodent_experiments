import json
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartrodent.filter import FilterOllama, FilterVLLM, VLMFilter


@pytest.fixture(autouse=True)
def fake_ollama_client(monkeypatch):
    """Keep tests independent of a locally running Ollama daemon."""

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


def make_ollama_filter(tmp_path, **kwargs):
    return FilterOllama(
        prompt="prompt",
        system_prompt="system",
        labels=["kept", "rejected"],
        taskname="animal",
        photo_id_column="photo.id",
        image_path_column="image_path",
        model="llava",
        **kwargs,
    )


def make_vllm_filter(tmp_path, *, batch_size=2, **kwargs):
    return FilterVLLM(
        prompt="prompt",
        system_prompt="system",
        labels=["kept", "rejected"],
        taskname="animal",
        photo_id_column="photo.id",
        image_path_column="image_path",
        model_name="/models/qwen",
        batch_size=batch_size,
        **kwargs,
    )


def make_records(tmp_path, rows):
    """Create records whose paths point to real temporary image files."""
    records = []
    for photo_id, label, filename in rows:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image-bytes")
        records.append(
            {"photo.id": photo_id, "image_path": str(path), "source": filename}
        )
    return pd.DataFrame(records)


def test_parse_response_accepts_only_configured_labels(tmp_path):
    vlm_filter = make_ollama_filter(tmp_path)

    parsed = vlm_filter.parse_response(json.dumps({"label": "rejected"}))
    unknown = vlm_filter.parse_response(json.dumps({"label": "unsure"}))

    assert parsed["label"] == "rejected"
    assert parsed["parse_error"] is False
    assert unknown["label"] is None
    assert unknown["parse_error"] is True


@pytest.mark.parametrize("raw", ["not json", "{"])
def test_parse_response_marks_invalid_json_as_failure(raw, tmp_path):
    parsed = make_ollama_filter(tmp_path).parse_response(raw)

    assert parsed["label"] is None
    assert parsed["parse_error"] is True
    assert parsed["raw_response"] == raw


def test_filter_data_preserves_frames_and_adds_configured_label_columns(
    monkeypatch, tmp_path
):
    records = make_records(
        tmp_path,
        [(101, None, "kept.jpg"), (102, None, "rejected.jpg")],
    ).drop(columns="source")
    records.index = [7, 11]
    original = records.copy(deep=True)
    vlm_filter = make_ollama_filter(tmp_path)
    outcomes = {
        str(path): label
        for path, label in zip(records.image_path, ["kept", "rejected"])
    }
    monkeypatch.setattr(
        vlm_filter,
        "classify",
        lambda path: {"label": outcomes[str(path)], "parse_error": False},
    )

    result = vlm_filter.filter_data({"mouse": records})

    assert list(result) == ["mouse"]
    pd.testing.assert_frame_equal(records, original)
    pd.testing.assert_index_equal(result["mouse"].index, original.index)
    assert result["mouse"]["kept_animal"].tolist() == [True, False]
    assert result["mouse"]["rejected_animal"].tolist() == [False, True]
    pd.testing.assert_frame_equal(
        result["mouse"].drop(columns=["kept_animal", "rejected_animal"]), original
    )


def test_duplicate_photo_is_classified_once_and_result_is_repeated(
    monkeypatch, tmp_path
):
    image_path = tmp_path / "same.jpg"
    image_path.write_bytes(b"image-bytes")
    records = pd.DataFrame(
        {
            "photo.id": [55, 55, 56],
            "image_path": [str(image_path), str(image_path), str(image_path)],
        },
        index=[2, 4, 8],
    )
    vlm_filter = make_ollama_filter(tmp_path)
    calls = []

    def classify(path):
        calls.append(Path(path))
        return {"label": "kept", "parse_error": False}

    monkeypatch.setattr(vlm_filter, "classify", classify)

    result = vlm_filter.filter_data({"mouse": records})["mouse"]

    assert len(calls) == 2
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
        vlm_filter = make_ollama_filter(tmp_path)
        calls = []

        def classify(path):
            calls.append(Path(path))
            label = "kept" if int(Path(path).stem) % 2 else "rejected"
            return {"label": label, "parse_error": False}

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(vlm_filter, "classify", classify)
            result = vlm_filter.filter_data({"mouse": records})["mouse"]

        assert len(calls) == len(set(photo_ids))
        for photo_id, row in zip(photo_ids, result.to_dict("records"), strict=True):
            expected_label = "kept" if photo_id % 2 else "rejected"
            assert row[f"{expected_label}_animal"] is True
            other_label = "rejected" if expected_label == "kept" else "kept"
            assert row[f"{other_label}_animal"] is False


def test_same_photo_id_in_different_species_is_processed_independently(
    monkeypatch, tmp_path
):
    mouse = make_records(tmp_path, [(55, None, "mouse.jpg")])
    rat = make_records(tmp_path, [(55, None, "rat.jpg")])
    vlm_filter = make_ollama_filter(tmp_path)
    calls = []
    outcomes = {mouse.image_path.iloc[0]: "kept", rat.image_path.iloc[0]: "rejected"}

    def classify(path):
        calls.append(Path(path))
        return {"label": outcomes[str(path)], "parse_error": False}

    monkeypatch.setattr(vlm_filter, "classify", classify)

    result = vlm_filter.filter_data({"mouse": mouse, "rat": rat})

    assert len(calls) == 2
    assert result["mouse"]["kept_animal"].tolist() == [True]
    assert result["rat"]["rejected_animal"].tolist() == [True]


def test_duplicate_photo_id_with_different_paths_fails_before_inference(
    monkeypatch, tmp_path
):
    records = make_records(
        tmp_path,
        [(55, None, "one.jpg"), (55, None, "two.jpg")],
    )
    vlm_filter = make_ollama_filter(tmp_path)
    calls = []
    monkeypatch.setattr(vlm_filter, "classify", lambda path: calls.append(path))

    with pytest.raises(ValueError, match="photo.id.*image_path"):
        vlm_filter.filter_data({"mouse": records})

    assert calls == []


def test_null_photo_id_fails_explicitly(monkeypatch, tmp_path):
    records = make_records(tmp_path, [(None, None, "missing-id.jpg")])
    vlm_filter = make_ollama_filter(tmp_path)
    calls = []
    monkeypatch.setattr(vlm_filter, "classify", lambda path: calls.append(path))

    with pytest.raises(ValueError, match="photo.id"):
        vlm_filter.filter_data({"mouse": records})

    assert calls == []


def test_inference_failure_sets_all_task_columns_to_nan(monkeypatch, tmp_path):
    records = make_records(tmp_path, [(55, None, "bad-response.jpg")])
    vlm_filter = make_ollama_filter(tmp_path)
    monkeypatch.setattr(
        vlm_filter,
        "classify",
        lambda path: {"label": None, "parse_error": True},
    )

    result = vlm_filter.filter_data({"mouse": records})["mouse"]

    assert pd.isna(result.loc[0, "kept_animal"])
    assert pd.isna(result.loc[0, "rejected_animal"])


@pytest.mark.parametrize("existing_column", ["kept_animal", "rejected_animal"])
def test_existing_task_columns_raise_before_inference(
    monkeypatch, tmp_path, existing_column
):
    records = make_records(tmp_path, [(55, None, "existing.jpg")])
    records[existing_column] = True
    vlm_filter = make_ollama_filter(tmp_path)
    calls = []
    monkeypatch.setattr(vlm_filter, "classify", lambda path: calls.append(path))

    with pytest.raises(ValueError, match=existing_column):
        vlm_filter.filter_data({"mouse": records})

    assert calls == []
    assert records[existing_column].tolist() == [True]


def test_empty_species_frame_is_preserved_with_task_columns(tmp_path):
    records = pd.DataFrame(columns=["photo.id", "image_path"])
    vlm_filter = make_ollama_filter(tmp_path)

    result = vlm_filter.filter_data({"empty species": records})["empty species"]

    assert result.empty
    assert "kept_animal" in result.columns
    assert "rejected_animal" in result.columns


def test_vlm_filter_rejects_invalid_label_configuration(tmp_path):
    with pytest.raises(ValueError, match="labels"):
        FilterOllama(
            prompt="prompt",
            system_prompt="system",
            labels=["kept", "kept"],
            taskname="animal",
            photo_id_column="photo.id",
            image_path_column="image_path",
            model="llava",
        )


def test_ollama_classify_uses_persistent_client(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")
    vlm_filter = make_ollama_filter(tmp_path)

    parsed = vlm_filter.classify(image_path)

    assert parsed["label"] == "kept"
    assert vlm_filter.client.host is None
    assert vlm_filter.client.pull_calls == [{"model": "llava", "stream": False}]
    assert vlm_filter.client.generate_calls[0] == {
        "model": "llava",
        "system": "system",
        "prompt": "prompt",
        "images": ["YWJj"],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }


def test_ollama_closes_after_filtering(monkeypatch, tmp_path):
    records = make_records(tmp_path, [(55, None, "image.jpg")])
    vlm_filter = make_ollama_filter(tmp_path)
    monkeypatch.setattr(
        vlm_filter, "classify", lambda path: {"label": "kept", "parse_error": False}
    )

    vlm_filter.filter_data({"mouse": records})

    assert vlm_filter.client.generate_calls[-1] == {
        "model": "llava",
        "keep_alive": 0,
    }


def test_vllm_schema_uses_configured_labels(tmp_path):
    vlm_filter = make_vllm_filter(tmp_path)

    assert vlm_filter.response_json_schema["properties"]["label"]["enum"] == [
        "kept",
        "rejected",
    ]


def test_vllm_image_to_data_url_uses_mime_suffix(tmp_path):
    jpg = tmp_path / "a.JPG"
    png = tmp_path / "b.png"
    jpg.write_bytes(b"jpg")
    png.write_bytes(b"png")

    assert FilterVLLM.image_to_data_url(jpg) == "data:image/jpeg;base64,anBn"
    assert FilterVLLM.image_to_data_url(png) == "data:image/png;base64,cG5n"


def test_vllm_filter_data_batches_unique_photos_and_preserves_files(
    monkeypatch, tmp_path
):
    path1 = tmp_path / "a.jpg"
    path2 = tmp_path / "b.jpg"
    path1.write_bytes(b"image-a")
    path2.write_bytes(b"image-b")
    records = pd.DataFrame(
        {
            "photo.id": [1, 1, 2],
            "image_path": [str(path1), str(path1), str(path2)],
        }
    )
    vlm_filter = make_vllm_filter(tmp_path, batch_size=2)
    seen_chunks = []

    def classify_batch(paths):
        seen_chunks.append(list(paths))
        return [
            {
                "label": "kept" if Path(path) == path1 else "rejected",
                "parse_error": False,
            }
            for path in paths
        ]

    monkeypatch.setattr(vlm_filter, "_classify_batch", classify_batch)

    result = vlm_filter.filter_data({"mouse": records})["mouse"]

    assert seen_chunks == [[path1, path2]]
    assert result["kept_animal"].tolist() == [True, True, False]
    assert result["rejected_animal"].tolist() == [False, False, True]
    assert path1.read_bytes() == b"image-a"
    assert path2.read_bytes() == b"image-b"


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


def test_from_config_passes_filter_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": "ollama",
                "prompt": "prompt",
                "system_prompt": "system",
                "taskname": "animal",
                "labels": ["kept", "rejected"],
                "columns": {"photo_id": "photo.id", "image_path": "image_path"},
                "paths": {},
                "ollama": {"model": "llava", "pull_model": False},
            }
        )
    )

    vlm_filter = VLMFilter.from_config(config_path)

    assert isinstance(vlm_filter, FilterOllama)
    assert vlm_filter.labels == ["kept", "rejected"]
    assert vlm_filter.taskname == "animal"
    assert vlm_filter.photo_id_column == "photo.id"
    assert vlm_filter.image_path_column == "image_path"
