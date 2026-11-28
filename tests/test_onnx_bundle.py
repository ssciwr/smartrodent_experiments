"""Bundle export of the registered classifier for the sentinel-rat-ml-pipeline contract."""

import filecmp
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
import yaml
from classifier_task_test_support import (
    CLASS_NAMES,
    IMAGE_SIZE,
    classifier_document,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

from smartrodent.classifier_task import ClassifierTask
from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.onnx_bundle import (
    TAXON_FIELDS,
    OnnxBundleError,
    OnnxBundleExporter,
    order_classes,
)
from smartrodent.stage_configurations import ClassifierConfiguration
from smartrodent.transforms import ImageTransforms, TransformSettings

TAXONOMY = {
    "mouse": {"genus": "Mus", "species": "musculus", "common_name": "house mouse"},
    "rat": {"genus": "Rattus", "species": "rattus", "common_name": "black rat"},
    "shrew": {"genus": "Sorex", "common_name": "shrew"},
}
ONNX = {"dynamic": True, "simplify": False}


def _train(tmp_path, store, *, onnx=ONNX, **overrides):
    """Train and register a tiny classifier; return the result."""
    document = classifier_document(
        tmp_path, mlflow=store.configuration, onnx_export=onnx, **overrides
    )
    experiment = ClassifierExperiment(ClassifierConfiguration.from_mapping(document))
    return experiment.run_with_result()


def _exporter(store, tmp_path, *, taxonomy=None, **options):
    """Build an exporter for the tiny registry."""
    return OnnxBundleExporter(
        tracking_uri=store.configuration["tracking_uri"],
        registered_model_name=store.registry_name,
        alias=store.configuration["champion_alias"],
        output_directory=tmp_path / "bundle",
        taxonomy=TAXONOMY if taxonomy is None else taxonomy,
        **options,
    )


def _champion_version(store) -> str:
    """Read the version behind the configured alias."""
    alias = store.configuration["champion_alias"]
    return str(
        store.client.get_model_version_by_alias(store.registry_name, alias).version
    )


def _replace_artifact(store, result, filename: str, content: bytes) -> None:
    """Overwrite a file of the run's ``best_model/`` in the local artifact store."""
    run = store.client.get_run(result.mlflow_run_id)
    root = Path(run.info.artifact_uri.removeprefix("file://"))
    (root / "best_model" / filename).write_bytes(content)


def _manifest(bundle: Path) -> dict:
    return yaml.safe_load((bundle / "config.yaml").read_text())


def _rebuild(preprocessing) -> v2.Compose:
    """Rebuild torchvision v2 transforms from manifest entries, as a consumer would."""
    steps = []
    for entry in preprocessing:
        args = [_revive(value) for value in entry.get("args", [])]
        kwargs = {k: _revive(v) for k, v in entry.get("kwargs", {}).items()}
        steps.append(getattr(v2, entry["name"])(*args, **kwargs))
    return v2.Compose(steps)


def _revive(value):
    """Turn dtype and interpolation names back into objects."""
    if value in ("float32", "uint8"):
        return getattr(torch, value)
    if value in [mode.value for mode in InterpolationMode]:
        return InterpolationMode(value)
    return value


def _image() -> Image.Image:
    generator = np.random.default_rng(3)
    pixels = generator.integers(0, 255, (13, 17, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_bundle_has_exactly_three_files_copied_from_the_registered_run(
    tmp_path, mlflow_store
):
    result = _train(tmp_path, mlflow_store)

    bundle = _exporter(mlflow_store, tmp_path).export()

    assert {p.name for p in bundle.iterdir()} == {
        "config.yaml",
        "model.onnx",
        "README.md",
    }
    assert filecmp.cmp(bundle / "model.onnx", result.onnx_model, shallow=False)
    card = result.best_checkpoint.with_suffix(".model_card.md")
    assert (bundle / "README.md").read_text() == card.read_text()


def test_registered_run_carries_its_config_in_mlflow(tmp_path, mlflow_store):
    result = _train(tmp_path, mlflow_store)

    path = mlflow_store.client.download_artifacts(
        result.mlflow_run_id, "best_model/config.yaml", str(tmp_path / "dl")
    )

    assert yaml.safe_load(Path(path).read_text()) == yaml.safe_load(
        (result.output_directory / "config.yaml").read_text()
    )


def test_alias_resolves_to_its_version_and_explicit_version_overrides_it(
    tmp_path, mlflow_store
):
    first = _train(tmp_path / "a", mlflow_store).registered_model_version
    # Promotion (classifier_champions.py) keeps run 1 as champion. A second run is
    # not trained: any other version is reached by number instead.
    assert _champion_version(mlflow_store) == first
    client = mlflow_store.client
    source_version = client.get_model_version(mlflow_store.registry_name, first)
    second = str(
        client.create_model_version(
            mlflow_store.registry_name,
            source_version.source,
            run_id=source_version.run_id,
        ).version
    )

    by_alias = _exporter(mlflow_store, tmp_path / "x").export()
    by_version = _exporter(mlflow_store, tmp_path / "y", version=second).export()

    assert _manifest(by_alias)["info"] == {
        "name": mlflow_store.registry_name,
        "version": first,
    }
    assert _manifest(by_version)["info"]["version"] == second


def test_classes_follow_output_order_with_taxon_keys_only(tmp_path, mlflow_store):
    _train(tmp_path, mlflow_store)

    manifest = _manifest(_exporter(mlflow_store, tmp_path).export())

    assert manifest["classes"] == [TAXONOMY[name] for name in CLASS_NAMES]
    for taxon in manifest["classes"]:
        assert set(taxon) <= TAXON_FIELDS


@settings(max_examples=25, deadline=None)
@given(
    labels=st.lists(
        st.text("abcdefgh", min_size=1, max_size=4), min_size=1, max_size=6, unique=True
    ),
    data=st.data(),
)
def test_class_order_ignores_mapping_order(labels, data):
    names = {index: label for index, label in enumerate(labels)}
    shuffled = data.draw(st.permutations(labels))
    taxonomy = {label: {"common_name": label} for label in shuffled}

    assert order_classes(names, taxonomy) == [{"common_name": l} for l in labels]


@pytest.mark.parametrize(
    "taxonomy",
    [
        {k: v for k, v in TAXONOMY.items() if k != "rat"},
        {**TAXONOMY, "badger": {"genus": "Meles"}},
        {**TAXONOMY, "rat": {"genus": "Rattus", "colour": "black"}},
    ],
    ids=["missing", "extra", "unknown-key"],
)
def test_bad_taxonomy_fails_before_anything_is_written(
    tmp_path, mlflow_store, taxonomy
):
    _train(tmp_path, mlflow_store)

    with pytest.raises(OnnxBundleError, match="taxonomy|Taxon"):
        _exporter(mlflow_store, tmp_path, taxonomy=taxonomy).export()

    assert not (tmp_path / "bundle").exists()


def test_manifest_matches_the_graph_and_dynamic_axes_become_null(
    tmp_path, mlflow_store
):
    _train(tmp_path, mlflow_store)
    bundle = _exporter(mlflow_store, tmp_path).export()
    manifest = _manifest(bundle)
    graph = onnx.load(str(bundle / "model.onnx")).graph

    assert manifest["schema_version"] == 1
    assert manifest["role"] == "classifier"
    assert manifest["model_file"] == "model.onnx"
    assert manifest["input"] == {
        "name": graph.input[0].name,
        "dtype": "float32",
        "shape": [None, 3, None, None],
    }
    assert manifest["output"] == {
        "name": graph.output[0].name,
        "dtype": "float32",
        "shape": [None, len(CLASS_NAMES)],
        "score_semantics": "probabilities",
    }
    assert manifest["calibration"] == {"method": "none"}
    (opset,) = onnx.load(str(bundle / "model.onnx")).opset_import
    assert opset.domain in ("", "ai.onnx")
    assert manifest["opsets"] == {"ai.onnx": opset.version}


def test_static_shapes_are_written_as_integers(tmp_path, mlflow_store):
    _train(tmp_path, mlflow_store, onnx={"dynamic": False, "simplify": False})

    manifest = _manifest(_exporter(mlflow_store, tmp_path).export())

    assert manifest["input"]["shape"] == [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    assert manifest["output"]["shape"] == [1, len(CLASS_NAMES)]


@pytest.mark.parametrize(
    "resize",
    [{}, {"interpolation": "bicubic", "antialias": False}],
    ids=["default", "bicubic"],
)
def test_manifest_preprocessing_reproduces_the_run_eval_transform(
    tmp_path, mlflow_store, resize
):
    _train(
        tmp_path,
        mlflow_store,
        transforms={"image_size": IMAGE_SIZE, "resize": resize},
    )
    manifest = _manifest(_exporter(mlflow_store, tmp_path).export())
    expected = ImageTransforms(
        TransformSettings(image_size=IMAGE_SIZE, resize=resize)
    ).validation()

    rebuilt = _rebuild(manifest["preprocessing"])

    assert torch.equal(rebuilt(_image()), expected(_image()))


@pytest.mark.parametrize("simplify", [False, True], ids=["raw", "onnxslim"])
def test_exported_graph_obeys_the_contract_graph_rules(
    tmp_path, mlflow_store, simplify
):
    _train(tmp_path, mlflow_store, onnx={"dynamic": True, "simplify": simplify})
    bundle = _exporter(mlflow_store, tmp_path).export()

    model = onnx.load(str(bundle / "model.onnx"), load_external_data=False)

    onnx.checker.check_model(model)
    assert {i.domain or "ai.onnx" for i in model.opset_import} == {"ai.onnx"}
    assert len(model.opset_import) == 1
    assert len(model.functions) == 0
    assert all((n.domain or "ai.onnx") == "ai.onnx" for n in model.graph.node)
    assert not any(
        t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer
    )


def test_registered_run_without_config_fails_clearly(tmp_path, mlflow_store):
    result = _train(tmp_path, mlflow_store)
    # Simulate an older run by removing the logged config from the artifact store.
    run = mlflow_store.client.get_run(result.mlflow_run_id)
    root = Path(run.info.artifact_uri.removeprefix("file://"))
    (root / "best_model" / "config.yaml").unlink()

    with pytest.raises(OnnxBundleError, match="config.yaml"):
        _exporter(mlflow_store, tmp_path).export()

    assert not (tmp_path / "bundle").exists()


def test_missing_model_alias_or_version_fails_clearly(tmp_path, mlflow_store):
    with pytest.raises(OnnxBundleError, match="not found|does not exist"):
        _exporter(mlflow_store, tmp_path).export()
    _train(tmp_path, mlflow_store)
    with pytest.raises(OnnxBundleError, match="alias"):
        OnnxBundleExporter(
            tracking_uri=mlflow_store.configuration["tracking_uri"],
            registered_model_name=mlflow_store.registry_name,
            alias="nonexistent",
            output_directory=tmp_path / "bundle",
            taxonomy=TAXONOMY,
        ).export()
    with pytest.raises(OnnxBundleError, match="version"):
        _exporter(mlflow_store, tmp_path, version="99").export()


def test_run_without_onnx_file_fails_clearly(tmp_path, mlflow_store):
    _train(tmp_path, mlflow_store, onnx=None)

    with pytest.raises(OnnxBundleError, match="ONNX"):
        _exporter(mlflow_store, tmp_path).export()


def test_non_empty_output_directory_is_refused(tmp_path, mlflow_store):
    _train(tmp_path, mlflow_store)
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    with pytest.raises(OnnxBundleError, match="not empty"):
        _exporter(mlflow_store, tmp_path).export()

    assert (target / "keep.txt").read_text() == "x"


def test_runtime_on_the_bundle_matches_the_lightning_classifier(tmp_path, mlflow_store):
    result = _train(tmp_path, mlflow_store)
    bundle = _exporter(mlflow_store, tmp_path).export()
    manifest = _manifest(bundle)
    tensor = _rebuild(manifest["preprocessing"])(_image()).unsqueeze(0)

    session = onnxruntime.InferenceSession(
        str(bundle / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    (probabilities,) = session.run(None, {manifest["input"]["name"]: tensor.numpy()})

    task = ClassifierTask.load_from_checkpoint(
        result.best_checkpoint, map_location="cpu"
    ).eval()
    with torch.no_grad():
        expected = task(tensor).softmax(dim=1).numpy()
    np.testing.assert_allclose(probabilities, expected, rtol=1e-4, atol=1e-5)


def test_version_without_a_source_run_fails_clearly(tmp_path, mlflow_store):
    client = mlflow_store.client
    client.create_registered_model(mlflow_store.registry_name)
    version = client.create_model_version(
        mlflow_store.registry_name, (tmp_path / "src").as_uri()
    )

    with pytest.raises(OnnxBundleError, match="no source run"):
        _exporter(mlflow_store, tmp_path, version=version.version).export()


def test_explicit_none_is_written_when_the_default_is_not_none(tmp_path, mlflow_store):
    _train(
        tmp_path,
        mlflow_store,
        transforms={"image_size": IMAGE_SIZE, "resize": {"antialias": None}},
    )

    manifest = _manifest(_exporter(mlflow_store, tmp_path).export())

    assert manifest["preprocessing"][-1]["kwargs"]["antialias"] is None


def _model(*, inputs=1, outputs=1, elem_type=onnx.TensorProto.FLOAT, names="{0: 'a'}"):
    """Build a bare ModelProto with the given tensors and ``names`` metadata."""
    helper = onnx.helper
    graph = helper.make_graph(
        [],
        "g",
        [
            helper.make_tensor_value_info(f"in{i}", elem_type, [1, 3])
            for i in range(inputs)
        ],
        [
            helper.make_tensor_value_info(f"out{i}", elem_type, [1, 1])
            for i in range(outputs)
        ],
    )
    model = helper.make_model(graph)
    if names is not None:
        model.metadata_props.add(key="names", value=names)
    return model


@pytest.mark.parametrize(
    "model,message",
    [
        (_model(names=None), "names"),
        (_model(names="not python {"), "names"),
        (_model(names="[1, 2]"), "names"),
        (_model(inputs=2), "exactly one"),
        (_model(outputs=0), "exactly one"),
        (_model(elem_type=onnx.TensorProto.DOUBLE), "float32"),
    ],
    ids=[
        "no-names",
        "bad-names",
        "names-not-a-map",
        "two-inputs",
        "no-output",
        "not-float32",
    ],
)
def test_malformed_graphs_are_rejected(tmp_path, mlflow_store, model, message):
    result = _train(tmp_path, mlflow_store)
    _replace_artifact(mlflow_store, result, "best.onnx", model.SerializeToString())

    with pytest.raises(OnnxBundleError, match=message):
        _exporter(mlflow_store, tmp_path).export()

    assert not (tmp_path / "bundle").exists()


def test_a_transform_with_an_unreadable_parameter_is_rejected(
    tmp_path, mlflow_store, monkeypatch
):
    class Odd(v2.Transform):
        def __init__(self, hidden):
            super().__init__()

    class Stub:
        def __init__(self, settings):
            pass

        def validation(self):
            return v2.Compose([Odd(1)])

    _train(tmp_path, mlflow_store)
    monkeypatch.setattr("smartrodent.onnx_bundle.ImageTransforms", Stub)

    with pytest.raises(OnnxBundleError, match="hidden"):
        _exporter(mlflow_store, tmp_path).export()
