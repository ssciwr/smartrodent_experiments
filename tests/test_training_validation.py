"""Compatibility and consistency contracts for the new training configuration layer.

Legacy YOLO trainers are deliberately outside this refactor. Existing integration
suites cover training, sweeps, checkpoint safety, and champion publication; these
checks target schema gaps and the Python/YAML interface that must survive.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest
import yaml

from smartrodent.supcon_training import TrainingConfiguration
from smartrodent.classifier_training import ClassifierConfiguration
from smartrodent.experiment_support import (
    EarlyStopping,
    MlflowConfiguration,
    SchedulerConfiguration,
)


@pytest.fixture(params=["supcon", "classifier"])
def configuration(request, tmp_path):
    """Validate settings without requiring images, model weights, or output directories."""
    values = {
        "train_dataset": str(tmp_path / "train"),
        "output_directory": str(tmp_path / "outputs"),
        "class_names": ["rat", "mouse"],
        "workers": 0,
        "device": "cpu",
        "optimizer": {"lr": 0.001},
        "transforms": {
            "image_size": 16,
            "resize": {"antialias": True},
            "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
            "random_grayscale": {"p": 0.0},
            "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
            "random_horizontal_flip": {"p": 0.0},
            "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
        },
    }
    if request.param == "supcon":
        values.update(
            model_name=str(tmp_path / "model.yaml"),
            classes_per_batch=2,
            samples_per_class=2,
        )
        return TrainingConfiguration.from_mapping(values)
    elif request.param == "classifier":
        values.pop("optimizer")
        values["train_kwargs"] = {"optimizer": "AdamW", "lr0": 0.001}
        values["initialization"] = {
            "encoder": {"source": "supcon", "checkpoint": str(tmp_path / "encoder.pt")},
            "head": {"replace": "linear", "source": "random"},
        }
        values["freeze"] = 0
        return ClassifierConfiguration.from_mapping(values)
    else:
        raise AssertionError(f"Unknown training system: {request.param}")


def _parse_update(configuration, path, value):
    """Change one input only, so a failure cannot be explained by another missing field."""
    document = deepcopy(configuration.to_dict())
    if path == ("learning_rate",):
        path = (
            ("optimizer", "lr")
            if isinstance(configuration, TrainingConfiguration)
            else ("train_kwargs", "lr0")
        )
    node = document
    for name in path[:-1]:
        node = node[name]
    node[path[-1]] = value
    return type(configuration).from_mapping(document)


@pytest.mark.parametrize(
    "path,value",
    [
        (("epochs",), True),
        (("epochs",), "2"),
        (("epochs",), 1.5),
        (("workers",), True),
        (("workers",), "0"),
        (("workers",), 0.5),
        (("seed",), True),
        (("seed",), -1),
        (("seed",), 2**32),
        (("seed",), "42"),
        (("seed",), 0.5),
        (("learning_rate",), True),
        (("learning_rate",), "0.001"),
        (("learning_rate",), 0),
        (("learning_rate",), float("nan")),
        (("learning_rate",), float("inf")),
        (("transforms", "image_size"), True),
        (("transforms", "image_size"), 0),
    ],
)
def test_new_training_systems_reject_invalid_scalar_settings_consistently(
    configuration, path, value
):
    """Reject coercible strings, bool-as-number inputs, and invalid ranges before training."""
    with pytest.raises((ValueError, TypeError)):
        _parse_update(configuration, path, value)
    assert not configuration.output_directory.exists()


@pytest.mark.parametrize("collection", [set, iter])
def test_class_names_require_a_sequence_not_an_unordered_or_one_shot_iterable(
    configuration, collection
):
    """Schema coercion must not broaden the existing ordered class-name contract."""
    with pytest.raises((ValueError, TypeError), match="class_names"):
        _parse_update(configuration, ("class_names",), collection(["rat", "mouse"]))


@pytest.mark.parametrize("seed", [True, -1, 2**32, "42", 0.5])
def test_invalid_seeds_are_rejected_through_dataclass_replace_too(configuration, seed):
    """Direct Python callers must not bypass constraints enforced by mapping loaders."""
    with pytest.raises((ValueError, TypeError)) as error:
        replace(configuration, seed=seed)
    assert "seed" in str(error.value)


@pytest.mark.parametrize("configuration", ["supcon"], indirect=True)
@pytest.mark.parametrize(
    "field,value", [("temperature", True), ("dropout", True), ("freeze", [True])]
)
def test_supcon_only_numeric_settings_also_exclude_booleans(
    configuration, field, value
):
    """Algorithm-specific settings use the same strict numeric conventions."""
    with pytest.raises((ValueError, TypeError)):
        _parse_update(configuration, (field,), value)


@pytest.mark.parametrize("missing", ["resize", "random_resized_crop"])
def test_missing_transform_sections_produce_configuration_errors_not_key_errors(
    configuration, missing
):
    """Nested input shape is checked declaratively rather than failing inside a builder."""
    transforms = deepcopy(configuration.transforms)
    del transforms[missing]
    with pytest.raises((ValueError, TypeError)) as error:
        replace(configuration, transforms=transforms)
    assert "transforms" in str(error.value) or missing in str(error.value)


@pytest.mark.parametrize(
    "path",
    [
        ("unknown_option",),
        ("mlflow", "unknown_option"),
        ("scheduler", "unknown_option"),
    ],
)
def test_unknown_configuration_keys_remain_errors(configuration, path):
    """Schema migration must not silently discard typos at the root or shared boundaries."""
    options = {
        "validation_dataset": configuration.train_dataset,
        "mlflow": MlflowConfiguration(
            tracking_uri="sqlite:///unused-validation-test.db",
            experiment_name="validation-test",
        ),
    }
    if path[0] == "scheduler":
        if isinstance(configuration, ClassifierConfiguration):
            path = ("train_kwargs", path[-1])
        else:
            options["scheduler"] = SchedulerConfiguration(
                "ReduceLROnPlateau", {"mode": "min"}
            )
    configured = replace(configuration, **options)
    with pytest.raises((ValueError, TypeError, SyntaxError)) as error:
        _parse_update(configured, path, 1)
    assert path[-1] in str(error.value)


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(0, 2**32 - 1),
    workers=st.integers(0, 8),
    lr=st.floats(min_value=0.0001, max_value=0.1, allow_nan=False),
)
def test_valid_settings_keep_dataclass_replace_and_yaml_round_trip(
    configuration, tmp_path, seed, workers, lr
):
    """Preserve immutable configuration callers, meaningful numeric values, and portable paths."""
    field, rate_key = (
        ("optimizer", "lr")
        if isinstance(configuration, TrainingConfiguration)
        else ("train_kwargs", "lr0")
    )
    options = {**getattr(configuration, field), rate_key: lr}
    updated = replace(configuration, seed=seed, workers=workers, **{field: options})
    document = updated.to_dict()
    assert isinstance(updated.train_dataset, Path)
    assert document["seed"] == seed and document["workers"] == workers
    assert document[field][rate_key] == lr
    assert document["class_names"] == ["rat", "mouse"]
    assert isinstance(document["train_dataset"], str)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(document))
    assert type(updated).from_mapping(document).to_dict() == document
    assert type(updated).from_yaml(path).to_dict() == document
    assert configuration.seed == 42 and configuration.workers == 0
    assert not configuration.output_directory.exists()


def test_readonly_mapping_inputs_remain_supported_and_independently_owned(
    configuration,
):
    """Public Mapping inputs need not be pickleable, and mutable children remain caller-owned."""
    document = configuration.to_dict()
    expected = deepcopy(document)
    parsed = type(configuration).from_mapping(MappingProxyType(document))
    document["transforms"]["random_resized_crop"]["scale"][0] = 0.5
    assert parsed.to_dict() == expected


@pytest.mark.parametrize(
    "schema,document,attribute,expected",
    [
        (
            MlflowConfiguration,
            {"tracking_uri": "sqlite:///unused.db", "experiment_name": "readonly"},
            "experiment_name",
            "readonly",
        ),
        (
            SchedulerConfiguration,
            {
                "name": "ReduceLROnPlateau",
                "parameters": MappingProxyType({"mode": "max"}),
            },
            "mode",
            "max",
        ),
        (
            EarlyStopping,
            {"patience": 3, "tolerance": 0.01, "comparison": "less"},
            "patience",
            3,
        ),
    ],
)
def test_shared_setting_loaders_accept_readonly_mappings(
    schema, document, attribute, expected
):
    """MLflow, scheduler and stopping loaders retain their Mapping contracts too."""
    parsed = schema.from_mapping(MappingProxyType(document))
    assert getattr(parsed, attribute) == expected


@pytest.mark.parametrize(
    "field,settings",
    [
        (
            "mlflow",
            {"tracking_uri": "sqlite:///unused.db", "experiment_name": "nested"},
        ),
        ("scheduler", {"name": "ReduceLROnPlateau", "parameters": {"mode": "min"}}),
    ],
)
def test_nested_configuration_mappings_accept_readonly_inputs(
    configuration, field, settings
):
    """Shared dataclass fields retain the Mapping contract of their standalone loaders."""
    if field == "scheduler" and isinstance(configuration, ClassifierConfiguration):
        field, settings = "train_kwargs", {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
        }
    document = configuration.to_dict()
    document.update(validation_dataset=document["train_dataset"], **{field: settings})
    expected = type(configuration).from_mapping(document).to_dict()
    document[field] = MappingProxyType(settings)
    assert type(configuration).from_mapping(document).to_dict() == expected


def test_native_optimizer_and_metric_options_are_not_lost_by_schema_validation(
    configuration, tmp_path
):
    """Do not replace native option dictionaries with a restricted project-specific subset."""
    optimizer = {
        "lr": 0.001,
        "betas": [0.8, 0.95],
        "eps": 1e-6,
        "weight_decay": 0.02,
        "amsgrad": True,
        "maximize": False,
        "foreach": False,
        "fused": False,
    }
    metrics = {
        "macro_f1": {
            "function": "f1_score",
            "average": "macro",
            "labels": [0, 1],
            "zero_division": 0,
        }
    }
    field = "optimizer"
    if isinstance(configuration, ClassifierConfiguration):
        field = "train_kwargs"
        optimizer = {
            "optimizer": "AdamW",
            "lr0": 0.001,
            "momentum": 0.8,
            "weight_decay": 0.02,
            "cos_lr": True,
            "lrf": 0.01,
            "patience": 3,
            "warmup_epochs": 0.0,
            "amp": False,
        }
    expected_optimizer, expected_metrics = deepcopy(optimizer), deepcopy(metrics)
    updated = replace(
        configuration,
        **{field: optimizer},
        validation_dataset=configuration.train_dataset,
        validation_metrics=metrics,
        tracked_metric="macro_f1",
        tracked_mode="max",
        model_card_template=Path("configs/model_card_template.yaml"),
    )
    document = updated.to_dict()
    path = tmp_path / "native-options.yaml"
    path.write_text(yaml.safe_dump(document))
    restored = [
        updated,
        type(updated).from_mapping(document),
        type(updated).from_yaml(path),
    ]
    for result in restored:
        assert result.to_dict()[field] == expected_optimizer
        assert result.to_dict()["validation_metrics"] == expected_metrics
    assert optimizer == expected_optimizer and metrics == expected_metrics


@pytest.mark.parametrize("configuration", ["classifier"], indirect=True)
@pytest.mark.parametrize("source", ["encoder", "head"])
@pytest.mark.parametrize("checkpoint", ["", "  "])
def test_checkpoint_strings_cannot_bypass_nonblank_validation(
    configuration, source, checkpoint
):
    """A blank string must not fall through a path union and become the current directory."""
    document = configuration.to_dict()
    document["initialization"][source].update(
        source="yolo_checkpoint" if source == "head" else "supcon",
        checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        type(configuration).from_mapping(document)


@pytest.mark.parametrize("source", ["validation_dataset", "checkpoint"])
def test_each_standalone_configuration_protects_all_its_inputs(configuration, source):
    """A shared safety implementation must protect both standalone APIs, not only the pipeline."""
    if source == "validation_dataset":
        changes = {"validation_dataset": configuration.train_dataset.parent / "val"}
        changes["output_directory"] = changes["validation_dataset"] / "outputs"
    elif source == "checkpoint":
        relative = (
            "best.pt"
            if isinstance(configuration, TrainingConfiguration)
            else "weights/best.pt"
        )
        protected = configuration.output_directory / relative
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b"protected source checkpoint")
        if isinstance(configuration, TrainingConfiguration):
            changes = {"model_name": str(protected)}
        else:
            initialization = deepcopy(configuration.initialization)
            initialization["encoder"]["checkpoint"] = str(protected)
            changes = {"initialization": initialization}
    else:
        raise AssertionError(f"Unknown source: {source}")
    with pytest.raises(ValueError, match="outside|collid|source|checkpoint"):
        replace(configuration, **changes)
    if source == "checkpoint":
        assert protected.read_bytes() == b"protected source checkpoint"
