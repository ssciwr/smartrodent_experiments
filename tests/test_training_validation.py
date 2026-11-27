"""Shared training settings: schema validation, serialization and input protection.

Covers YAML and ``dataclasses.replace`` round trips, read-only mapping inputs,
pass-through of native optimizer and metric options, ordered class names, and the
guard that keeps run outputs from overwriting datasets, checkpoints and templates.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from smartrodent.classifier_training import ClassifierConfiguration
from smartrodent.experiment_support import (
    EarlyStopping,
    MlflowConfiguration,
    SchedulerConfiguration,
)
from smartrodent.supcon_training import TrainingConfiguration


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
        field, settings = (
            "train_kwargs",
            {
                "optimizer": "AdamW",
                "lr0": 0.001,
                "cos_lr": True,
            },
        )
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


def test_class_names_must_keep_their_declared_order(configuration):
    """A set would make class indices differ between runs, so it is rejected."""
    document = configuration.to_dict()
    document["class_names"] = {"rat", "mouse"}
    with pytest.raises(ValueError, match="class_names"):
        type(configuration).from_mapping(document)
