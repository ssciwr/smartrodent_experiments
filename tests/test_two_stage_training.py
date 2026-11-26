"""Contracts for unified SupCon-sweep → selected encoder → CE orchestration."""

import hashlib
from copy import deepcopy
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from mlflow.exceptions import MlflowException
from training_test_support import (
    assert_registered_classifier,
    download_artifact,
    serialized_result,
    widened_architecture,
)
from ultralytics import YOLO

from smartrodent.experiment_support import MlflowConfiguration
from smartrodent.supcon_training import SupConExperiment
from smartrodent.two_stage_training import (
    TwoStageConfiguration,
    TwoStageExperiment,
    TwoStageResult,
)


def _configuration(document):
    """Parse the complete pipeline before constructing an experiment."""
    return TwoStageConfiguration.from_mapping(document)


@pytest.mark.parametrize(
    "mode,values,winner",
    [
        # ranking itself is tested with the result ranking; one winner and one tie
        ("max", [0.2, 0.8, 0.5], 1),
        ("min", [0.2, 0.2, 0.8], 0),
    ],
)
def test_pipeline_uses_configured_objective_and_passes_winning_best_not_last(
    pipeline_document, recorded_stages, mode, values, winner
):
    """No rescoring, hardcoded loss, last checkpoint, or unordered filesystem discovery."""
    document = pipeline_document
    document["supcon"]["tracked_mode"] = mode
    document["supcon"]["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "parameters": {"mode": mode, "patience": 2, "factor": 0.5},
    }
    document["supcon"]["early_stopping"] = {
        "patience": 5,
        "tolerance": 0.01,
        "comparison": "less" if mode == "min" else "greater",
    }
    document["classifier"]["train_kwargs"].update(patience=4, cos_lr=True, lrf=0.2)
    recorded_stages.values = values
    config = _configuration(document)
    before = deepcopy(config.to_dict())
    result = TwoStageExperiment(config).run()
    assert isinstance(result, TwoStageResult)
    assert recorded_stages.events == ["supcon", "supcon", "supcon", "classifier"]
    assert tuple(result.supcon_results) == tuple(recorded_stages.results)
    assert result.selected_supcon == result.supcon_results[winner]
    selected = result.selected_supcon
    classifier = recorded_stages.classifier[0]
    assert (
        selected.tracked_metric == "balanced_accuracy" and selected.tracked_mode == mode
    )
    assert selected.best_value == values[winner] and selected.best_epoch == 2
    assert classifier.initialization["encoder"]["source"] == "supcon"
    source = Path(classifier.initialization["encoder"]["checkpoint"])
    assert source.resolve() == selected.best_checkpoint.resolve()
    assert source != selected.last_checkpoint
    assert torch.load(source, weights_only=True) == {"candidate": winner, "epoch": 2}
    root = Path(document["output_directory"])
    assert result.output_directory == root
    assert [item.output_directory for item in result.supcon_results] == [
        root / "supcon" / f"run_{index:03d}" for index in (1, 2, 3)
    ]
    assert (
        classifier.output_directory
        == result.classifier_result.output_directory
        == root / "classifier"
    )
    assert result.classifier_result.tracked_metric == "ce_loss_mean"
    assert result.classifier_result.best_value == 0.4
    assert [item.optimizer["lr"] for item in recorded_stages.supcon] == [
        0.001,
        0.002,
        0.003,
    ]
    assert all(
        item.seed == 11 and item.temperature == 0.1 for item in recorded_stages.supcon
    )
    assert (
        classifier.seed == 29
        and classifier.train_kwargs == document["classifier"]["train_kwargs"]
    )
    assert classifier.batch_size == 3
    assert classifier.encoder_batchnorm == "eval"
    assert classifier.freeze == document["classifier"]["freeze"]
    assert (
        classifier.initialization["head"]
        == document["classifier"]["initialization"]["head"]
    )
    for stage in [*recorded_stages.supcon, classifier]:
        stage_settings = (
            document["classifier"] if stage is classifier else document["supcon"]
        )
        assert stage.transforms == stage_settings["transforms"]
        if stage is classifier:
            assert stage.train_kwargs == stage_settings["train_kwargs"]
        else:
            assert stage.scheduler.name == stage_settings["scheduler"]["name"]
            assert (
                stage.scheduler.parameters == stage_settings["scheduler"]["parameters"]
            )
            assert stage.early_stopping == stage_settings["early_stopping"]
        assert stage.train_dataset == Path(document["data"]["train_dataset"])
        assert stage.validation_dataset == Path(document["data"]["validation_dataset"])
        assert stage.class_names == ("rat", "mouse")
    assert config.to_dict() == before
    assert yaml.safe_load((root / "config.yaml").read_text()) == before
    summary = yaml.safe_load((root / "selection.yaml").read_text())
    for candidate, saved in zip(
        result.supcon_results, summary["supcon_results"], strict=True
    ):
        assert serialized_result(candidate).items() <= saved.items()
    assert serialized_result(selected).items() <= summary["selected_supcon"].items()
    assert (
        summary["selected_supcon"]["checkpoint_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (
        serialized_result(result.classifier_result).items()
        <= summary["classifier_result"].items()
    )


def test_single_supcon_candidate_still_uses_the_same_handoff_contract(
    pipeline_document, recorded_stages
):
    """A non-sweep pipeline remains valid, with stable isolation from the CE outputs."""
    pipeline_document["supcon"]["optimizer"]["lr"] = 0.001
    result = TwoStageExperiment(_configuration(pipeline_document)).run()
    assert len(result.supcon_results) == 1
    assert result.selected_supcon == result.supcon_results[0]
    assert (
        result.selected_supcon.output_directory
        == Path(pipeline_document["output_directory"]) / "supcon" / "run_001"
    )
    assert recorded_stages.events == ["supcon", "classifier"]


@pytest.mark.parametrize("fail_at", [0, 2])  # first and last candidate
def test_failed_supcon_candidate_stops_without_selecting_a_partial_sweep(
    pipeline_document, recorded_stages, fail_at
):
    """Propagate failure, preserve completed artifacts, and never start CE or remaining candidates."""
    recorded_stages.fail_at = fail_at
    with pytest.raises(RuntimeError, match="SupCon candidate failure"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == ["supcon"] * (fail_at + 1)
    assert not recorded_stages.classifier
    root = Path(pipeline_document["output_directory"])
    assert not (root / "selection.yaml").exists()
    for index in range(fail_at):
        assert (
            torch.load(
                root / "supcon" / f"run_{index + 1:03d}" / "best.pt", weights_only=True
            )["candidate"]
            == index
        )


def test_non_yolo_head_is_rejected_before_any_supcon_candidate(
    pipeline_document, recorded_stages, tmp_path
):
    """Even a later unsupported head must fail during preparation, not after training."""
    valid = tmp_path / "valid-head.pt"
    YOLO(pipeline_document["supcon"]["model_name"], task="classify").save(valid)
    unsupported = tmp_path / "unsupported-head.pt"
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))
    model.names = {0: "mouse", 1: "rat"}
    torch.save({"model": model, "train_args": {"task": "classify"}}, unsupported)
    pipeline_document["classifier"]["initialization"]["head"] = {
        "replace": "full",
        "source": "yolo_checkpoint",
        "checkpoint": {"type": "sweep", "values": [str(valid), str(unsupported)]},
    }
    with pytest.raises(NotImplementedError, match="YOLO|support"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize(
    "head_replace,freeze", [("full", 2), ("full", [1]), ("linear", 3), ("linear", [2])]
)
def test_out_of_range_classifier_freeze_is_rejected_before_any_supcon_candidate(
    pipeline_document, recorded_stages, head_replace, freeze
):
    """The SupCon architecture fixes the classifier's layer count, so check it before SupCon trains."""
    # Pipeline network: 0=Conv encoder, 1=Classify head.
    pipeline_document["classifier"]["initialization"]["head"]["replace"] = head_replace
    pipeline_document["classifier"]["freeze"] = freeze
    with pytest.raises(ValueError, match="freeze"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


def test_classifier_freeze_is_checked_against_every_supcon_architecture(
    pipeline_document, recorded_stages, tmp_path
):
    """The winning encoder is unknown before SupCon, so a freeze must fit every candidate."""
    deeper = tmp_path / "deeper-cls.yaml"
    deeper.write_text(
        yaml.safe_dump(
            {
                "nc": 2,
                "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 1]]],
                "head": [[-1, 1, "Classify", [2]]],
            }
        )
    )
    shallow = pipeline_document["supcon"]["model_name"]
    pipeline_document["supcon"]["model_name"] = {
        "type": "sweep",
        "values": [str(deeper), shallow],
    }
    # Count 3 freezes the deeper network's conv layers and head conv; the shallow one has 2 layers.
    pipeline_document["classifier"]["freeze"] = 3
    with pytest.raises(ValueError, match="freeze"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


def test_each_stage_section_exports_every_run_it_configures(
    pipeline_document, tmp_path
):
    """SupCon candidates export encoders and CE candidates export classifiers, all reported."""

    # Native classifier export needs a network above Ultralytics' minimum file size.
    pipeline_document["supcon"]["model_name"] = str(
        widened_architecture(
            Path(pipeline_document["supcon"]["model_name"]), tmp_path / "wide-cls.yaml"
        )
    )
    pipeline_document["supcon"]["onnx_export"] = {}
    pipeline_document["classifier"]["onnx_export"] = {}
    pipeline_document["classifier"]["seed"] = {"type": "sweep", "values": [29, 31]}
    result = TwoStageExperiment(_configuration(pipeline_document)).run()
    runs = [*result.supcon_results, *result.classifier_results]
    assert len(result.supcon_results) == 3 and len(result.classifier_results) == 2
    for supcon in result.supcon_results:
        assert supcon.onnx_model == supcon.output_directory / "best.onnx"
    for classifier in result.classifier_results:
        assert (
            classifier.onnx_model
            == classifier.output_directory / "weights" / "best.onnx"
        )
    assert all(run.onnx_model.is_file() for run in runs)
    summary = yaml.safe_load((result.output_directory / "selection.yaml").read_text())
    for name, results in (
        ("supcon_results", result.supcon_results),
        ("classifier_results", result.classifier_results),
    ):
        assert [item["onnx_model"] for item in summary[name]] == [
            str(item.onnx_model) for item in results
        ]


def test_classifier_failure_is_not_reported_as_a_completed_pipeline(
    pipeline_document, recorded_stages
):
    """Classifier exceptions propagate without deleting successful SupCon artifacts."""
    recorded_stages.fail_classifier = True
    with pytest.raises(RuntimeError, match="classifier failure"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == ["supcon", "supcon", "supcon", "classifier"]
    assert all(result.best_checkpoint.is_file() for result in recorded_stages.results)
    summary_path = Path(pipeline_document["output_directory"]) / "selection.yaml"
    if summary_path.exists():
        assert yaml.safe_load(summary_path.read_text()).get("classifier_result") is None


def test_stale_artifacts_are_not_ranked_or_pruned(pipeline_document, recorded_stages):
    """Rank only current structured results; unrelated prior run files remain untouched."""
    root = Path(pipeline_document["output_directory"])
    stale = root / "supcon" / "run_999" / "best.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old checkpoint, intentionally not loadable")
    result = TwoStageExperiment(_configuration(pipeline_document)).run()
    assert result.selected_supcon == result.supcon_results[1]
    assert stale.read_bytes() == b"old checkpoint, intentionally not loadable"
    assert len(result.supcon_results) == 3


@pytest.mark.parametrize(
    "override",
    [
        {"tracked_metric": "supcon_loss_mean"},
        {"tracked_mode": "min"},
        {"validation_dataset": None},
    ],
)
def test_incompatible_reported_candidate_metadata_cannot_start_ce(
    pipeline_document, recorded_stages, override
):
    """A completed result with an incompatible ranking contract is not silently ignored."""
    recorded_stages.overrides[1] = override
    with pytest.raises(ValueError, match="metric|mode|validation"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert not recorded_stages.classifier


def test_configuration_round_trips_without_resolving_the_selected_source_early(
    pipeline_document, tmp_path
):
    """Parsing does not require a checkpoint that will only exist after SupCon training."""
    before = deepcopy(pipeline_document)
    config = _configuration(pipeline_document)
    assert pipeline_document == before
    serialized = config.to_dict()
    assert serialized["classifier"]["initialization"]["encoder"] == {
        "source": "selected_supcon"
    }
    assert TwoStageConfiguration.from_mapping(serialized).to_dict() == serialized
    path = tmp_path / "roundtrip.yaml"
    path.write_text(yaml.safe_dump(serialized))
    assert TwoStageConfiguration.from_yaml(path).to_dict() == serialized
    assert not Path(pipeline_document["output_directory"]).exists()


def test_supcon_yaml_sweeps_preserve_existing_coupled_tag_semantics(
    pipeline_document, recorded_stages, tmp_path
):
    """Keep full-root target paths and sweep-product order from the existing loader."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"] = "SWEEP_LR"
    document["supcon"]["output_features"] = "COUPLED_FEATURES"
    path = tmp_path / "sweep.yaml"
    text = (
        yaml.safe_dump(document)
        .replace("lr: SWEEP_LR", "lr: !sweep\n      values: [0.001, 0.002, 0.003]")
        .replace(
            "output_features: COUPLED_FEATURES",
            "output_features: !coupled-sweep\n    target: supcon.optimizer.lr\n    values: [4, 8, 12]",
        )
    )
    path.write_text(text)
    config = TwoStageConfiguration.from_yaml(path)
    TwoStageExperiment(config).run()
    assert [
        (run.optimizer["lr"], run.output_features) for run in recorded_stages.supcon
    ] == [
        (0.001, 4),
        (0.002, 8),
        (0.003, 12),
    ]
    assert len(recorded_stages.classifier) == 1


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("data", "validation_dataset"), None, "validation"),
        (("data", "class_names"), ["mouse", "mouse"], "class_names"),
        (
            ("supcon", "optimizer", "lr"),
            {"type": "sweep", "values": [0.001, 0]},
            "optimizer",
        ),
        (
            ("supcon", "optimizer", "lr"),
            {"type": "sweep", "values": []},
            "empty|candidate|sweep",
        ),
        (
            ("supcon", "temperature"),
            {"type": "sweep", "values": [0.05, 0.2]},
            "temperature",
        ),
        (
            ("supcon", "tracked_mode"),
            {"type": "sweep", "values": ["min", "max"]},
            "mode|direction",
        ),
        (
            ("supcon", "tracked_metric"),
            {"type": "sweep", "values": ["balanced_accuracy", "supcon_loss_mean"]},
            "metric",
        ),
        (
            ("data", "train_dataset"),
            {"type": "sweep", "values": ["a", "b"]},
            "data|sweep",
        ),
        (
            ("classifier", "initialization", "encoder"),
            {"source": "supcon", "checkpoint": "some.pt"},
            "selected_supcon|encoder",
        ),
        # the handoff is exactly {source: selected_supcon}; no checkpoint may be added
        (
            ("classifier", "initialization", "encoder", "checkpoint"),
            "some.pt",
            "encoder|checkpoint",
        ),
        (("classifier", "initialization", "head", "replace"), "other", "replace"),
        (("classifier", "train_kwargs", "resume"), True, "resume"),
        (
            ("supcon", "scheduler"),
            {"name": "ReduceLROnPlateau", "parameters": {"mode": "min"}},
            "direction|tracked_mode",
        ),
    ],
)
def test_complete_configuration_is_validated_before_any_candidate_or_output(
    pipeline_document, recorded_stages, path, value, message
):
    """Later invalid candidates and classifier settings must fail before the expensive first stage."""
    parent = pipeline_document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        config = _configuration(pipeline_document)
        TwoStageExperiment(config).run()
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
def test_pipeline_output_must_stay_outside_both_shared_splits(
    pipeline_document, recorded_stages, split
):
    """The pipeline root and every child stage must preserve input split contents."""
    pipeline_document["output_directory"] = str(
        Path(pipeline_document["data"][split]) / "pipeline"
    )
    with pytest.raises(ValueError, match="outside|output_directory"):
        _configuration(pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize(
    "section,key,value",
    [
        # unknown root/data keys, and shared keys repeated inside each stage
        (None, "unknown", True),
        ("data", "test_dataset", "must-not-be-used"),
        ("supcon", "train_dataset", "other-training"),
        ("supcon", "output_directory", "other-output"),
        ("classifier", "class_names", ["other"]),
        ("classifier", "validation_dataset", "other-validation"),
    ],
)
def test_unknown_or_conflicting_ownership_is_rejected_not_silently_ignored(
    pipeline_document, section, key, value
):
    """One data owner and one output layout; unknown root or data settings are errors."""
    target = pipeline_document if section is None else pipeline_document[section]
    target[key] = value
    with pytest.raises(ValueError, match=key):
        _configuration(pipeline_document)
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("section", ["supcon", "classifier"])
def test_pipeline_tracking_settings_are_owned_by_the_root(pipeline_document, section):
    """Stage overrides cannot silently create unrelated tracking runs or registries."""
    pipeline_document[section]["mlflow"] = {
        "tracking_uri": "sqlite:///unused.db",
        "experiment_name": "pipeline",
    }
    with pytest.raises(ValueError, match="mlflow|tracking"):
        _configuration(pipeline_document)


@pytest.mark.parametrize(
    "source,relative",
    [
        ("model", "config.yaml"),
        ("model", "supcon/run_001/best.pt"),
        ("head", "classifier/weights/best.pt"),
    ],
)
def test_generated_pipeline_artifacts_cannot_overwrite_model_or_head_inputs(
    pipeline_document, recorded_stages, source, relative
):
    """Pipeline root/child publications must retain the standalone source-safety contract."""
    path = Path(pipeline_document["output_directory"]) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"source bytes must remain unchanged")
    if source == "model":
        pipeline_document["supcon"]["model_name"] = str(path)
    else:
        pipeline_document["classifier"]["initialization"]["head"] = {
            "replace": "linear",
            "source": "yolo_checkpoint",
            "checkpoint": str(path),
        }
    # Require the collision guard itself, not an unrelated attempt to load these
    # intentionally opaque bytes as model weights or architecture YAML.
    with pytest.raises(ValueError, match="collision|collid"):
        _configuration(pipeline_document)
    assert recorded_stages.events == []
    assert path.read_bytes() == b"source bytes must remain unchanged"


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    rates=st.lists(st.sampled_from([0.001, 0.002, 0.003]), min_size=1, max_size=4),
    mode=st.sampled_from(["min", "max"]),
    batchnorm=st.sampled_from(["eval", "adapt"]),
    freeze=st.one_of(st.integers(0, 2), st.lists(st.integers(0, 1), unique=True)),
)
def test_configuration_round_trip_preserves_sweep_order_and_classifier_policy(
    pipeline_document, rates, mode, batchnorm, freeze
):
    """Repeated candidates, scoring direction, BN and freeze policy survive serialization."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"]["values"] = rates
    document["supcon"]["tracked_mode"] = mode
    # The BatchNorm policy only exists when some layer is frozen.
    batchnorm = batchnorm if freeze not in (0, []) else None
    document["classifier"]["encoder_batchnorm"] = batchnorm
    document["classifier"]["freeze"] = freeze
    config = _configuration(document)
    saved = config.to_dict()
    assert saved["supcon"]["optimizer"]["lr"]["values"] == rates
    assert saved["supcon"]["tracked_mode"] == mode
    assert saved["classifier"]["encoder_batchnorm"] == batchnorm
    assert saved["classifier"]["freeze"] == freeze
    assert TwoStageConfiguration.from_mapping(saved).to_dict() == saved


@pytest.mark.parametrize("mode", ["min"])  # direction is covered by stubbed tests
def test_real_cpu_stages_select_best_epoch_then_train_a_portable_ce_classifier(
    pipeline_document, monkeypatch, mode
):
    """Run real optimizers and source conversion; script only SupCon's validation scores."""
    document = pipeline_document
    document["supcon"]["optimizer"]["lr"]["values"] = [0.001, 0.002]
    document["supcon"]["tracked_mode"] = mode
    # Linear probe on the 2-layer network: encoder and head conv frozen, only Linear trains.
    document["classifier"]["freeze"] = 2
    values = (
        [[0.3, 0.8, 0.5], [0.2, 0.9, 0.6]]
        if mode == "max"
        else [[0.7, 0.2, 0.5], [0.8, 0.1, 0.4]]
    )
    scores = iter(
        (value, ce)
        for run in values
        for value, ce in zip(run, [0.1, 0.2, 0.3], strict=True)
    )
    calls = []

    def validate(self, model, loader, criterion):
        """Make the configured epoch/run winner conflict with hardcoded loss selection."""
        calls.append(self.configuration.output_directory)
        value, ce = next(scores)
        return {"balanced_accuracy": value, "supcon_loss_mean": ce}

    monkeypatch.setattr(SupConExperiment, "_validate", validate)
    images = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ("train_dataset", "validation_dataset")
        for path in Path(document["data"][split]).rglob("*.png")
    }
    result = TwoStageExperiment(_configuration(document)).run()
    assert (
        len(calls) == 6
    )  # Exactly ordinary epoch validation; never a second scoring pass.
    assert len(result.supcon_results) == 2
    assert result.selected_supcon == result.supcon_results[1]
    assert result.selected_supcon.best_epoch == 2
    assert result.selected_supcon.best_value == values[1][1]
    source = result.selected_supcon.best_checkpoint
    source_state = torch.load(source, weights_only=True)
    assert source_state["epoch"] == 2
    assert (
        torch.load(result.selected_supcon.last_checkpoint, weights_only=True)["epoch"]
        == 3
    )
    classifier = YOLO(
        str(result.classifier_result.last_checkpoint), task="classify"
    ).model
    assert classifier.names == {
        index: name for name, index in source_state["label_names"].items()
    }
    assert classifier.provenance["encoder_checkpoint"] == str(source.resolve())
    assert (
        classifier.provenance["encoder_checkpoint_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    boundary = f"model.{len(source_state['architecture']['backbone'])}.linear."
    for name, tensor in classifier.state_dict().items():
        if not name.startswith(boundary):
            # Native checkpoint saving uses FP16, unlike the source SupCon snapshot.
            expected = source_state["model_state_dict"]["full_model.model." + name]
            torch.testing.assert_close(
                tensor.float(), expected.float(), rtol=0.001, atol=0.0001
            )
    restored = YOLO(
        str(result.classifier_result.best_checkpoint), task="classify"
    ).model.eval()
    with torch.inference_mode():
        output = restored(torch.rand(2, 3, 32, 32))
        assert output[0].shape == (2, 2)
    assert result.classifier_result.best_checkpoint.with_suffix(
        ".model_card.yaml"
    ).is_file()
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        for path, checksum in images.items()
    )


# every shared key takes the same ownership check; the tracking section and one data key
@pytest.mark.parametrize("key", ["mlflow", "train_dataset"])
def test_whole_supcon_mapping_sweeps_cannot_bypass_shared_ownership(
    pipeline_document, recorded_stages, key
):
    """Validate restrictions inside expanded mappings, including later candidates."""
    valid = deepcopy(pipeline_document["supcon"])
    valid["optimizer"]["lr"] = 0.001
    invalid = deepcopy(valid)
    overrides = {
        "mlflow": {"tracking_uri": "sqlite:///unused.db", "experiment_name": "bypass"},
        "train_dataset": "other-training",
        "validation_dataset": "other-validation",
        "class_names": ["other"],
        "output_directory": "other-output",
    }
    invalid[key] = overrides[key]
    pipeline_document["supcon"] = {"type": "sweep", "values": [valid, invalid]}
    with pytest.raises(ValueError, match=key):
        _configuration(pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("section", ["supcon", "classifier"])
@pytest.mark.parametrize("seed", [-1, True])  # one range and one type violation
def test_every_stage_seed_is_validated_before_any_candidate(
    pipeline_document, recorded_stages, section, seed
):
    """The supported Python/NumPy/PyTorch RNG intersection is an unsigned 32-bit integer."""
    value = {"type": "sweep", "values": [11, seed]} if section == "supcon" else seed
    pipeline_document[section]["seed"] = value
    with pytest.raises(ValueError, match="seed"):
        _configuration(pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("fail_at", [0, 2])  # first and last candidate
def test_failed_rerun_invalidates_previous_success_without_pruning_artifacts(
    pipeline_document, recorded_stages, fail_at
):
    """A new failed invocation cannot retain an older completed-classifier summary."""
    first = TwoStageExperiment(_configuration(pipeline_document)).run()
    summary_path = first.output_directory / "selection.yaml"
    assert yaml.safe_load(summary_path.read_text())["classifier_result"] is not None
    classifier_bytes = first.classifier_result.best_checkpoint.read_bytes()
    recorded_stages.supcon.clear()
    recorded_stages.classifier.clear()
    recorded_stages.results.clear()
    recorded_stages.events.clear()
    recorded_stages.fail_at = fail_at
    pipeline_document["supcon"]["seed"] = 19
    with pytest.raises(RuntimeError, match="SupCon candidate failure"):
        TwoStageExperiment(_configuration(pipeline_document)).run()
    assert recorded_stages.events == ["supcon"] * (fail_at + 1)
    assert not recorded_stages.classifier
    summary = yaml.safe_load(summary_path.read_text())
    assert summary.get("classifier_result") is None
    assert summary.get("selected_supcon") is None
    assert not summary.get("supcon_results")
    assert first.classifier_result.best_checkpoint.read_bytes() == classifier_bytes


@pytest.fixture
def tracking_document(pipeline_document):
    """Use two three-epoch candidates plus CE, with distinct stage objectives/directions."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"]["values"] = [0.001, 0.002]
    document["classifier"].update(
        epochs=3,
        tracked_metric="balanced_accuracy",
        tracked_mode="max",
        validation_metrics={
            "balanced_accuracy": {"function": "balanced_accuracy_score"}
        },
    )
    return document


def _run_tracking_document(document, store, *, register=False):
    """Enable shared tracking without mutating the reusable document fixture."""
    values = deepcopy(document)
    values["mlflow"] = deepcopy(store.configuration)
    if register:
        values["mlflow"]["registered_model_name"] = store.registry_name
    config = TwoStageConfiguration.from_mapping(values)
    return config, TwoStageExperiment(config).run()


def _expected_encoder_provenance(result):
    """Require the selected run's original configured objective, not a sign-inverted score."""
    selected = result.selected_supcon
    return {
        "encoder_checkpoint": str(selected.best_checkpoint.resolve()),
        "encoder_checkpoint_sha256": hashlib.sha256(
            selected.best_checkpoint.read_bytes()
        ).hexdigest(),
        "encoder_run_id": selected.mlflow_run_id,
        "encoder_epoch": selected.best_epoch,
        "encoder_tracked_metric": selected.tracked_metric,
        "encoder_tracked_mode": selected.tracked_mode,
        "encoder_best_value": selected.best_value,
        "pipeline_run_id": result.mlflow_run_id,
    }


@pytest.mark.parametrize("register", [False, True])
def test_shared_tracking_configuration_round_trips_before_training(
    tracking_document, tracking_store, tmp_path, register
):
    """One optional root config owns tracking; no backend run or output is created by parsing."""
    document = tracking_document
    document["mlflow"] = deepcopy(tracking_store.configuration)
    if register:
        document["mlflow"]["registered_model_name"] = tracking_store.registry_name
    before = deepcopy(document)
    config = TwoStageConfiguration.from_mapping(document)
    assert config.mlflow == MlflowConfiguration.from_mapping(before["mlflow"])
    assert document == before
    path = tmp_path / "tracking.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    restored = TwoStageConfiguration.from_yaml(path)
    assert restored.to_dict() == config.to_dict()
    assert restored.mlflow == config.mlflow
    assert tracking_store.client.search_runs([tracking_store.experiment_id]) == []
    assert not Path(document["output_directory"]).exists()


@pytest.mark.parametrize("mode", ["min"])  # direction is covered by stubbed tests
@pytest.mark.parametrize("register", [False, True])
def test_real_pipeline_tracks_one_parent_and_stage_children_with_selected_provenance(
    tracking_document, tracking_store, scripted_validation, mode, register
):
    """Metric-selected epochs, registry model, cards and tracking artifacts must describe one result."""
    tracking_document["supcon"]["tracked_mode"] = mode
    config, result = _run_tracking_document(
        tracking_document, tracking_store, register=register
    )
    assert (
        len(scripted_validation.supcon) == 6
        and len(scripted_validation.classifier) == 3
    )
    assert result.selected_supcon == result.supcon_results[1]
    assert result.selected_supcon.best_epoch == result.classifier_result.best_epoch == 2
    assert result.selected_supcon.best_value == (0.1 if mode == "min" else 0.9)
    assert result.classifier_result.best_value == 0.9
    child_ids = [
        item.mlflow_run_id
        for item in (*result.supcon_results, result.classifier_result)
    ]
    assert all(child_ids) and result.mlflow_run_id not in child_ids
    runs = tracking_store.client.search_runs([tracking_store.experiment_id])
    assert len(runs) == 4 and {run.info.run_id for run in runs} == {
        result.mlflow_run_id,
        *child_ids,
    }
    assert all(run.info.status == "FINISHED" for run in runs)
    parent = tracking_store.client.get_run(result.mlflow_run_id)
    assert "mlflow.parentRunId" not in parent.data.tags
    for run in runs:
        assert run.data.tags["git_commit"] == parent.data.tags["git_commit"]
        assert (
            run.data.tags["dataset_revision"]
            == tracking_store.configuration["dataset_revision"]
        )
        if run.info.run_id in child_ids:
            assert run.data.tags["mlflow.parentRunId"] == result.mlflow_run_id
    for candidate in result.supcon_results:
        history = tracking_store.client.get_metric_history(
            candidate.mlflow_run_id, "validation/balanced_accuracy"
        )
        assert [item.step for item in history] == [1, 2, 3]
        assert (
            tracking_store.client.get_run(candidate.mlflow_run_id).data.metrics[
                "best_validation_balanced_accuracy"
            ]
            == candidate.best_value
        )
        card = yaml.safe_load(
            (candidate.output_directory / "best.model_card.yaml").read_text()
        )
        assert card["model"]["task"] == "contrastive-encoder"
        assert (
            card["model"]["registry_version"] is None
        )  # No untrained classifier registry versions.
    ce_run_id = result.classifier_result.mlflow_run_id
    ce_history = tracking_store.client.get_metric_history(
        ce_run_id, "validation/balanced_accuracy"
    )
    assert [item.step for item in ce_history] == [1, 2, 3]
    assert [item.value for item in ce_history] == [0.1, 0.9, 0.4]
    assert (
        tracking_store.client.get_run(ce_run_id).data.metrics[
            "best_validation_balanced_accuracy"
        ]
        == 0.9
    )
    provenance = _expected_encoder_provenance(result)
    for checkpoint in (
        result.classifier_result.best_checkpoint,
        result.classifier_result.last_checkpoint,
    ):
        model = YOLO(str(checkpoint), task="classify").model
        assert provenance.items() <= model.provenance.items()
    card = yaml.safe_load(
        result.classifier_result.best_checkpoint.with_suffix(
            ".model_card.yaml"
        ).read_text()
    )
    assert provenance.items() <= card["provenance"].items()
    assert card["model"]["task"] == "image-classification"
    assert card["validation"]["epoch"] == card["provenance"]["epoch"] == 2
    assert card["provenance"]["git_commit"] == parent.data.tags["git_commit"]
    assert (
        card["provenance"]["dataset_revision"]
        == tracking_store.configuration["dataset_revision"]
    )
    assert (
        card["provenance"]["checkpoint_sha256"]
        == hashlib.sha256(
            result.classifier_result.best_checkpoint.read_bytes()
        ).hexdigest()
    )
    for child in (*result.supcon_results, result.classifier_result):
        for filename in ("best.pt", "best.model_card.yaml", "best.model_card.md"):
            downloaded = download_artifact(
                tracking_store, child.mlflow_run_id, f"best_model/{filename}"
            )
            assert (
                downloaded.read_bytes()
                == (child.best_checkpoint.parent / filename).read_bytes()
            )
    for filename in ("config.yaml", "selection.yaml"):
        assert (
            download_artifact(
                tracking_store, result.mlflow_run_id, filename
            ).read_bytes()
            == (result.output_directory / filename).read_bytes()
        )
    summary = yaml.safe_load((result.output_directory / "selection.yaml").read_text())
    assert summary["mlflow_run_id"] == result.mlflow_run_id
    assert (
        summary["selected_supcon"]["mlflow_run_id"]
        == result.selected_supcon.mlflow_run_id
    )
    assert (
        summary["classifier_result"]["mlflow_run_id"]
        == result.classifier_result.mlflow_run_id
    )
    assert (
        yaml.safe_load((result.output_directory / "config.yaml").read_text())
        == config.to_dict()
    )
    if register:
        version = assert_registered_classifier(tracking_store, result.classifier_result)
        assert {
            model.name for model in tracking_store.client.search_registered_models()
        } == {tracking_store.registry_name}
        assert version.tags["validation_status"] == "passed"
        assert version.tags["encoder_run_id"] == result.selected_supcon.mlflow_run_id
        assert (
            version.tags["encoder_checkpoint_sha256"]
            == provenance["encoder_checkpoint_sha256"]
        )
        assert (
            tracking_store.client.get_model_version_by_alias(
                tracking_store.registry_name,
                tracking_store.configuration["champion_alias"],
            ).version
            == version.version
        )
    else:
        assert tracking_store.client.search_registered_models() == []
    assert mlflow.active_run() is None


def test_untracked_pipeline_keeps_selected_provenance_without_touching_callers_tracking(
    tracking_document, tracking_store, scripted_validation
):
    """Provenance is local functionality too; disabled tracking neither nests nor logs into a caller."""
    with mlflow.start_run(run_name="unrelated-caller") as caller:
        mlflow.set_tag("caller_tag", "unchanged")
        before = tracking_store.client.get_run(caller.info.run_id)
        result = TwoStageExperiment(
            TwoStageConfiguration.from_mapping(tracking_document)
        ).run()
        assert result.mlflow_run_id is None
        assert all(
            item.mlflow_run_id is None
            for item in (*result.supcon_results, result.classifier_result)
        )
        provenance = _expected_encoder_provenance(result)
        for checkpoint in (
            result.classifier_result.best_checkpoint,
            result.classifier_result.last_checkpoint,
        ):
            assert (
                provenance.items()
                <= YOLO(str(checkpoint), task="classify").model.provenance.items()
            )
        card = yaml.safe_load(
            result.classifier_result.best_checkpoint.with_suffix(
                ".model_card.yaml"
            ).read_text()
        )
        assert provenance.items() <= card["provenance"].items()
        assert mlflow.active_run().info.run_id == caller.info.run_id
        after = tracking_store.client.get_run(caller.info.run_id)
        assert after.info.status == "RUNNING"
        assert (
            after.data.metrics == before.data.metrics
            and after.data.params == before.data.params
        )
        assert after.data.tags == before.data.tags
        assert tracking_store.client.list_artifacts(caller.info.run_id) == []
        assert (
            len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
        )
    assert mlflow.active_run() is None


@pytest.mark.parametrize("failure", ["supcon", "classifier", "publication"])
def test_pipeline_failure_closes_owned_runs_and_preserves_the_existing_champion(
    tracking_document,
    tracking_store,
    scripted_validation,
    monkeypatch,
    tmp_path,
    failure,
):
    """A failed candidate/CE/publication is never skipped, promoted, or reported as completed."""
    tracking_store.client.create_registered_model(tracking_store.registry_name)
    prior = tmp_path / "prior-model"
    prior.mkdir()
    previous = tracking_store.client.create_model_version(
        tracking_store.registry_name, prior.as_uri()
    ).version
    tracking_store.client.set_registered_model_alias(
        tracking_store.registry_name,
        tracking_store.configuration["champion_alias"],
        previous,
    )
    scripted_validation.failure = failure
    if failure == "publication":

        def fail_publication(*args, **kwargs):
            """Simulate an external MLflow model-publication failure after successful CE training."""
            raise RuntimeError("deliberate classifier publication failure")

        monkeypatch.setattr(mlflow.pytorch, "log_model", fail_publication)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run_tracking_document(tracking_document, tracking_store, register=True)
    runs = tracking_store.client.search_runs([tracking_store.experiment_id])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(parents) == 1 and parents[0].info.status == "FAILED"
    children = [
        run
        for run in runs
        if run.data.tags.get("mlflow.parentRunId") == parents[0].info.run_id
    ]
    assert len(children) == (2 if failure == "supcon" else 3)
    assert [run.info.status for run in children].count("FAILED") == 1
    assert all(run.info.status in {"FINISHED", "FAILED"} for run in runs)
    if failure == "supcon":
        assert scripted_validation.classifier == []
    versions = tracking_store.client.search_model_versions(
        f"name='{tracking_store.registry_name}'"
    )
    assert len(versions) == 1 and versions[0].version == previous
    assert (
        tracking_store.client.get_model_version_by_alias(
            tracking_store.registry_name, tracking_store.configuration["champion_alias"]
        ).version
        == previous
    )
    summary = Path(tracking_document["output_directory"]) / "selection.yaml"
    if summary.exists():
        assert yaml.safe_load(summary.read_text()).get("classifier_result") is None
    assert mlflow.active_run() is None


def test_tracked_pipeline_rejects_an_unrelated_active_run_before_creating_outputs(
    tracking_document, tracking_store
):
    """Only the pipeline owns its parent run; never hijack, end, or attach to the caller's run."""
    document = deepcopy(tracking_document)
    document["mlflow"] = tracking_store.configuration
    configuration = TwoStageConfiguration.from_mapping(document)
    with mlflow.start_run(run_name="unrelated-caller") as caller:
        with pytest.raises(
            (ValueError, RuntimeError, MlflowException), match="active|existing|run"
        ):
            TwoStageExperiment(configuration).run()
        assert mlflow.active_run().info.run_id == caller.info.run_id
        assert (
            tracking_store.client.get_run(caller.info.run_id).info.status == "RUNNING"
        )
        assert (
            len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
        )
        assert not Path(document["output_directory"]).exists()
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"experiment_name": ""}, "experiment_name"),
        (
            {
                "tracking_uri": {
                    "type": "sweep",
                    "values": ["sqlite:///a.db", "sqlite:///b.db"],
                }
            },
            "mlflow|tracking_uri|sweep",
        ),
        ({"unexpected": True}, "unexpected"),
    ],
)
def test_invalid_shared_tracking_settings_fail_before_runs_or_training(
    tracking_document, tracking_store, updates, message
):
    """New root-level tracking support must retain complete prevalidation and strict parsing."""
    tracking_document["mlflow"] = {**tracking_store.configuration, **updates}
    with pytest.raises((TypeError, ValueError), match=message):
        TwoStageConfiguration.from_mapping(tracking_document)
    assert tracking_store.client.search_runs([tracking_store.experiment_id]) == []
    assert not Path(tracking_document["output_directory"]).exists()


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    revision=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12
    ),
    alias=st.sampled_from(["champion", "candidate"]),
    register=st.booleans(),
)
def test_tracking_metadata_round_trip_preserves_optional_registry_identity(
    tracking_document, tracking_store, revision, alias, register
):
    """Serialization cannot silently reset registry aliases or dataset provenance."""
    document = deepcopy(tracking_document)
    document["mlflow"] = {
        **tracking_store.configuration,
        "dataset_revision": revision,
        "champion_alias": alias,
    }
    if register:
        document["mlflow"]["registered_model_name"] = tracking_store.registry_name
    config = TwoStageConfiguration.from_mapping(document)
    restored = TwoStageConfiguration.from_mapping(config.to_dict())
    assert restored.mlflow.dataset_revision == revision
    assert restored.mlflow.champion_alias == alias
    assert restored.mlflow.registered_model_name == (
        tracking_store.registry_name if register else None
    )


def test_tracking_rejects_environment_run_resumption_without_modifying_the_existing_run(
    tracking_document, tracking_store, monkeypatch, tmp_path
):
    """An environment run ID is not authorization to resume, modify, or close someone else's run."""
    caller = tracking_store.client.create_run(
        tracking_store.experiment_id, tags={"owner": "caller"}
    )
    tracking_store.client.log_param(caller.info.run_id, "caller_parameter", "unchanged")
    sentinel = tmp_path / "caller-artifact.txt"
    sentinel.write_text("unchanged caller artifact")
    tracking_store.client.log_artifact(caller.info.run_id, str(sentinel))
    before = tracking_store.client.get_run(caller.info.run_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", caller.info.run_id)
    document = deepcopy(tracking_document)
    document["mlflow"] = tracking_store.configuration
    config = TwoStageConfiguration.from_mapping(document)
    assert mlflow.active_run() is None
    with pytest.raises(ValueError, match="run|resum"):
        TwoStageExperiment(config).run()
    after = tracking_store.client.get_run(caller.info.run_id)
    assert after.info.status == before.info.status == "RUNNING"
    assert after.info.end_time == before.info.end_time
    assert after.data.params == before.data.params
    assert after.data.metrics == before.data.metrics
    assert after.data.tags == before.data.tags
    assert [
        item.path for item in tracking_store.client.list_artifacts(caller.info.run_id)
    ] == [sentinel.name]
    assert (
        download_artifact(
            tracking_store, caller.info.run_id, sentinel.name
        ).read_bytes()
        == sentinel.read_bytes()
    )
    assert len(tracking_store.client.search_runs([tracking_store.experiment_id])) == 1
    assert mlflow.active_run() is None
    assert not Path(document["output_directory"]).exists()
