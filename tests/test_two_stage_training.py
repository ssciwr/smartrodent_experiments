"""Contracts for unified SupCon-sweep → selected encoder → CE orchestration."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
from PIL import Image
import pytest
import torch
import yaml

from scripts.train_supcon import SupConExperiment
from smartrodent.classifier_model import YoloClassifierModel
from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.experiment_results import ExperimentResult


@pytest.fixture
def pipeline_api():
    """Keep the unimplemented orchestration API red without breaking collection."""
    from smartrodent import two_stage_training

    return two_stage_training


@pytest.fixture
def pipeline_document(tmp_path):
    """Build both existing splits and a tiny local model, never touching real data."""
    torch.set_num_threads(2)
    for split in ("train", "val"):
        for label, name in enumerate(("mouse", "rat")):
            directory = tmp_path / split / name
            directory.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (16, 16), (40 + 20 * index, label * 220, 80)).save(
                    directory / f"{index}.png"
                )
    architecture = tmp_path / "tiny-cls.yaml"
    architecture.write_text(yaml.safe_dump({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
        "head": [[-1, 1, "Classify", [2]]],
    }))
    transforms = {
        "image_size": 16,
        "resize": {"antialias": True},
        "random_adjust_sharpness": {"sharpness_factor": 0.5, "p": 0.0},
        "random_grayscale": {"p": 0.0},
        "random_perspective": {"distortion_scale": 0.5, "p": 0.0},
        "random_horizontal_flip": {"p": 0.0},
        "random_resized_crop": {"scale": [1.0, 1.0], "ratio": [1.0, 1.0]},
    }
    classifier_transforms = deepcopy(transforms)
    classifier_transforms["random_horizontal_flip"]["p"] = 0.3
    return {
        "output_directory": str(tmp_path / "pipeline"),
        "data": {
            "train_dataset": str(tmp_path / "train"),
            "validation_dataset": str(tmp_path / "val"),
            "class_names": ["rat", "mouse"],
        },
        "supcon": {
            "model_name": str(architecture), "epochs": 3,
            "transforms": deepcopy(transforms),
            "optimizer": {"lr": {"type": "sweep", "values": [0.001, 0.002, 0.003]}},
            "classes_per_batch": 2, "samples_per_class": 2, "samples_per_epoch": 8,
            "workers": 0, "device": "cpu", "seed": 11,
            "dropout": 0.0, "output_features": 4, "temperature": 0.1,
            "tracked_metric": "balanced_accuracy", "tracked_mode": "max",
            "validation_metrics": {"balanced_accuracy": {"function": "balanced_accuracy_score"}},
        },
        "classifier": {
            "initialization": {
                "encoder": {"source": "selected_supcon"},
                "head": {"replace": "linear", "source": "random"},
            },
            "freeze_encoder": True, "encoder_batchnorm": "eval",
            "epochs": 2, "batch_size": 3, "workers": 0, "device": "cpu", "seed": 29,
            "transforms": classifier_transforms, "optimizer": {"lr": 0.004},
            "tracked_metric": "ce_loss_mean", "tracked_mode": "min",
        },
    }


@pytest.fixture
def recorded_stages(monkeypatch):
    """Replace only stage execution, retaining real configuration validation and ranking."""
    observed = SimpleNamespace(
        supcon=[], classifier=[], results=[], events=[], values=[0.2, 0.8, 0.5],
        fail_at=None, fail_classifier=False, overrides={},
    )

    def run_supcon(self):
        """Publish distinguishable best/last bytes and report original configured scores."""
        config = self.configuration
        index = len(observed.supcon)
        observed.supcon.append(config)
        observed.events.append("supcon")
        if index == observed.fail_at:
            raise RuntimeError("deliberate SupCon candidate failure")
        directory = Path(config.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        best, last = directory / "best.pt", directory / "last.pt"
        torch.save({"candidate": index, "epoch": 2}, best)
        torch.save({"candidate": index, "epoch": 3}, last)
        result = ExperimentResult(
            output_directory=directory, last_checkpoint=last, best_checkpoint=best,
            best_epoch=2, best_value=observed.values[index],
            tracked_metric=config.effective_tracked_metric,
            tracked_mode=config.effective_tracked_mode,
            validation_dataset=config.validation_dataset,
        )
        result = replace(result, **observed.overrides.get(index, {}))
        observed.results.append(result)
        return result

    def run_classifier(self):
        """Record the actual handoff configuration without requiring fake SupCon weights to load."""
        config = self.configuration
        observed.classifier.append(config)
        observed.events.append("classifier")
        if observed.fail_classifier:
            raise RuntimeError("deliberate classifier failure")
        directory = Path(config.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        best, last = directory / "best.pt", directory / "last.pt"
        torch.save({"checkpoint_type": "classifier", "epoch": 1}, best)
        torch.save({"checkpoint_type": "classifier", "epoch": 2}, last)
        return ExperimentResult(
            output_directory=directory, last_checkpoint=last, best_checkpoint=best,
            best_epoch=1, best_value=0.4, tracked_metric=config.tracked_metric,
            tracked_mode=config.effective_tracked_mode, validation_dataset=config.validation_dataset,
        )

    monkeypatch.setattr(SupConExperiment, "run_with_result", run_supcon)
    monkeypatch.setattr(ClassifierExperiment, "run_with_result", run_classifier)
    return observed


def _configuration(pipeline_api, document):
    """Parse the complete pipeline before constructing an experiment."""
    return pipeline_api.TwoStageConfiguration.from_mapping(document)


def _serialized_result(result):
    """Specify the portable result fields, allowing additional summary metadata."""
    return {
        "output_directory": str(result.output_directory),
        "last_checkpoint": str(result.last_checkpoint),
        "best_checkpoint": str(result.best_checkpoint),
        "best_epoch": result.best_epoch, "best_value": result.best_value,
        "tracked_metric": result.tracked_metric, "tracked_mode": result.tracked_mode,
        "validation_dataset": str(result.validation_dataset) if result.validation_dataset else None,
        "mlflow_run_id": result.mlflow_run_id,
    }


@pytest.mark.parametrize(
    "mode,values,winner",
    [
        ("max", [0.2, 0.8, 0.5], 1),
        ("min", [0.8, 0.2, 0.5], 1),
        ("max", [0.8, 0.8, 0.2], 0),
        ("min", [0.2, 0.2, 0.8], 0),
    ],
)
def test_pipeline_uses_configured_objective_and_passes_winning_best_not_last(
    pipeline_api, pipeline_document, recorded_stages, mode, values, winner
):
    """No rescoring, hardcoded loss, last checkpoint, or unordered filesystem discovery."""
    document = pipeline_document
    document["supcon"]["tracked_mode"] = mode
    document["supcon"]["scheduler"] = {
        "name": "ReduceLROnPlateau", "parameters": {"mode": mode, "patience": 2, "factor": 0.5},
    }
    document["supcon"]["early_stopping"] = {
        "patience": 5, "tolerance": 0.01, "comparison": "less" if mode == "min" else "greater",
    }
    document["classifier"]["scheduler"] = {
        "name": "ReduceLROnPlateau", "parameters": {"mode": "min", "patience": 3, "factor": 0.2},
    }
    document["classifier"]["early_stopping"] = {
        "patience": 4, "tolerance": 0.02, "comparison": "less",
    }
    recorded_stages.values = values
    config = _configuration(pipeline_api, document)
    before = deepcopy(config.to_dict())
    result = pipeline_api.TwoStageExperiment(config).run()
    assert isinstance(result, pipeline_api.TwoStageResult)
    assert recorded_stages.events == ["supcon", "supcon", "supcon", "classifier"]
    assert tuple(result.supcon_results) == tuple(recorded_stages.results)
    assert result.selected_supcon == result.supcon_results[winner]
    selected = result.selected_supcon
    classifier = recorded_stages.classifier[0]
    assert selected.tracked_metric == "balanced_accuracy" and selected.tracked_mode == mode
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
    assert classifier.output_directory == result.classifier_result.output_directory == root / "classifier"
    assert result.classifier_result.tracked_metric == "ce_loss_mean"
    assert result.classifier_result.best_value == 0.4
    assert [item.optimizer["lr"] for item in recorded_stages.supcon] == [0.001, 0.002, 0.003]
    assert all(item.seed == 11 and item.temperature == 0.1 for item in recorded_stages.supcon)
    assert classifier.seed == 29 and classifier.optimizer == {"lr": 0.004}
    assert classifier.batch_size == 3 and classifier.freeze_encoder
    assert classifier.encoder_batchnorm == "eval"
    assert classifier.initialization["head"] == document["classifier"]["initialization"]["head"]
    for stage in [*recorded_stages.supcon, classifier]:
        stage_settings = document["classifier"] if stage is classifier else document["supcon"]
        assert stage.transforms == stage_settings["transforms"]
        assert stage.scheduler.name == stage_settings["scheduler"]["name"]
        assert stage.scheduler.parameters == stage_settings["scheduler"]["parameters"]
        assert stage.early_stopping == stage_settings["early_stopping"]
        assert stage.train_dataset == Path(document["data"]["train_dataset"])
        assert stage.validation_dataset == Path(document["data"]["validation_dataset"])
        assert stage.class_names == ("rat", "mouse")
    assert config.to_dict() == before
    assert yaml.safe_load((root / "config.yaml").read_text()) == before
    summary = yaml.safe_load((root / "selection.yaml").read_text())
    for candidate, saved in zip(result.supcon_results, summary["supcon_results"], strict=True):
        assert _serialized_result(candidate).items() <= saved.items()
    assert _serialized_result(selected).items() <= summary["selected_supcon"].items()
    assert summary["selected_supcon"]["checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert _serialized_result(result.classifier_result).items() <= summary["classifier_result"].items()


def test_single_supcon_candidate_still_uses_the_same_handoff_contract(
    pipeline_api, pipeline_document, recorded_stages
):
    """A non-sweep pipeline remains valid, with stable isolation from the CE outputs."""
    pipeline_document["supcon"]["optimizer"]["lr"] = 0.001
    result = pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert len(result.supcon_results) == 1
    assert result.selected_supcon == result.supcon_results[0]
    assert result.selected_supcon.output_directory == Path(pipeline_document["output_directory"]) / "supcon" / "run_001"
    assert recorded_stages.events == ["supcon", "classifier"]


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_failed_supcon_candidate_stops_without_selecting_a_partial_sweep(
    pipeline_api, pipeline_document, recorded_stages, fail_at
):
    """Propagate failure, preserve completed artifacts, and never start CE or remaining candidates."""
    recorded_stages.fail_at = fail_at
    with pytest.raises(RuntimeError, match="SupCon candidate failure"):
        pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert recorded_stages.events == ["supcon"] * (fail_at + 1)
    assert not recorded_stages.classifier
    root = Path(pipeline_document["output_directory"])
    assert not (root / "selection.yaml").exists()
    for index in range(fail_at):
        assert torch.load(root / "supcon" / f"run_{index + 1:03d}" / "best.pt", weights_only=True)["candidate"] == index


def test_classifier_failure_is_not_reported_as_a_completed_pipeline(
    pipeline_api, pipeline_document, recorded_stages
):
    """Classifier exceptions propagate without deleting successful SupCon artifacts."""
    recorded_stages.fail_classifier = True
    with pytest.raises(RuntimeError, match="classifier failure"):
        pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert recorded_stages.events == ["supcon", "supcon", "supcon", "classifier"]
    assert all(result.best_checkpoint.is_file() for result in recorded_stages.results)
    summary_path = Path(pipeline_document["output_directory"]) / "selection.yaml"
    if summary_path.exists():
        assert yaml.safe_load(summary_path.read_text()).get("classifier_result") is None


def test_stale_artifacts_are_not_ranked_or_pruned(
    pipeline_api, pipeline_document, recorded_stages
):
    """Rank only current structured results; unrelated prior run files remain untouched."""
    root = Path(pipeline_document["output_directory"])
    stale = root / "supcon" / "run_999" / "best.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old checkpoint, intentionally not loadable")
    result = pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert result.selected_supcon == result.supcon_results[1]
    assert stale.read_bytes() == b"old checkpoint, intentionally not loadable"
    assert len(result.supcon_results) == 3


@pytest.mark.parametrize(
    "override",
    [
        {"tracked_metric": "supcon_loss_mean"},
        {"tracked_mode": "min"},
        {"validation_dataset": None},
        {"validation_dataset": Path("different-validation")},
    ],
)
def test_incompatible_reported_candidate_metadata_cannot_start_ce(
    pipeline_api, pipeline_document, recorded_stages, override
):
    """A completed result with an incompatible ranking contract is not silently ignored."""
    recorded_stages.overrides[1] = override
    with pytest.raises(ValueError, match="metric|mode|validation"):
        pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert not recorded_stages.classifier


def test_configuration_round_trips_without_resolving_the_selected_source_early(
    pipeline_api, pipeline_document, tmp_path
):
    """Parsing does not require a checkpoint that will only exist after SupCon training."""
    before = deepcopy(pipeline_document)
    config = _configuration(pipeline_api, pipeline_document)
    assert pipeline_document == before
    serialized = config.to_dict()
    assert serialized["classifier"]["initialization"]["encoder"] == {"source": "selected_supcon"}
    assert pipeline_api.TwoStageConfiguration.from_mapping(serialized).to_dict() == serialized
    path = tmp_path / "roundtrip.yaml"
    path.write_text(yaml.safe_dump(serialized))
    assert pipeline_api.TwoStageConfiguration.from_yaml(path).to_dict() == serialized
    assert not Path(pipeline_document["output_directory"]).exists()


def test_yaml_sweeps_expand_only_supcon_with_existing_coupled_tag_semantics(
    pipeline_api, pipeline_document, recorded_stages, tmp_path
):
    """Keep full-root target paths and sweep-product order from the existing loader."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"] = "SWEEP_LR"
    document["supcon"]["output_features"] = "COUPLED_FEATURES"
    path = tmp_path / "sweep.yaml"
    text = yaml.safe_dump(document).replace(
        "lr: SWEEP_LR", "lr: !sweep\n      values: [0.001, 0.002, 0.003]"
    ).replace(
        "output_features: COUPLED_FEATURES",
        "output_features: !coupled-sweep\n    target: supcon.optimizer.lr\n    values: [4, 8, 12]",
    )
    path.write_text(text)
    config = pipeline_api.TwoStageConfiguration.from_yaml(path)
    pipeline_api.TwoStageExperiment(config).run()
    assert [(run.optimizer["lr"], run.output_features) for run in recorded_stages.supcon] == [
        (0.001, 4), (0.002, 8), (0.003, 12),
    ]
    assert len(recorded_stages.classifier) == 1


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("data", "validation_dataset"), None, "validation"),
        (("data", "class_names"), ["mouse", "mouse"], "class_names"),
        (("supcon", "optimizer", "lr"), {"type": "sweep", "values": [0.001, 0]}, "optimizer"),
        (("supcon", "optimizer", "lr"), {"type": "sweep", "values": []}, "empty|candidate|sweep"),
        (("supcon", "temperature"), {"type": "sweep", "values": [0.05, 0.2]}, "temperature"),
        (("supcon", "tracked_mode"), {"type": "sweep", "values": ["min", "max"]}, "mode|direction"),
        (("supcon", "tracked_metric"), {"type": "sweep", "values": ["balanced_accuracy", "supcon_loss_mean"]}, "metric"),
        (("classifier", "batch_size"), 0, "batch_size"),
        (("classifier", "optimizer", "lr"), {"type": "sweep", "values": [0.001, 0.002]}, "classifier|sweep"),
        (("data", "train_dataset"), {"type": "sweep", "values": ["a", "b"]}, "data|sweep"),
        (("classifier", "initialization", "encoder"), {"source": "supcon", "checkpoint": "some.pt"}, "selected_supcon|encoder"),
        (("classifier", "initialization", "encoder", "checkpoint"), "some.pt", "encoder|checkpoint"),
        (("classifier", "initialization", "head", "replace"), "other", "replace"),
        (("classifier", "initialization", "head"), {"replace": "linear", "source": "yolo_checkpoint"}, "checkpoint"),
        (("classifier", "scheduler"), {"name": "ReduceLROnPlateau", "parameters": {"mode": "max"}}, "direction|tracked_mode"),
        (("supcon", "scheduler"), {"name": "ReduceLROnPlateau", "parameters": {"mode": "min"}}, "direction|tracked_mode"),
    ],
)
def test_complete_configuration_is_validated_before_any_candidate_or_output(
    pipeline_api, pipeline_document, recorded_stages, path, value, message
):
    """Later invalid candidates and classifier settings must fail before the expensive first stage."""
    parent = pipeline_document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        config = _configuration(pipeline_api, pipeline_document)
        pipeline_api.TwoStageExperiment(config).run()
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("split", ["train_dataset", "validation_dataset"])
def test_pipeline_output_must_stay_outside_both_shared_splits(
    pipeline_api, pipeline_document, recorded_stages, split
):
    """The pipeline root and every child stage must preserve input split contents."""
    pipeline_document["output_directory"] = str(Path(pipeline_document["data"][split]) / "pipeline")
    with pytest.raises(ValueError, match="outside|output_directory"):
        _configuration(pipeline_api, pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "selection", {"tracked_metric": "supcon_loss_mean"}),
        (None, "selection_seed", 77),
        (None, "unknown", True),
        ("data", "test_dataset", "must-not-be-used"),
        ("supcon", "train_dataset", "other-training"),
        ("supcon", "validation_dataset", "other-validation"),
        ("supcon", "class_names", ["other"]),
        ("supcon", "output_directory", "other-output"),
        ("classifier", "train_dataset", "other-training"),
        ("classifier", "validation_dataset", "other-validation"),
        ("classifier", "class_names", ["other"]),
        ("classifier", "output_directory", "other-output"),
    ],
)
def test_unknown_or_conflicting_ownership_is_rejected_not_silently_ignored(
    pipeline_api, pipeline_document, section, key, value
):
    """One data owner, one output layout, no secondary selection configuration or seed."""
    target = pipeline_document if section is None else pipeline_document[section]
    target[key] = value
    with pytest.raises(ValueError, match=key):
        _configuration(pipeline_api, pipeline_document)
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("section", ["supcon", "classifier"])
def test_pipeline_tracking_settings_are_owned_by_the_root(
    pipeline_api, pipeline_document, section
):
    """Stage overrides cannot silently create unrelated tracking runs or registries."""
    pipeline_document[section]["mlflow"] = {"tracking_uri": "sqlite:///unused.db", "experiment_name": "pipeline"}
    with pytest.raises(ValueError, match="mlflow|tracking"):
        _configuration(pipeline_api, pipeline_document)


@pytest.mark.parametrize(
    "source,relative",
    [
        ("model", "config.yaml"), ("model", "selection.yaml"),
        ("model", "supcon/run_001/best.pt"),
        ("head", "classifier/best.pt"),
    ],
)
def test_generated_pipeline_artifacts_cannot_overwrite_model_or_head_inputs(
    pipeline_api, pipeline_document, recorded_stages, source, relative
):
    """Pipeline root/child publications must retain the standalone source-safety contract."""
    path = Path(pipeline_document["output_directory"]) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"source bytes must remain unchanged")
    if source == "model":
        pipeline_document["supcon"]["model_name"] = str(path)
    else:
        pipeline_document["classifier"]["initialization"]["head"] = {
            "replace": "linear", "source": "yolo_checkpoint", "checkpoint": str(path),
        }
    # Require the collision guard itself, not an unrelated attempt to load these
    # intentionally opaque bytes as model weights or architecture YAML.
    with pytest.raises(ValueError, match="collision|collid"):
        _configuration(pipeline_api, pipeline_document)
    assert recorded_stages.events == []
    assert path.read_bytes() == b"source bytes must remain unchanged"


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    rates=st.lists(st.sampled_from([0.001, 0.002, 0.003]), min_size=1, max_size=4),
    mode=st.sampled_from(["min", "max"]),
    freeze=st.booleans(),
)
def test_configuration_round_trip_preserves_sweep_order_and_classifier_policy(
    pipeline_api, pipeline_document, rates, mode, freeze
):
    """Repeated candidates and user-controlled direction/freeze settings survive serialization."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"]["values"] = rates
    document["supcon"]["tracked_mode"] = mode
    document["classifier"]["freeze_encoder"] = freeze
    config = _configuration(pipeline_api, document)
    saved = config.to_dict()
    assert saved["supcon"]["optimizer"]["lr"]["values"] == rates
    assert saved["supcon"]["tracked_mode"] == mode
    assert saved["classifier"]["freeze_encoder"] == freeze
    assert pipeline_api.TwoStageConfiguration.from_mapping(saved).to_dict() == saved


@pytest.mark.parametrize("mode", ["min", "max"])
def test_real_cpu_stages_select_best_epoch_then_train_a_portable_ce_classifier(
    pipeline_api, pipeline_document, monkeypatch, mode
):
    """Run real optimizers and source conversion; script only SupCon's validation scores."""
    document = pipeline_document
    document["supcon"]["optimizer"]["lr"]["values"] = [0.001, 0.002]
    document["supcon"]["tracked_mode"] = mode
    values = [[0.3, 0.8, 0.5], [0.2, 0.9, 0.6]] if mode == "max" else [[0.7, 0.2, 0.5], [0.8, 0.1, 0.4]]
    scores = iter((value, ce) for run in values for value, ce in zip(run, [0.1, 0.2, 0.3], strict=True))
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
    result = pipeline_api.TwoStageExperiment(_configuration(pipeline_api, document)).run()
    assert len(calls) == 6  # Exactly ordinary epoch validation; never a second scoring pass.
    assert len(result.supcon_results) == 2
    assert result.selected_supcon == result.supcon_results[1]
    assert result.selected_supcon.best_epoch == 2
    assert result.selected_supcon.best_value == values[1][1]
    source = result.selected_supcon.best_checkpoint
    source_state = torch.load(source, weights_only=True)
    assert source_state["epoch"] == 2
    assert torch.load(result.selected_supcon.last_checkpoint, weights_only=True)["epoch"] == 3
    classifier = torch.load(result.classifier_result.last_checkpoint, weights_only=True)
    assert classifier["checkpoint_type"] == "classifier" and classifier["epoch"] == 2
    assert classifier["label_names"] == source_state["label_names"] == {"mouse": 0, "rat": 1}
    assert classifier["provenance"]["encoder_checkpoint"] == str(source.resolve())
    assert classifier["provenance"]["encoder_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(int(state["step"]) == 6 for state in classifier["optimizer_state_dict"]["state"].values())
    boundary = f"network.model.{len(source_state['architecture']['backbone'])}.linear."
    assert any(not name.startswith(boundary) for name in classifier["model_state_dict"])
    # SupCon's native network prefix changes during conversion; compare every retained
    # encoder tensor by its model-state suffix rather than assuming parameter IDs.
    for name, tensor in classifier["model_state_dict"].items():
        if name.startswith(boundary):
            continue
        source_name = name.replace("network.", "full_model.model.", 1)
        assert torch.equal(tensor, source_state["model_state_dict"][source_name]), name
    restored = YoloClassifierModel.from_checkpoint(result.classifier_result.best_checkpoint)
    with torch.inference_mode():
        assert restored(torch.rand(2, 3, 16, 16)).shape == (2, 2)
    assert (result.classifier_result.output_directory / "best.model_card.yaml").is_file()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == checksum for path, checksum in images.items())


def test_two_stage_cli_runs_the_actual_cpu_pipeline(pipeline_api, pipeline_document, tmp_path):
    """The new thin entry point must execute both stages without downloads or test data."""
    document = deepcopy(pipeline_document)
    document["supcon"]["optimizer"]["lr"] = 0.001
    document["supcon"]["epochs"] = document["classifier"]["epochs"] = 1
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(document))
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_two_stage.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(path)], capture_output=True, text=True, timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Final classifier checkpoint:" in completed.stdout
    root = Path(document["output_directory"])
    assert (root / "supcon" / "run_001" / "best.pt").is_file()
    assert (root / "selection.yaml").is_file()
    restored = YoloClassifierModel.from_checkpoint(root / "classifier" / "best.pt")
    assert restored.label_names == {"mouse": 0, "rat": 1}


@pytest.mark.parametrize("key", ["mlflow", "train_dataset", "validation_dataset", "class_names", "output_directory"])
def test_whole_supcon_mapping_sweeps_cannot_bypass_shared_ownership(
    pipeline_api, pipeline_document, recorded_stages, key
):
    """Validate restrictions inside expanded mappings, including later candidates."""
    valid = deepcopy(pipeline_document["supcon"])
    valid["optimizer"]["lr"] = 0.001
    invalid = deepcopy(valid)
    overrides = {
        "mlflow": {"tracking_uri": "sqlite:///unused.db", "experiment_name": "bypass"},
        "train_dataset": "other-training", "validation_dataset": "other-validation",
        "class_names": ["other"], "output_directory": "other-output",
    }
    invalid[key] = overrides[key]
    pipeline_document["supcon"] = {"type": "sweep", "values": [valid, invalid]}
    with pytest.raises(ValueError, match=key):
        _configuration(pipeline_api, pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("section", ["supcon", "classifier"])
@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.5])
def test_every_stage_seed_is_validated_before_any_candidate(
    pipeline_api, pipeline_document, recorded_stages, section, seed
):
    """The supported Python/NumPy/PyTorch RNG intersection is an unsigned 32-bit integer."""
    value = {"type": "sweep", "values": [11, seed]} if section == "supcon" else seed
    pipeline_document[section]["seed"] = value
    with pytest.raises(ValueError, match="seed"):
        _configuration(pipeline_api, pipeline_document)
    assert recorded_stages.events == []
    assert not Path(pipeline_document["output_directory"]).exists()


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_failed_rerun_invalidates_previous_success_without_pruning_artifacts(
    pipeline_api, pipeline_document, recorded_stages, fail_at
):
    """A new failed invocation cannot retain an older completed-classifier summary."""
    first = pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
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
        pipeline_api.TwoStageExperiment(_configuration(pipeline_api, pipeline_document)).run()
    assert recorded_stages.events == ["supcon"] * (fail_at + 1)
    assert not recorded_stages.classifier
    summary = yaml.safe_load(summary_path.read_text())
    assert summary.get("classifier_result") is None
    assert summary.get("selected_supcon") is None
    assert not summary.get("supcon_results")
    assert first.classifier_result.best_checkpoint.read_bytes() == classifier_bytes
