"""Two-stage pipeline: preflight over every candidate encoder and the CE handoff."""

from pathlib import Path

import mlflow
import pytest
import torch
import yaml
from classifier_task_test_support import (
    pipeline_document,
    sample_inputs,
    tagged_document,
)
from study_test_support import grid

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}


from classifier_task_test_support import (
    MODULE,
    write_class_folders,
)


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _plan(document):
    """Prepare everything that runs before the first stage trains."""
    from smartrodent.two_stage_training import TwoStageConfiguration, TwoStagePlan

    return TwoStagePlan(TwoStageConfiguration.from_mapping(document))


def _searched_encoders(tmp_path, **classifier):
    """A SupCon study over two encoder classes and a classifier section to check."""
    document = pipeline_document(tmp_path)
    document["supcon"]["encoder"]["name"] = "__ENCODER__"
    document["classifier"].update(classifier)
    document["tuning"] = {"supcon": grid()}
    choices = [f"{MODULE}:TinyEncoder", f"{MODULE}:LayeredEncoder"]
    return tagged_document(
        document, ENCODER=f"!suggest-categorical {{choices: [{', '.join(choices)}]}}"
    )


def test_freeze_prefix_missing_from_any_candidate_encoder_fails_before_training(
    tmp_path,
):
    """``first`` exists in LayeredEncoder only, yet every candidate is checked."""
    document = _searched_encoders(
        tmp_path, freeze={"prefixes": ["first"], "batchnorm": "eval"}
    )
    with pytest.raises(ValueError, match="first"):
        _plan(document)
    assert not (tmp_path / "pipeline").exists()


def test_reinitialization_prefix_missing_from_any_candidate_fails_before_training(
    tmp_path,
):
    """Random reinitialization prefixes are preflighted against every candidate too."""
    document = _searched_encoders(
        tmp_path, reinitialization={"prefixes": ["third"], "source": "random"}
    )
    with pytest.raises(ValueError, match="third"):
        _plan(document)


def test_native_export_request_is_checked_against_every_candidate_encoder(tmp_path):
    """TinyEncoder cannot export natively, so the pipeline fails in preflight."""
    document = pipeline_document(tmp_path)
    document["classifier"]["native_checkpoint"] = True
    with pytest.raises(ValueError, match="native"):
        _plan(document)


def test_full_head_is_checked_against_every_candidate_encoder(tmp_path):
    """TinyEncoder has no post-pooling renewal, so ``head: full`` fails in preflight."""
    document = pipeline_document(tmp_path)
    document["classifier"]["head"] = "full"
    with pytest.raises(ValueError, match="post-pooling"):
        _plan(document)


def test_encoder_is_owned_by_the_pipeline(tmp_path):
    """The CE encoder is a reserved key; the pipeline fills it in."""
    document = pipeline_document(tmp_path)
    _plan(document)  # a classifier section without an encoder is the valid form
    document["classifier"]["encoder"] = {"checkpoint": "elsewhere.ckpt"}
    with pytest.raises(ValueError, match="encoder"):
        _plan(document)


def test_aliased_classifier_artifact_in_the_pipeline_is_rejected(tmp_path):
    """The pipeline plan applies the input guard to both stage directories."""
    train, _ = write_class_folders(tmp_path)
    document = pipeline_document(tmp_path)
    classifier_directory = Path(document["output_directory"]) / "classifier"
    classifier_directory.mkdir(parents=True)
    (classifier_directory / "metrics.csv").symlink_to(next(train.rglob("*.png")))

    with pytest.raises(ValueError, match="dataset|hardlink"):
        _plan(document)


def test_classifier_trains_on_the_winning_supcon_best_checkpoint(tmp_path):
    """With every encoder parameter frozen, CE features are the winner's features."""
    from smartrodent.classifier_task import ClassifierTask
    from smartrodent.supcon_task import SupConTask
    from smartrodent.two_stage_training import (
        TwoStageConfiguration,
        TwoStageExperiment,
    )

    document = pipeline_document(tmp_path)
    document["supcon"]["optimizer"]["parameters"]["lr"] = "__LR__"
    document["classifier"]["freeze"] = {"prefixes": ["linear"], "batchnorm": "eval"}
    document["tuning"] = {"supcon": grid()}
    document = tagged_document(
        document, LR="!suggest-categorical {choices: [0.0, 0.5]}"
    )

    result = TwoStageExperiment(TwoStageConfiguration.from_mapping(document)).run()

    def features(checkpoint, task_class):
        """Encoder features of one checkpoint on fixed inputs."""
        task = task_class.load_from_checkpoint(checkpoint, map_location="cpu")
        with torch.inference_mode():
            return task.encoder.eval()(sample_inputs())

    winner = features(result.selected_supcon.best_checkpoint, SupConTask)
    classifier = features(result.selected_classifier.best_checkpoint, ClassifierTask)
    torch.testing.assert_close(classifier, winner)
    others = [
        features(item.best_checkpoint, SupConTask)
        for item in result.supcon_results
        if item is not result.selected_supcon
    ]
    assert others
    assert not any(torch.allclose(classifier, other) for other in others)


@pytest.mark.parametrize("alias", ["symbolic", "hard"])
@pytest.mark.parametrize("name", ["config.yaml", "selection.yaml"])
def test_pipeline_root_files_cannot_alias_dataset_files(tmp_path, name, alias):
    """The root's own config and summary are written, so they are guarded too."""
    train, _ = write_class_folders(tmp_path)
    image = next(train.rglob("*.png"))
    document = pipeline_document(tmp_path)
    root = Path(document["output_directory"])
    root.mkdir(parents=True)
    if alias == "symbolic":
        (root / name).symlink_to(image)
    else:
        (root / name).hardlink_to(image)

    with pytest.raises(ValueError, match="dataset|hardlink"):
        _plan(document)
    assert image.exists()


def test_malformed_classifier_card_template_fails_before_any_training(tmp_path):
    """The classifier's template is read during preparation, not after SupCon."""
    template = tmp_path / "bad-template.yaml"
    template.write_text("model: 1\n")
    document = pipeline_document(tmp_path)
    document["classifier"]["model_card_template"] = str(template)

    with pytest.raises(ValueError, match="model_card_template"):
        _plan(document)
    assert not (tmp_path / "pipeline").exists()


def test_pipeline_with_a_classifier_study_registers_and_reports_a_champion(
    tmp_path, mlflow_store
):
    """A tracked pipeline runs a classifier study and records the selection."""
    from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment

    document = pipeline_document(tmp_path, mlflow=mlflow_store.configuration)
    document["classifier"]["optimizer"]["parameters"]["lr"] = "__LR__"
    document["tuning"] = {"classifier": grid(study_name="ce")}
    document = tagged_document(
        document, LR="!suggest-categorical {choices: [0.0, 0.05]}"
    )
    configuration = TwoStageConfiguration.from_mapping(document)

    result = TwoStageExperiment(configuration).run()

    root = Path(document["output_directory"])
    summary = yaml.safe_load((root / "selection.yaml").read_text())
    assert summary["registered_champion_version"] is not None
    assert result.mlflow_run_id is not None
    assert configuration.to_dict()["tuning"]["classifier"]["study_name"] == "ce"
    assert mlflow.active_run() is None


def test_pipeline_requires_a_study_in_its_tuning_section(tmp_path):
    """An empty tuning section is a mistake, not a request for single runs."""
    from smartrodent.two_stage_training import TwoStageConfiguration

    with pytest.raises(ValueError, match="at least one stage"):
        TwoStageConfiguration.from_mapping(pipeline_document(tmp_path, tuning={}))


def test_rerunning_a_pipeline_directory_replaces_its_summary(tmp_path):
    """The earlier selection is invalidated, then rewritten by the new run."""
    from smartrodent.two_stage_training import TwoStageConfiguration, TwoStageExperiment

    document = pipeline_document(tmp_path)
    for _ in range(2):
        TwoStageExperiment(TwoStageConfiguration.from_mapping(document)).run()
    summary = Path(document["output_directory"]) / "selection.yaml"
    assert yaml.safe_load(summary.read_text())["selected_classifier"] is not None
