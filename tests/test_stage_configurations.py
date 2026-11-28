"""Stage configurations, their shared base, stages, and the input-path guard wiring."""

import pytest
import yaml
from classifier_task_test_support import (
    classifier_document,
    pipeline_document,
    supcon_document,
)

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.stage_configurations import (
    ClassifierConfiguration,
    SupConConfiguration,
)
from smartrodent.stages import CLASSIFIER, SUPCON

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

import dataclasses

from classifier_task_test_support import (
    encoder_spec,
)

DOCUMENTS = {"supcon": supcon_document, "classifier": classifier_document}


def _parse(stage: str, document: dict):
    """Parse a stage document with the stage's configuration class."""
    from smartrodent.stage_configurations import (
        ClassifierConfiguration,
        SupConConfiguration,
    )

    configuration = {
        "supcon": SupConConfiguration,
        "classifier": ClassifierConfiguration,
    }[stage]
    return configuration.from_mapping(document)


@pytest.fixture(params=sorted(DOCUMENTS))
def stage(request):
    """Run each test for both stage configurations."""
    return request.param


def test_both_stages_share_the_optimizer_scheduler_stopping_and_trainer_sections(
    tmp_path, stage
):
    """The common base parses the four sections identically for either stage."""
    document = DOCUMENTS[stage](
        tmp_path,
        scheduler=[{"name": "StepLR", "parameters": {"step_size": 1}}],
        early_stopping={"patience": 2, "min_delta": 0.0},
    )
    configuration = _parse(stage, document)

    assert configuration.optimizer.name == "SGD"
    assert [phase.name for phase in configuration.scheduler] == ["StepLR"]
    assert configuration.early_stopping == {"patience": 2, "min_delta": 0.0}
    assert configuration.trainer["max_epochs"] == 1


@pytest.mark.parametrize("key", ["callbacks", "logger", "default_root_dir"])
def test_trainer_settings_owned_by_the_runner_are_rejected_in_both_stages(
    tmp_path, stage, key
):
    """Misconfiguration fails when the document is parsed, not when training starts."""
    document = DOCUMENTS[stage](tmp_path)
    document["trainer"][key] = None
    with pytest.raises(ValueError, match=key):
        _parse(stage, document)


def test_trainer_must_state_max_epochs_in_both_stages(tmp_path, stage):
    """Lightning's silent default epoch limit is never used."""
    document = DOCUMENTS[stage](tmp_path)
    del document["trainer"]["max_epochs"]
    with pytest.raises(ValueError, match="max_epochs"):
        _parse(stage, document)


def test_stages_bundle_their_configuration_and_experiment():
    """``SUPCON`` and ``CLASSIFIER`` are frozen values naming what differs per stage."""
    from smartrodent.stage_configurations import (
        ClassifierConfiguration,
        SupConConfiguration,
    )

    assert SUPCON.configuration is SupConConfiguration
    assert CLASSIFIER.configuration is ClassifierConfiguration
    with pytest.raises(dataclasses.FrozenInstanceError):
        SUPCON.experiment = CLASSIFIER.experiment


def test_output_directory_inside_an_input_split_is_rejected(tmp_path, stage):
    """Configurations call the input guard, as the old trainers did."""
    document = DOCUMENTS[stage](tmp_path)
    document["output_directory"] = f"{document['train_dataset']}/run"
    with pytest.raises(ValueError, match="outside the input datasets"):
        _parse(stage, document)


@pytest.mark.parametrize("alias", ["symbolic", "hard"])
def test_an_artifact_aliasing_a_dataset_file_is_rejected(tmp_path, stage, alias):
    """A symlink or hard link at a writer's path cannot reach a training image."""
    document = DOCUMENTS[stage](tmp_path)
    image = next((tmp_path / "data" / "train").rglob("*.png"))
    output = tmp_path / "linked"
    output.mkdir()
    document["output_directory"] = str(output)
    if alias == "symbolic":
        (output / "metrics.csv").symlink_to(image)
    else:
        (output / "metrics.csv").hardlink_to(image)

    with pytest.raises(ValueError, match="dataset|hardlink"):
        _parse(stage, document)
    assert image.exists()


def test_classifier_cannot_write_over_the_encoder_checkpoint_it_reads(tmp_path):
    """The encoder source is a protected input of the classifier stage."""
    output = tmp_path / "shared"
    output.mkdir()
    (output / "best.ckpt").write_bytes(b"encoder")
    document = classifier_document(
        tmp_path,
        output_directory=str(output),
        encoder={"checkpoint": str(output / "best.ckpt")},
    )

    with pytest.raises(ValueError, match="collides with source"):
        _parse("classifier", document)


def test_native_export_request_fails_before_the_run_when_the_encoder_cannot_export(
    tmp_path,
):
    """Validation of the experiment, not training, reports the missing capability."""
    from smartrodent.classifier_training import ClassifierExperiment

    configuration = _parse(
        "classifier", classifier_document(tmp_path, native_checkpoint=True)
    )
    with pytest.raises(ValueError, match="native"):
        ClassifierExperiment(configuration)
    assert not configuration.output_directory.exists()


def test_native_export_request_is_accepted_for_a_capable_encoder(tmp_path):
    """An encoder with ``to_native`` passes the same validation."""
    from smartrodent.classifier_training import ClassifierExperiment

    configuration = _parse(
        "classifier",
        classifier_document(
            tmp_path,
            native_checkpoint=True,
            encoder=encoder_spec("NativeTinyEncoder"),
        ),
    )
    ClassifierExperiment(configuration)


def test_unavailable_tracked_metric_is_rejected(tmp_path):
    """An alias that is not configured cannot be tracked."""
    configuration = ClassifierConfiguration.from_mapping(
        classifier_document(
            tmp_path, tracked_metric="f1", validation_metrics={"other": METRICS["f1"]}
        )
    )
    with pytest.raises(ValueError, match="not available"):
        ClassifierExperiment(configuration)


def test_validation_objectives_need_a_validation_split(tmp_path):
    """Scores cannot be tracked without validation data."""
    document = classifier_document(
        tmp_path, validation_metrics=METRICS, validation_dataset=None
    )
    with pytest.raises(ValueError, match="validation_dataset"):
        ClassifierConfiguration.from_mapping(document)


def test_stage_configurations_reject_unusable_batch_layouts(tmp_path):
    """Batches must fit the classes and the epoch."""
    with pytest.raises(ValueError, match="classes_per_batch"):
        SupConConfiguration.from_mapping(supcon_document(tmp_path, classes_per_batch=4))
    with pytest.raises(ValueError, match="full batch"):
        SupConConfiguration.from_mapping(supcon_document(tmp_path, samples_per_epoch=1))


def test_from_yaml_rejects_search_tags(tmp_path):
    """A single run needs concrete values."""
    document = classifier_document(
        tmp_path, optimizer={"name": "SGD", "parameters": {"lr": "__LR__"}}
    )
    text = yaml.safe_dump(document).replace(
        "__LR__", "!suggest-categorical {choices: [0.1, 0.2]}"
    )
    path = tmp_path / "tagged.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match="tuning"):
        ClassifierConfiguration.from_yaml(path)


def test_from_yaml_reads_concrete_stage_and_pipeline_documents(tmp_path):
    """Both loaders accept concrete documents written to a file."""
    from smartrodent.two_stage_training import TwoStageConfiguration

    stage = tmp_path / "stage.yaml"
    stage.write_text(yaml.safe_dump(classifier_document(tmp_path)))
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(pipeline_document(tmp_path)))

    assert ClassifierConfiguration.from_yaml(stage).batch_size == 4
    assert TwoStageConfiguration.from_yaml(pipeline).output_directory.name == "pipeline"
