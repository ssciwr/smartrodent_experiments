"""One StageStudy serves both stages: encoder identity on resume, no partial promotion."""

import pytest
import torch
import yaml
from classifier_task_test_support import (
    classifier_document,
    supcon_document,
    tagged_document,
    train_supcon,
)
from study_test_support import grid, study_report

from smartrodent.stages import CLASSIFIER, SUPCON
from smartrodent.training_studies import StageStudy, is_study_configuration

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

import shutil

from classifier_task_test_support import (
    MODULE,
)
from mlflow.exceptions import MlflowException

LR_CHOICES = "!suggest-categorical {choices: [0.0, 0.05]}"


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    """Keep short CPU runs fast and restore the caller's thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def encoders(tmp_path):
    """Two SupCon checkpoints with different bytes."""
    first = train_supcon(tmp_path / "a")[1].best_checkpoint
    second = train_supcon(tmp_path / "b")[1].best_checkpoint
    assert first.read_bytes() != second.read_bytes()
    return first, second


def _study(tmp_path, checkpoint, **tuning):
    """A classifier study over the learning rate on a fixed encoder checkpoint."""
    from smartrodent.stages import CLASSIFIER
    from smartrodent.training_studies import StageStudy

    document = classifier_document(
        tmp_path / "a",
        encoder={"checkpoint": str(checkpoint)},
        optimizer={"name": "SGD", "parameters": {"lr": "__LR__"}},
        output_directory=str(tmp_path / "study"),
    )
    document["tuning"] = grid(
        storage=f"sqlite:///{tmp_path / 'optuna.db'}", load_if_exists=True, **tuning
    )
    return StageStudy(CLASSIFIER, tagged_document(document, LR=LR_CHOICES))


def test_resume_with_a_different_encoder_checkpoint_is_refused(tmp_path, encoders):
    """Trials trained on other encoder bytes are not comparable."""
    first, second = encoders
    _study(tmp_path, first).run()

    with pytest.raises(ValueError, match="encoder checkpoints"):
        _study(tmp_path, second).run()
    assert len(study_report(tmp_path / "study")["trials"]) == 2


def test_resume_keeps_the_identity_when_an_unchanged_encoder_file_moves(
    tmp_path, encoders
):
    """Identity is the file's content, not its path."""
    first, _ = encoders
    _study(tmp_path, first).run()
    moved = tmp_path / "moved.ckpt"
    shutil.copy(first, moved)

    result = _study(tmp_path, moved).run()

    # The earlier invocation's trials are kept; resuming may add more.
    assert len(result.results) >= 2


def test_supcon_and_classifier_studies_are_the_same_class(tmp_path):
    """No stage-specific study subclasses remain."""
    from smartrodent import training_studies
    from smartrodent.stages import SUPCON
    from smartrodent.training_studies import StageStudy

    assert not hasattr(training_studies, "SupConStudy")
    assert not hasattr(training_studies, "ClassifierStudy")
    document = supcon_document(
        tmp_path,
        output_directory=str(tmp_path / "study"),
        optimizer={"name": "SGD", "parameters": {"lr": "__LR__"}},
    )
    document["tuning"] = grid()

    result = StageStudy(SUPCON, tagged_document(document, LR=LR_CHOICES)).run()

    assert result.selected.tracked_metric == "supcon_loss_mean"
    assert result.selected.best_value == min(r.best_value for r in result.results)


def test_failed_study_cannot_promote_a_partial_winner(tmp_path, mlflow_store):
    """An unexpected failure in a later trial leaves the champion alias unset."""
    from smartrodent.stages import CLASSIFIER
    from smartrodent.training_studies import StageStudy

    document = classifier_document(
        tmp_path / "a",
        optimizer={"name": "SGD", "parameters": {"lr": "__LR__"}},
        output_directory=str(tmp_path / "study"),
        mlflow=mlflow_store.configuration,
    )
    # Trial 0 completes and registers; the pruner then fails the second trial.
    document["tuning"] = grid(pruner={"name": f"{MODULE}:ExplodingPruner"})
    study = StageStudy(CLASSIFIER, tagged_document(document, LR=LR_CHOICES))

    with pytest.raises(RuntimeError, match="pruner failure"):
        study.run()

    states = [t["state"] for t in study_report(tmp_path / "study")["trials"]]
    assert states == ["COMPLETE", "FAIL"]
    client = mlflow_store.client
    name = mlflow_store.registry_name
    assert len(client.search_model_versions(f"name='{name}'")) == 1
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(name, "approved")


def _study_document(tmp_path, **changes):
    """A SupCon study document searching the learning rate."""
    document = supcon_document(
        tmp_path,
        output_directory=str(tmp_path / "study"),
        optimizer={"name": "SGD", "parameters": {"lr": "__LR__"}},
    )
    document["tuning"] = grid()
    return (
        tagged_document(document, LR="!suggest-categorical {choices: [0.0, 0.05]}")
        | changes
    )


def test_study_document_must_be_a_complete_mapping(tmp_path):
    """A study needs a mapping, a tuning section and an output directory."""
    with pytest.raises(ValueError, match="mapping"):
        StageStudy(SUPCON, [])
    document = _study_document(tmp_path)
    without_tuning = {k: v for k, v in document.items() if k != "tuning"}
    with pytest.raises(ValueError, match="tuning"):
        StageStudy(SUPCON, without_tuning)
    without_output = {k: v for k, v in document.items() if k != "output_directory"}
    with pytest.raises(ValueError, match="output_directory"):
        StageStudy(SUPCON, without_output)


def test_tracked_study_opens_one_parent_run_for_its_trials(tmp_path, tracking_store):
    """The parent run and both trial runs are recorded; the winner is reported."""
    document = _study_document(tmp_path, mlflow=tracking_store.configuration)
    study = StageStudy(SUPCON, document)
    study.preflight()
    result = study.run()

    runs = tracking_store.client.search_runs([tracking_store.experiment_id])
    assert len(runs) == 3
    assert result.selected.tracked_metric == "supcon_loss_mean"
    assert study_report(tmp_path / "study")["selected_trial"] is not None


def test_study_from_yaml_and_detection(tmp_path):
    """The study file helpers read the tuning section."""
    text = yaml.safe_dump(
        supcon_document(tmp_path, output_directory=str(tmp_path / "s"))
    )
    plain = tmp_path / "plain.yaml"
    plain.write_text(text)
    assert not is_study_configuration(plain)
    tagged = tmp_path / "tagged.yaml"
    tagged.write_text(
        text.replace("lr: 0.05", "lr: !suggest-categorical {choices: [0.0, 0.05]}")
        + yaml.safe_dump({"tuning": grid()})
    )
    assert is_study_configuration(tagged)
    assert StageStudy.from_yaml(SUPCON, tagged).reference.output_directory.name


def test_inputs_inside_the_study_directory_are_rejected(tmp_path):
    """Trials write under the study directory, so no input may live there."""
    document = _study_document(tmp_path)
    document["model_card_template"] = str(tmp_path / "study" / "template.yaml")
    with pytest.raises(ValueError, match="study output directory"):
        StageStudy(SUPCON, document)


def test_invalid_sampled_combinations_fail_one_trial_not_the_study(tmp_path):
    """A freeze prefix missing from one sampled encoder is a trial-level error."""
    train_supcon(tmp_path)
    document = classifier_document(
        tmp_path,
        output_directory=str(tmp_path / "study"),
        encoder={"name": "__ENCODER__"},
        freeze={"prefixes": ["first"], "batchnorm": "eval"},
    )
    document["tuning"] = grid()
    study = StageStudy(
        CLASSIFIER,
        tagged_document(
            document,
            ENCODER="!suggest-categorical {choices: "
            "[classifier_task_test_support:TinyEncoder, "
            "classifier_task_test_support:LayeredEncoder]}",
        ),
    )
    result = study.run()

    states = {t["state"] for t in study_report(tmp_path / "study")["trials"]}
    assert states == {"COMPLETE", "FAIL"}
    assert len(result.results) == 1
