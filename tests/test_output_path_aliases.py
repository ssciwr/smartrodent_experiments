"""Guard every artifact of the classifier and pipeline writers against input aliases.

The metrics-CSV symlink and hard-link case, directory overlap, dataset containment
and source collision were already covered for a single SupCon run in
``test_lightning_training_support.py``. These tests extend that coverage to the
artifacts only the classifier writes (native checkpoint, ONNX file, cards) and to
the two-directory layout of the pipeline.
"""

import pytest

from smartrodent.output_paths import RunArtifacts, validate_output_paths

CLASSIFIER_ONLY = ["native_checkpoint", "onnx_model", "card_yaml", "card_markdown"]


def _alias(path, target, kind):
    """Make ``path`` another name for ``target``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symbolic":
        path.symlink_to(target)
    else:
        path.hardlink_to(target)


@pytest.fixture
def dataset(tmp_path):
    """One input image inside a training split."""
    split = tmp_path / "train"
    split.mkdir()
    (split / "image.png").write_bytes(b"input")
    return split


@pytest.mark.parametrize("alias", ["symbolic", "hard"])
@pytest.mark.parametrize("artifact", CLASSIFIER_ONLY)
def test_classifier_artifacts_cannot_alias_dataset_files(
    tmp_path, dataset, artifact, alias
):
    """Native, ONNX and card paths are guarded like the checkpoint paths."""
    artifacts = RunArtifacts(tmp_path / "run", onnx=True, native=True)
    _alias(getattr(artifacts, artifact), dataset / "image.png", alias)

    with pytest.raises(ValueError, match="dataset|hardlink"):
        validate_output_paths([artifacts], datasets=[dataset])
    assert (dataset / "image.png").read_bytes() == b"input"


@pytest.mark.parametrize("alias", ["symbolic", "hard"])
def test_pipeline_artifacts_of_the_second_stage_cannot_alias_inputs(
    tmp_path, dataset, alias
):
    """Both stage directories are checked, not only the first one."""
    supcon = RunArtifacts(tmp_path / "pipeline" / "supcon")
    classifier = RunArtifacts(tmp_path / "pipeline" / "classifier", native=True)
    _alias(classifier.native_checkpoint, dataset / "image.png", alias)

    with pytest.raises(ValueError, match="dataset|hardlink"):
        validate_output_paths([supcon, classifier], datasets=[dataset])


def test_classifier_cannot_write_over_the_supcon_checkpoint_it_reads(tmp_path):
    """The classifier's encoder source is protected from its own best.ckpt."""
    supcon = RunArtifacts(tmp_path / "pipeline" / "supcon")
    classifier = RunArtifacts(tmp_path / "pipeline" / "supcon_again")
    supcon.directory.mkdir(parents=True)
    classifier.directory.symlink_to(supcon.directory)

    with pytest.raises(ValueError, match="overlap|collides"):
        validate_output_paths([classifier], sources=[supcon.best_checkpoint])


def test_disjoint_pipeline_stages_pass(tmp_path, dataset):
    """Sibling stage directories with every optional artifact are accepted."""
    validate_output_paths(
        [
            RunArtifacts(tmp_path / "pipeline" / "supcon", onnx=True),
            RunArtifacts(tmp_path / "pipeline" / "classifier", onnx=True, native=True),
        ],
        datasets=[dataset],
        sources=[tmp_path / "encoder.ckpt"],
    )
