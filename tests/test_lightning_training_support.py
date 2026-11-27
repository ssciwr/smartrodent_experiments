"""Behavior shared by the Lightning training stages."""

import numpy as np
import pytest
import torch
from PIL import Image

from smartrodent.components import ComponentSpec


def test_training_augmentations_run_in_order_and_leave_validation_deterministic():
    """An ordered crop and pad give the training image a border of black pixels."""
    from smartrodent.transforms import ImageTransforms, TransformSettings

    transforms = ImageTransforms(
        TransformSettings(
            image_size=4,
            augmentations=[
                {"name": "CenterCrop", "parameters": {"size": 2}},
                {"name": "Pad", "parameters": {"padding": 1}},
            ],
        )
    )
    image = Image.new("RGB", (5, 6), (255, 128, 0))
    rgb = torch.tensor([1.0, 128 / 255, 0.0])[:, None, None]
    expected_training = torch.zeros((3, 4, 4))
    expected_training[:, 1:3, 1:3] = rgb
    torch.testing.assert_close(transforms.training()(image), expected_training)

    for _ in range(2):
        torch.testing.assert_close(transforms.validation()(image), rgb.expand(3, 4, 4))


@pytest.fixture
def optimization():
    """Use an SGD optimizer on a single trainable parameter."""
    from smartrodent.experiment_results import Objective
    from smartrodent.training_optimization import configure_optimizers

    parameter = torch.nn.Parameter(torch.ones(1))
    spec = ComponentSpec("SGD", {"lr": 0.1})
    objective = Objective("balanced_accuracy", "max")

    def configure(phases):
        """Configure the public Lightning optimizer interface for these phases."""
        return configure_optimizers([parameter], spec, phases, objective)

    return configure


def test_empty_scheduler_leaves_the_learning_rate_constant(optimization):
    """Without phases, Lightning receives the optimizer alone."""
    optimizer = optimization([])
    assert isinstance(optimizer, torch.optim.Optimizer)
    optimizer.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_scheduler_phases_switch_at_the_configured_epoch(optimization):
    """Two warmup epochs lead into the next scheduler at the original rate."""
    from smartrodent.training_optimization import SchedulerPhase

    configured = optimization(
        [
            SchedulerPhase(
                "LinearLR", {"start_factor": 0.25, "total_iters": 2}, epochs=2
            ),
            SchedulerPhase("StepLR", {"step_size": 1, "gamma": 0.5}),
        ]
    )
    optimizer = configured["optimizer"]
    schedule = configured["lr_scheduler"]
    assert schedule["interval"] == "epoch"
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        optimizer.step()
        schedule["scheduler"].step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates == pytest.approx([0.025, 0.0625, 0.1, 0.05])


def test_a_single_scheduler_can_advance_with_each_optimizer_step(optimization):
    """OneCycleLR uses Lightning's step cadence and changes rate between updates."""
    from smartrodent.training_optimization import SchedulerPhase

    configured = optimization(
        [
            SchedulerPhase(
                "OneCycleLR", {"max_lr": 0.2, "total_steps": 8}, interval="step"
            )
        ]
    )
    schedule = configured["lr_scheduler"]
    optimizer = configured["optimizer"]
    assert schedule["interval"] == "step"
    before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    schedule["scheduler"].step()
    assert optimizer.param_groups[0]["lr"] > before


def test_plateau_uses_the_objective_metric_and_direction(optimization):
    """A maximizing score keeps the original rate on an improvement."""
    from smartrodent.training_optimization import SchedulerPhase

    configured = optimization(
        [SchedulerPhase("ReduceLROnPlateau", {"patience": 0, "factor": 0.5})]
    )
    optimizer = configured["optimizer"]
    schedule = configured["lr_scheduler"]
    assert schedule["monitor"] == "balanced_accuracy"
    schedule["scheduler"].step(0.6)
    schedule["scheduler"].step(0.8)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    schedule["scheduler"].step(0.7)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_plateau_cannot_be_part_of_a_scheduler_chain(optimization):
    """A metric-driven scheduler requires a single epoch-based phase."""
    from smartrodent.training_optimization import SchedulerPhase

    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        optimization(
            [
                SchedulerPhase("StepLR", {"step_size": 1}, epochs=2),
                SchedulerPhase("ReduceLROnPlateau"),
            ]
        )


def test_metric_specs_keep_the_legacy_comparison_definition():
    """A named scorer computes the requested statistic and keeps its registry identity."""
    from smartrodent.metrics import ValidationMetrics

    scores = ValidationMetrics(
        {
            "macro_f1": ComponentSpec(
                "f1_score", {"average": "macro", "zero_division": 0}
            )
        }
    )
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 1])

    assert scores.score(targets, predictions) == {"macro_f1": pytest.approx(11 / 15)}
    assert scores.definition("macro_f1") == {
        "function": "f1_score",
        "average": "macro",
        "zero_division": 0,
    }


@pytest.mark.parametrize("alias", ["symbolic", "hard"])
def test_run_artifact_paths_cannot_alias_dataset_files(tmp_path, alias):
    """The path supplied to the metrics writer is also protected by the input guard."""
    from smartrodent.output_paths import RunArtifacts, validate_output_paths

    dataset = tmp_path / "train"
    dataset.mkdir()
    image = dataset / "image.png"
    image.write_bytes(b"input")
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.directory.mkdir()
    if alias == "symbolic":
        artifacts.metrics_csv.symlink_to(image)
    else:
        artifacts.metrics_csv.hardlink_to(image)

    with pytest.raises(ValueError, match="dataset|hardlink"):
        validate_output_paths([artifacts], datasets=[dataset])
    assert image.read_bytes() == b"input"


@pytest.mark.parametrize(
    "phases,message",
    [
        (
            [
                ("StepLR", {"step_size": 1}, 1, "step"),
                ("StepLR", {"step_size": 1}, None, "epoch"),
            ],
            "epoch intervals",
        ),
        ([("ReduceLROnPlateau", {}, None, "step")], "ReduceLROnPlateau"),
        ([("StepLR", {"step_size": 1}, 2, "epoch")], "omit epochs"),
        (
            [
                ("StepLR", {"step_size": 1}, None, "epoch"),
                ("StepLR", {"step_size": 1}, None, "epoch"),
            ],
            "requires epochs",
        ),
    ],
)
def test_invalid_scheduler_phase_layouts_are_rejected(optimization, phases, message):
    """Chains need epoch intervals and durations for all but the last phase."""
    from smartrodent.training_optimization import SchedulerPhase

    configured = [
        SchedulerPhase(name, parameters, epochs=epochs, interval=interval)
        for name, parameters, epochs, interval in phases
    ]
    with pytest.raises(ValueError, match=message):
        optimization(configured)


def test_metric_options_cannot_shadow_the_registry_identity_key():
    """`function` names the scorer in published definitions, so it is reserved."""
    from smartrodent.metrics import ValidationMetrics

    with pytest.raises(ValueError, match="reserved"):
        ValidationMetrics({"bad": ComponentSpec("f1_score", {"function": "x"})})


def test_a_scorer_that_does_not_return_a_scalar_is_rejected_before_training():
    """A confusion matrix is not a class-ID score."""
    from smartrodent.metrics import ValidationMetrics

    with pytest.raises(ValueError, match="cannot score class IDs"):
        ValidationMetrics({"matrix": ComponentSpec("confusion_matrix")})


def test_run_artifacts_list_enabled_files_and_expose_optional_ones_as_none(tmp_path):
    """Optional ONNX and native paths exist only when enabled."""
    from smartrodent.output_paths import RunArtifacts

    plain = RunArtifacts(tmp_path)
    full = RunArtifacts(tmp_path, onnx=True, native=True)

    assert plain.onnx_model is None and plain.native_checkpoint is None
    assert len(plain.paths) == 7
    assert full.onnx_model in full.paths and full.native_checkpoint in full.paths
    assert set(plain.paths) < set(full.paths)


def test_overlapping_output_directories_are_rejected(tmp_path):
    """Nested stage directories could overwrite each other's artifacts."""
    from smartrodent.output_paths import RunArtifacts, validate_output_paths

    runs = [RunArtifacts(tmp_path / "a"), RunArtifacts(tmp_path / "a" / "b")]
    with pytest.raises(ValueError, match="overlap"):
        validate_output_paths(runs)


def test_output_directory_inside_a_dataset_is_rejected(tmp_path):
    """Writers must stay outside the data they read."""
    from smartrodent.output_paths import RunArtifacts, validate_output_paths

    with pytest.raises(ValueError, match="outside the input datasets"):
        validate_output_paths(
            [RunArtifacts(tmp_path / "train" / "run")], datasets=[tmp_path / "train"]
        )


def test_output_artifact_cannot_overwrite_a_source_file(tmp_path):
    """A named input file is protected from the stage's own artifacts."""
    from smartrodent.output_paths import RunArtifacts, validate_output_paths

    artifacts = RunArtifacts(tmp_path / "run")
    with pytest.raises(ValueError, match="collides with source"):
        validate_output_paths([artifacts], sources=[artifacts.config])


def test_disjoint_outputs_pass_validation(tmp_path):
    """Sibling directories away from inputs are accepted."""
    from smartrodent.output_paths import RunArtifacts, validate_output_paths

    validate_output_paths(
        [RunArtifacts(tmp_path / "a"), RunArtifacts(tmp_path / "b")],
        datasets=[tmp_path / "train"],
        sources=[tmp_path / "model.pt"],
    )
