"""Freeze and reinitialize encoder parameters by dotted name prefix."""

import pytest
import torch
from torch import nn

ONNX = {"dynamic": True, "simplify": False}
METRICS = {"f1": {"name": "f1_score", "parameters": {"average": "macro"}}}

from hypothesis import given
from hypothesis import strategies as st


@given(
    index=st.integers(min_value=1, max_value=999),
    trailing_digit=st.integers(min_value=0, max_value=9),
    weight_only=st.booleans(),
)
def test_prefix_matches_complete_parameter_name_segments(
    index, trailing_digit, weight_only
):
    """A module name or exact parameter selects only its own parameters."""
    from smartrodent.encoders import FreezePolicy

    sibling = f"{index}{trailing_digit}"
    encoder = nn.Module()
    encoder.model = nn.ModuleDict(
        {str(index): nn.Linear(2, 2), sibling: nn.Linear(2, 2)}
    )
    prefix = f"model.{index}.weight" if weight_only else f"model.{index}"

    FreezePolicy(prefixes=(prefix,), batchnorm="eval").apply(encoder)

    assert not encoder.model[str(index)].weight.requires_grad
    assert encoder.model[str(index)].bias.requires_grad == weight_only
    assert all(
        parameter.requires_grad for parameter in encoder.model[sibling].parameters()
    )


@pytest.mark.parametrize("policy", ["freeze", "reinitialization"])
def test_unmatched_prefix_fails_preflight(policy):
    """Every prefix has to select a parameter, even alongside a valid prefix."""
    from smartrodent.encoders import FreezePolicy, Reinitialization

    encoder = nn.Sequential(nn.Linear(2, 2))
    policies = {
        "freeze": FreezePolicy(prefixes=("0", "9"), batchnorm="eval"),
        "reinitialization": Reinitialization(prefixes=("0", "9"), source="random"),
    }

    with pytest.raises(ValueError, match="9"):
        policies[policy].validate(encoder)


def test_configured_yolo_encoder_returns_spatially_pooled_features(tmp_path):
    """A configured YOLO encoder extracts and restores one vector per image."""
    from ultralytics.nn.tasks import ClassificationModel

    from smartrodent.components import ComponentSpec
    from smartrodent.encoders import build_encoder, encoder_checkpoint_spec

    classifier = ClassificationModel(
        cfg={
            "nc": 2,
            "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
            "head": [[-1, 1, "Classify", [2]]],
        },
        verbose=False,
    ).eval()
    checkpoint = tmp_path / "tiny-cls.pt"
    torch.save({"model": classifier}, checkpoint)
    spec = ComponentSpec(name="YoloEncoder", parameters={"model_name": str(checkpoint)})
    encoder = build_encoder(spec).eval()
    images = torch.linspace(0, 1, 2 * 3 * 8 * 12).reshape(2, 3, 8, 12)

    with torch.inference_mode():
        backbone, head = classifier.model
        expected = head.conv(backbone(images)).mean(dim=(-2, -1))
        features = encoder(images)

    assert features.shape == (2, encoder.feature_dim)
    torch.testing.assert_close(features, expected)

    restore_spec = encoder_checkpoint_spec(spec, encoder)
    checkpoint.unlink()
    restored = build_encoder(restore_spec).eval()
    restored.load_state_dict(encoder.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), expected)


def test_encoder_must_be_a_module_with_a_positive_feature_width():
    """A plain module without a feature width is not an encoder."""
    from smartrodent.components import ComponentSpec
    from smartrodent.encoders import build_encoder

    spec = ComponentSpec("torch.nn:Linear", {"in_features": 2, "out_features": 2})
    with pytest.raises(ValueError, match="feature_dim"):
        build_encoder(spec)


def test_encoder_without_checkpoint_parameters_is_rebuilt_from_its_own_spec():
    """Only encoders that offer checkpoint parameters change their restore spec."""
    from smartrodent.components import ComponentSpec
    from smartrodent.encoders import encoder_checkpoint_spec

    spec = ComponentSpec("torch.nn:Linear", {"in_features": 2, "out_features": 2})
    assert encoder_checkpoint_spec(spec, nn.Linear(2, 2)) is spec


@pytest.mark.parametrize(
    "arguments", [{}, {"model_name": "m.pt", "architecture": {"nc": 2}}]
)
def test_yolo_encoder_needs_exactly_one_model_source(arguments):
    """A checkpoint name and a parsed architecture are alternatives."""
    from smartrodent.yolo_encoder import YoloEncoder

    with pytest.raises(ValueError, match="exactly one"):
        YoloEncoder(**arguments)


def test_yolo_encoder_rejects_a_model_that_is_not_a_classifier(tmp_path):
    """A non-classification checkpoint has no pooled feature head to reuse."""
    from smartrodent.yolo_encoder import YoloEncoder

    checkpoint = tmp_path / "other.pt"
    torch.save({"model": nn.Linear(1, 1)}, checkpoint)
    with pytest.raises(ValueError, match="classification model"):
        YoloEncoder(checkpoint)


class _Scalar(nn.Module):
    """An encoder whose only parameter sits on the root module."""

    feature_dim = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, images):
        """Return scaled channel means padded to four features."""
        return images.mean(dim=(1, 2, 3)).unsqueeze(1) * self.scale


def test_random_reinitialization_needs_modules_that_can_reset():
    """A bare parameter has no initializer to redraw it with."""
    from smartrodent.encoders import Reinitialization

    with pytest.raises(ValueError, match="cannot be reinitialized"):
        Reinitialization(prefixes=["scale"], source="random").apply(_Scalar())
