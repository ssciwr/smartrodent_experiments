"""Behavioral tests for the YOLO SupCon model wrapper."""

import numpy as np
import pytest
import torch
from pytorch_metric_learning.losses import SupConLoss

from smartrodent.supcon_model import YoloSupConModel


def test_wrapper_propagates_gradients_and_respects_modes(experiment_configuration):
    """SupCon updates the encoder and projector, leaving the classifier outside the loss."""
    torch.set_num_threads(2)
    model = YoloSupConModel(
        experiment_configuration.model_name, num_classes=2, output_features=4
    )
    model.requires_grad_(True)
    model.train()
    assert (
        model.training and model.full_model.model.training and model.projector.training
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    encoder = next(model.full_model.model.model[0].parameters())
    before = encoder.detach().clone()
    logits, embeddings = model(torch.rand(4, 3, 16, 16))
    assert logits.shape == (4, 2) and embeddings.shape == (4, 4)
    np.testing.assert_allclose(embeddings.detach().norm(dim=1).numpy(), 1, atol=1e-5)
    SupConLoss()(embeddings, torch.tensor([0, 0, 1, 1])).backward()
    assert encoder.grad is not None and model.projector[-1].weight.grad is not None
    assert model.head.linear.weight.grad is None
    optimizer.step()
    assert not torch.equal(before, encoder)
    model.eval()
    assert (
        not model.training
        and not model.full_model.model.training
        and not model.projector.training
    )
    model.finalize()
    assert model.embeddings is None


def test_invalid_checkpoint_metadata_fails(tmp_path):
    """Old or unrelated tensor files are rejected with a format explanation."""
    path = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="train_supcon.py"):
        YoloSupConModel.from_checkpoint(path)
