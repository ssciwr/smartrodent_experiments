"""Behavioral tests for configurable supervised-contrastive YOLO training."""

from collections import Counter

import pytest
import torch
from PIL import Image
from torch import nn
from ultralytics.nn.modules.head import Classify

from smartrodent.yolo_supcon import (
    SupConClassificationHead,
    SupConClassificationOutput,
    PKSamplerFactory,
    SupConTrainerConfiguration,
    WeightedSupConLoss,
    YoloSupConClassificationTrainer,
    build_supcon_trainer,
)


class LegacySamplerInterface:
    """Represent the pytorch-metric-learning 2.9 sampler constructor."""

    def __init__(self, labels, m, batch_size, length_before_new_iter):
        self.labels = labels
        self.m = m
        self.batch_size = batch_size
        self.sample_count = length_before_new_iter


class CurrentSamplerInterface:
    """Represent the newer sampler constructor with generator support."""

    def __init__(self, labels, m, batch_size, num_samples, *, generator):
        self.labels = labels
        self.m = m
        self.batch_size = batch_size
        self.sample_count = num_samples
        self.seed = generator.initial_seed()


@pytest.fixture
def classification_dataset(tmp_path):
    """Create an imbalanced three-class dataset with train and validation splits."""
    dataset_root = tmp_path / "classification"
    train_counts = {"mouse": 3, "rat": 2, "shrew": 1}
    for class_index, (class_name, train_count) in enumerate(train_counts.items()):
        for split, image_count in (("train", train_count), ("val", 1)):
            class_directory = dataset_root / split / class_name
            class_directory.mkdir(parents=True)
            for image_index in range(image_count):
                Image.new(
                    "RGB",
                    (16, 16),
                    color=(class_index * 80, image_index * 40, 20),
                ).save(class_directory / f"{image_index}.png")
    return dataset_root


def test_supcon_configuration_derives_pk_batch_size():
    """A configured P-K batch exposes its effective batch size."""
    configuration = SupConTrainerConfiguration.from_mapping(
        {
            "enabled": True,
            "classes_per_batch": 8,
            "samples_per_class": 6,
            "epoch_size": 960,
            "projection_hidden_dimension": 512,
            "embedding_dimension": 128,
            "temperature": 0.1,
            "cross_entropy_weight": 1.0,
            "contrastive_weight": 0.25,
        }
    )

    assert configuration.batch_size == 48
    assert configuration.epoch_size == 960
    assert configuration.contrastive_weight == 0.25


@pytest.mark.parametrize(
    ("enabled", "expected_trainer"),
    [
        (True, YoloSupConClassificationTrainer),
        (False, None),
    ],
)
def test_supcon_trainer_factory_selects_only_when_enabled(enabled, expected_trainer):
    """The composition layer can select the trainer without wrapper knowledge."""
    configuration = SupConTrainerConfiguration(enabled=enabled)

    assert build_supcon_trainer(configuration) is expected_trainer


@pytest.mark.parametrize(
    "sampler_type",
    [LegacySamplerInterface, CurrentSamplerInterface],
)
def test_pk_sampler_factory_supports_old_and_new_size_parameters(sampler_type):
    """Project epoch size maps to either supported dependency interface."""
    configuration = SupConTrainerConfiguration(
        enabled=True,
        classes_per_batch=2,
        samples_per_class=2,
    )

    sampler = PKSamplerFactory(sampler_type).build(
        labels=[0, 0, 1, 1],
        configuration=configuration,
        epoch_size=8,
        seed=17,
    )

    assert sampler.labels == [0, 0, 1, 1]
    assert sampler.m == 2
    assert sampler.batch_size == 4
    assert sampler.sample_count == 8
    if isinstance(sampler, CurrentSamplerInterface):
        assert sampler.seed == 17


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"enabled": "yes"}, "enabled.*boolean"),
        ({"classes_per_batch": 1}, "classes_per_batch.*at least 2"),
        ({"samples_per_class": 1}, "samples_per_class.*at least 2"),
        ({"epoch_size": 0}, "epoch_size.*positive"),
        ({"epoch_size": 33}, "epoch_size.*batch size"),
        (
            {"projection_hidden_dimension": 0},
            "projection_hidden_dimension.*positive",
        ),
        ({"embedding_dimension": 0}, "embedding_dimension.*positive"),
        ({"temperature": 0.0}, "temperature.*greater than 0"),
        ({"cross_entropy_weight": -1.0}, "cross_entropy_weight.*non-negative"),
        ({"contrastive_weight": -1.0}, "contrastive_weight.*non-negative"),
        (
            {"cross_entropy_weight": 0.0, "contrastive_weight": 0.0},
            "loss weight must be greater than 0",
        ),
    ],
)
def test_supcon_configuration_rejects_invalid_values(override, message):
    """Invalid settings fail before an Ultralytics training run is created."""
    configuration = {
        "enabled": True,
        "classes_per_batch": 4,
        "samples_per_class": 2,
        "epoch_size": 32,
    }
    configuration.update(override)

    with pytest.raises((TypeError, ValueError), match=message):
        SupConTrainerConfiguration.from_mapping(configuration)


def test_supcon_head_returns_logits_and_projected_embeddings_while_training():
    """The dual head exposes both objectives from one shared representation."""
    configuration = SupConTrainerConfiguration(
        enabled=True,
        projection_hidden_dimension=32,
        embedding_dimension=12,
    )
    source_head = Classify(c1=16, c2=3)
    head = SupConClassificationHead.from_head(source_head, configuration)
    head.train()

    output = head(torch.randn(4, 16, 8, 8))

    assert isinstance(output, SupConClassificationOutput)
    assert output.logits.shape == (4, 3)
    assert output.embeddings.shape == (4, 12)


def test_supcon_head_preserves_pretrained_classifier_inference():
    """Adding the projection branch does not change classifier predictions."""
    torch.manual_seed(7)
    configuration = SupConTrainerConfiguration(enabled=True)
    source_head = Classify(c1=16, c2=3).eval()
    images = torch.randn(4, 16, 8, 8)
    expected_probabilities, expected_logits = source_head(images)

    head = SupConClassificationHead.from_head(source_head, configuration).eval()
    probabilities, logits = head(images)

    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(probabilities, expected_probabilities)


def test_supcon_head_rejects_non_classification_source():
    """Only an Ultralytics classification head can be adapted."""
    with pytest.raises(TypeError, match="source_head.*Classify"):
        SupConClassificationHead.from_head(
            nn.Linear(4, 2),
            SupConTrainerConfiguration(enabled=True),
        )


def test_weighted_supcon_loss_combines_terms_and_reaches_both_heads():
    """The joint objective reports its terms and differentiates both branches."""
    configuration = SupConTrainerConfiguration(
        enabled=True,
        classes_per_batch=2,
        samples_per_class=2,
        cross_entropy_weight=2.0,
        contrastive_weight=0.5,
    )
    criterion = WeightedSupConLoss(
        configuration,
        class_weights=torch.tensor([1.0, 3.0]),
    )
    assert not isinstance(criterion, nn.Module)
    logits = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        requires_grad=True,
    )

    loss, loss_items = criterion(
        SupConClassificationOutput(logits=logits, embeddings=embeddings),
        {"cls": torch.tensor([0, 0, 1, 1])},
    )

    assert set(loss_items) == {"loss", "ce_loss", "supcon_loss"}
    torch.testing.assert_close(loss_items["loss"], loss.detach())
    torch.testing.assert_close(
        loss.detach(),
        2.0 * loss_items["ce_loss"] + 0.5 * loss_items["supcon_loss"],
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad) > 0


def test_weighted_supcon_loss_accepts_standard_validation_predictions():
    """Validation logs weighted CE and an explicit zero SupCon term."""
    configuration = SupConTrainerConfiguration(
        enabled=True,
        classes_per_batch=2,
        samples_per_class=2,
        cross_entropy_weight=2.0,
        contrastive_weight=0.5,
    )
    criterion = WeightedSupConLoss(configuration)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    probabilities = logits.softmax(1)

    loss, loss_items = criterion(
        (probabilities, logits),
        {"cls": torch.tensor([0, 1])},
    )

    assert torch.isfinite(loss)
    assert loss_items["supcon_loss"] == 0
    torch.testing.assert_close(loss, 2.0 * loss_items["ce_loss"])


def test_supcon_trainer_builds_exact_pk_training_batches(
    tmp_path, classification_dataset
):
    """Every training batch contains K examples from each of P classes."""
    trainer = YoloSupConClassificationTrainer(
        overrides={
            "model": "yolo26n-cls.yaml",
            "data": str(classification_dataset),
            "device": "cpu",
            "workers": 0,
            "imgsz": 16,
            "project": str(tmp_path / "runs"),
            "name": "pk-loader-test",
            "supcon": {
                "enabled": True,
                "classes_per_batch": 2,
                "samples_per_class": 2,
                "epoch_size": 8,
            },
        }
    )

    loader = trainer.get_dataloader(
        str(classification_dataset / "train"),
        batch_size=4,
        rank=-1,
        mode="train",
    )

    assert len(loader) == 2
    for batch in loader:
        counts = Counter(batch["cls"].tolist())
        assert len(counts) == 2
        assert set(counts.values()) == {2}


def test_supcon_trainer_installs_dual_head_and_weighted_joint_loss(
    tmp_path, classification_dataset
):
    """Trainer setup connects original class counts to the dual-head criterion."""
    trainer = YoloSupConClassificationTrainer(
        overrides={
            "model": "yolo26n-cls.yaml",
            "data": str(classification_dataset),
            "device": "cpu",
            "workers": 0,
            "imgsz": 16,
            "project": str(tmp_path / "runs"),
            "name": "joint-loss-test",
            "cls_pw": 0.5,
            "supcon": {
                "enabled": True,
                "classes_per_batch": 2,
                "samples_per_class": 2,
                "epoch_size": 8,
            },
        }
    )
    trainer.model = trainer.get_model(cfg="yolo26n-cls.yaml", verbose=False)
    trainer.train_loader = trainer.get_dataloader(
        str(classification_dataset / "train"),
        batch_size=4,
        rank=-1,
        mode="train",
    )

    trainer.set_class_weights()

    assert isinstance(trainer.model.model[-1], SupConClassificationHead)
    assert isinstance(trainer.model.criterion, WeightedSupConLoss)
    weights = trainer.model.criterion.class_weights
    assert weights is not None
    assert weights[0] < weights[1] < weights[2]

    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    loss, loss_items = trainer.model.loss(batch)

    assert torch.isfinite(loss)
    assert set(loss_items) == {"loss", "ce_loss", "supcon_loss"}
    loss.backward()
    projection_gradient = trainer.model.model[-1].projector[-1].weight.grad
    assert projection_gradient is not None
    assert torch.count_nonzero(projection_gradient) > 0
