"""Behavioral tests for the SupCon training datasets and batch sampling."""

import pickle

import torch
from torchvision.transforms.v2 import Compose, PILToTensor, ToDtype

from smartrodent.datasets import (
    ImageDataset,
    MultiViewImageDataset,
    collate_views,
)


def test_library_datasets_pickle_and_multiview_batching(experiment_configuration):
    """Imported datasets preserve image labels through pickling and collation."""
    config = experiment_configuration
    transform = Compose([PILToTensor(), ToDtype(torch.float32, scale=True)])
    single = ImageDataset(config.train_dataset, config.class_names, transform)
    assert len(single) == 6 and single[0][0] == 0
    dataset = MultiViewImageDataset(config.train_dataset, config.class_names, transform)
    restored = pickle.loads(pickle.dumps(dataset))
    labels, images = collate_views([restored[0], restored[3]])
    assert labels.tolist() == [0, 1, 0, 1]
    assert images.shape == (4, 3, 16, 16)
    assert 0 <= images.min() <= images.max() <= 1
