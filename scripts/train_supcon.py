"""Run the standalone multiview SupCon experiment from a YAML configuration."""

import argparse
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    PILToTensor,
    RandomAdjustSharpness,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    Resize,
    ToDtype,
)
from tqdm.auto import tqdm

from smartrodent.supcon_datasets import MultiViewImageDataset, collate_views
from smartrodent.supcon_model import YoloSupConModel


@dataclass(frozen=True)
class TrainingConfiguration:
    """Configure the standalone, full-precision contrastive training experiment.

    Args:
        train_dataset: Directory containing one subdirectory per species.
        output_directory: Separate directory for configuration, history, checkpoints.
        class_names: Species directories to include.
        model_name: Pretrained classification checkpoint or model YAML.
        epochs: Number of training epochs.
        image_size: Square augmentation output size.
        classes_per_batch: Distinct classes selected per source-image batch.
        samples_per_class: Source images per selected class.
        samples_per_epoch: Source-image selections per epoch, rounded down to batches.
        workers: CPU DataLoader worker count.
        prefetch_factor: Batches prefetched per worker when workers are enabled.
        seed: Python, NumPy, and PyTorch random seed.
        device: PyTorch training device.
        learning_rate: AdamW learning rate.
        dropout: Dropout probability in the projection head.
        output_features: Contrastive embedding dimension.
        temperature: SupCon loss temperature.
        freeze: Backbone layer indices to freeze after enabling gradients.
        log_every: Number of steps between progress loss updates.
    """

    train_dataset: Path
    output_directory: Path
    class_names: tuple[str, ...]
    model_name: str = "yolo26x-cls.pt"
    epochs: int = 5
    image_size: int = 448
    classes_per_batch: int = 4
    samples_per_class: int = 6
    samples_per_epoch: int = 100000
    workers: int = 20
    prefetch_factor: int = 4
    seed: int = 42
    device: str = "cuda"
    learning_rate: float = 1e-4
    dropout: float = 0.5
    output_features: int = 128
    temperature: float = 0.1
    freeze: tuple[int, ...] = ()
    log_every: int = 50

    def __post_init__(self):
        """Reject invalid settings before building datasets or writing outputs."""
        integers = (
            self.epochs, self.image_size, self.classes_per_batch,
            self.samples_per_class, self.samples_per_epoch, self.prefetch_factor,
            self.output_features, self.log_every,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers):
            raise ValueError("Training sizes, counts, and log_every must be positive integers")
        if not isinstance(self.workers, int) or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if isinstance(self.class_names, str) or not self.class_names:
            raise ValueError("class_names must be a nonempty sequence of species names")
        if any(not isinstance(name, str) or not name for name in self.class_names):
            raise ValueError("class_names must contain nonempty strings")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if not 2 <= self.classes_per_batch <= len(self.class_names):
            raise ValueError("classes_per_batch must be between 2 and the number of species")
        if self.samples_per_epoch < self.batch_size:
            raise ValueError("samples_per_epoch must contain at least one full batch")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between zero and one")
        if any(not isinstance(index, int) or index < 0 for index in self.freeze):
            raise ValueError("freeze must contain nonnegative layer indices")
        output = Path(self.output_directory).resolve()
        source = Path(self.train_dataset).resolve()
        if output.is_relative_to(source):
            raise ValueError("output_directory must be outside the training dataset")

    @property
    def batch_size(self) -> int:
        """Return the number of source images per batch, before doubling views."""
        return self.classes_per_batch * self.samples_per_class

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfiguration":
        """Read plain YAML with paths relative to the invocation directory.

        Args:
            path: Training configuration file.

        Returns:
            Validated experiment settings.

        Raises:
            ValueError: The document is not a configuration mapping.
        """
        values = yaml.safe_load(path.read_text())
        if not isinstance(values, Mapping):
            raise ValueError("Training configuration must be a YAML mapping")
        values = dict(values)
        values["train_dataset"] = Path(values["train_dataset"])
        values["output_directory"] = Path(values["output_directory"])
        if not isinstance(values["class_names"], list):
            raise ValueError("class_names must be a YAML list")
        values["class_names"] = tuple(values["class_names"])
        values["freeze"] = tuple(values.get("freeze", ()))
        return cls(**values)

    def to_dict(self) -> dict:
        """Return plain, checkpoint-safe settings suitable for YAML output."""
        values = asdict(self)
        values["train_dataset"] = str(self.train_dataset)
        values["output_directory"] = str(self.output_directory)
        values["class_names"] = list(self.class_names)
        values["freeze"] = list(self.freeze)
        return values


class SupConExperiment:
    """Train the independent notebook experiment and save portable checkpoints.

    Args:
        configuration: Dataset, training, and output settings.
    """

    def __init__(self, configuration: TrainingConfiguration):
        self.configuration = configuration

    def _build_transform(self) -> Compose:
        """Reproduce the notebook's independently sampled augmentation pipeline."""
        size = self.configuration.image_size
        return Compose([
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Resize((size, size), antialias=True),
            RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
            RandomGrayscale(p=0.2),
            RandomPerspective(
                distortion_scale=0.5, p=0.2,
                interpolation=InterpolationMode.BILINEAR, fill=0,
            ),
            RandomHorizontalFlip(p=0.2),
            RandomResizedCrop(
                (size, size), scale=(0.75, 1.0), ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BILINEAR, antialias=True,
            ),
        ])

    def _build_loader(self) -> tuple[MultiViewImageDataset, DataLoader]:
        """Build label-aligned P-K batches with two views per selected image."""
        config = self.configuration
        dataset = MultiViewImageDataset(
            Path(config.train_dataset), labels=config.class_names,
            transform=self._build_transform(), size=config.image_size,
        )
        labels = [dataset.label_names[path.parent.name] for path in dataset.images]
        if len(set(labels)) < config.classes_per_batch:
            raise ValueError("Training dataset has too few nonempty classes for the requested batch")
        sampler = MPerClassSampler(
            labels=labels, m=config.samples_per_class, batch_size=config.batch_size,
            length_before_new_iter=config.samples_per_epoch,
        )
        options = dict(
            dataset=dataset, sampler=sampler, batch_size=config.batch_size,
            num_workers=config.workers, pin_memory=torch.device(config.device).type == "cuda",
            drop_last=True, collate_fn=collate_views,
            generator=torch.Generator().manual_seed(config.seed),
            persistent_workers=config.workers > 0,
        )
        if config.workers > 0:
            options["prefetch_factor"] = config.prefetch_factor
        return dataset, DataLoader(**options)

    def _save_checkpoint(self, model, optimizer, epoch, history) -> Path:
        """Save weights, architecture, label mapping, optimizer, and scalar history."""
        config = self.configuration
        path = Path(config.output_directory) / f"epoch_{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "architecture": deepcopy(model.full_model.model.yaml),
            "model_configuration": {
                "dropout": config.dropout, "output_features": config.output_features,
                "num_classes": len(config.class_names), "freeze": list(config.freeze),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_names": dict(model.label_names),
            "configuration": config.to_dict(),
            "history": list(history),
        }, path)
        return path

    def run(self) -> Path:
        """Train in FP32, checkpoint every epoch, and finalize hooks on exit.

        Returns:
            Path of the final epoch checkpoint.

        Raises:
            ValueError: Batch composition is incompatible with the source dataset.
            FloatingPointError: An epoch contains nonfinite losses.
        """
        config = self.configuration
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        dataset, loader = self._build_loader()
        model = YoloSupConModel(
            config.model_name, dropout=config.dropout, output_features=config.output_features,
            num_classes=len(config.class_names),
        ).to(config.device)
        model.requires_grad_(True)
        for index in config.freeze:
            model.full_model.model.model[index].requires_grad_(False)
        model.train(True)
        model.label_names = dict(dataset.label_names)
        model.training_configuration = config.to_dict()
        model.full_model.model.names = {index: name for name, index in model.label_names.items()}
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate
        )
        criterion = SupConLoss(temperature=config.temperature)
        output = Path(config.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
        print(
            f"{len(loader)} batches/epoch; {config.batch_size} source images and "
            f"{2 * config.batch_size} views/batch on {config.device}"
        )
        history = []
        try:
            for epoch in range(1, config.epochs + 1):
                losses = []
                progress = tqdm(loader, desc=f"Epoch {epoch}")
                for step, (labels, images) in enumerate(progress):
                    labels = labels.to(config.device, non_blocking=True)
                    images = images.to(config.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    _, embeddings = model(images)
                    loss = criterion(embeddings, labels)
                    loss.backward()
                    optimizer.step()
                    model.embeddings = None
                    losses.append(loss.detach())
                    if step % config.log_every == 0:
                        progress.set_postfix(sc_loss=f"{loss.item():.4f}")

                values = torch.stack(losses).float().cpu()
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite loss in epoch {epoch}")
                statistics = {
                    "epoch": epoch, "min": values.min().item(), "mean": values.mean().item(),
                    "std": values.std(correction=1 if len(values) > 1 else 0).item(),
                    "max": values.max().item(),
                }
                history.append(statistics)
                print(
                    f"Epoch: {epoch}, {statistics['min']:.6f}, "
                    f"{statistics['mean']:.6f} +/- {statistics['std']:.6f}, "
                    f"{statistics['max']:.6f}"
                )
                checkpoint = self._save_checkpoint(model, optimizer, epoch, history)
                (output / "history.yaml").write_text(yaml.safe_dump(history, sort_keys=False))
        finally:
            model.train(False)
            model.finalize()
        return checkpoint


def main(config_path: Path) -> Path:
    """Train one experiment from YAML.

    Args:
        config_path: YAML configuration; data paths are relative to the current directory.

    Returns:
        Final checkpoint path.
    """
    return SupConExperiment(TrainingConfiguration.from_yaml(config_path)).run()


def _parse_args() -> argparse.Namespace:
    """Parse the positional experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a standalone SupCon YAML configuration")
    return parser.parse_args()


if __name__ == "__main__":
    print("Final checkpoint:", main(_parse_args().config))
