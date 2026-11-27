"""Build native YOLO CE classifier networks from SupCon or Ultralytics encoders."""

import warnings
from collections.abc import Mapping
from pathlib import Path
from pickle import UnpicklingError

import torch
from ultralytics.nn.modules import Classify
from ultralytics.nn.tasks import ClassificationModel, load_checkpoint
from ultralytics.utils.torch_utils import initialize_weights, intersect_dicts

from smartrodent.supcon_model import YoloSupConModel
from smartrodent.training_validation import EncoderSource, InitializationSettings


class YoloClassifierBuilder:
    """Build plain Ultralytics classification networks from explicit initialization settings.

    The result is the native ``ClassificationModel`` itself: freezing, training
    modes, and checkpoint writing are owned by the native trainer.
    """

    @classmethod
    def initialize(
        cls,
        initialization: InitializationSettings,
        *,
        label_names: Mapping[str, int],
        head_donor: Classify | None = None,
    ) -> ClassificationModel:
        """Start a new classifier network from an explicitly identified encoder checkpoint.

        Args:
            initialization: Already-validated encoder and replacement-head settings.
            label_names: Class-name-to-ID mapping supplied by the training dataset.
            head_donor: Already prepared native head, avoiding a second checkpoint read.

        Returns:
            A newly initialized network in evaluation mode, on the CPU. No optimizer
            or scheduler state is resumed from either source checkpoint.

        Raises:
            ValueError: A checkpoint has the wrong format. Native checkpoints must be trusted.
            NotImplementedError: A model is not a YOLO classification model.
            FileNotFoundError: A requested local source does not exist.
        """
        encoder = initialization["encoder"]
        network = cls._load_encoder(
            cls._local_checkpoint(encoder["checkpoint"]), encoder["source"]
        )
        if head_donor is None:
            head_donor = cls.prepare_head(initialization["head"], label_names)
        cls._replace_head(network, label_names, initialization["head"], head_donor)
        network.names = {index: name for name, index in label_names.items()}
        network.yaml["nc"] = len(label_names)
        # Export-mode Classify returns only probabilities. Keep the normal
        # (probabilities, logits) evaluation contract.
        network.model[-1].export = False
        # Clear source-stage freezing (e.g. SupCon's own freeze); the native trainer
        # applies the classifier's freeze list. This also avoids one native warning
        # per re-enabled parameter, matching native get_model().
        network.requires_grad_(True)
        return network.eval()

    @classmethod
    def prepare_head(
        cls, settings: Mapping, labels: Mapping[str, int], *, cache: dict | None = None
    ) -> Classify | None:
        """Load a read-only YOLO head before training and flag equivalent linear initialization.

        Args:
            settings: Verified head source/replacement settings.
            labels: Target dataset mapping.
            cache: Optional checkpoint cache owned by a pipeline or study; never global.

        Returns:
            Native donor head, or None for random initialization.

        Raises:
            NotImplementedError: The checkpoint does not contain a YOLO classifier.
            FileNotFoundError: The local donor checkpoint does not exist.
            ValueError: The donor is not an Ultralytics checkpoint.
        """
        if settings["source"] == "random":
            return None
        path = cls._local_checkpoint(settings["checkpoint"])
        heads = {} if cache is None else cache
        if path not in heads:
            heads[path] = cls._classification_head(cls.load_native(path))
        donor = heads[path]
        if settings["replace"] == "linear" and donor.linear.out_features != len(labels):
            warnings.warn(
                f"Linear head {path}: {donor.linear.out_features} -> {len(labels)} classes; "
                "no output weights transfer. This is equivalent to random linear initialization "
                "for the same encoder and training settings; configured variants are not deduplicated.",
                UserWarning,
                stacklevel=2,
            )
        return donor

    @classmethod
    def load_native(cls, path: str | Path) -> ClassificationModel:
        """Load an explicitly requested local native checkpoint without fusing BatchNorm.

        Args:
            path: Local Ultralytics classification checkpoint; never downloaded.

        Returns:
            The checkpoint's network on the CPU.

        Raises:
            FileNotFoundError: The local checkpoint does not exist.
            ValueError: The file is not an Ultralytics checkpoint.
        """
        try:
            network, _ = load_checkpoint(
                str(cls._local_checkpoint(path)), device="cpu", fuse=False
            )
        except TypeError as error:
            raise ValueError(
                "Expected an Ultralytics classification checkpoint format"
            ) from error
        return network

    @classmethod
    def _load_encoder(
        cls, path: Path, source_format: EncoderSource
    ) -> ClassificationModel:
        """Dispatch only schema-approved formats; never guess a format after loader failure."""
        if source_format == "supcon":
            try:
                source = YoloSupConModel.from_checkpoint(path)
            except UnpicklingError as error:
                raise ValueError(
                    "Expected a standalone SupCon checkpoint format"
                ) from error
            # Removing the hook releases its closure over the SupCon wrapper;
            # only the native network, not the projector/optimizer, is retained.
            return source.finalize().model
        elif source_format == "ultralytics":
            return cls.load_native(path)
        else:
            raise ValueError(f"Unsupported encoder source format: {source_format}")

    @classmethod
    def _replace_head(
        cls,
        network: ClassificationModel,
        labels: Mapping[str, int],
        settings: Mapping,
        donor=None,
    ) -> None:
        """Use native constructors/shape intersection, never transplant a donor backbone."""
        original = cls._classification_head(network)
        if settings["replace"] == "linear":
            # Always create a fresh target, even if the SupCon output count already
            # matches: an incompatible donor must not retain the old output map.
            target = torch.nn.Linear(original.linear.in_features, len(labels))
            original.linear = target
            source = donor.linear if donor is not None else None
        elif settings["replace"] == "full":
            conv = original.conv.conv
            target = Classify(
                conv.in_channels,
                len(labels),
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.groups,
            )
            initialize_weights(target)
            target.i, target.f, target.type = original.i, original.f, original.type
            target.np = sum(parameter.numel() for parameter in target.parameters())
            network.model[-1] = target
            source = donor
        else:
            raise ValueError(f"Unsupported head replacement: {settings['replace']}")
        if source is not None:
            target.load_state_dict(
                intersect_dicts(source.state_dict(), target.state_dict()), strict=False
            )

    @staticmethod
    def _local_checkpoint(path: str | Path) -> Path:
        """Require local files before calling loaders that may otherwise download weights."""
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local checkpoint does not exist: {checkpoint}")
        # Ultralytics treats a corrupt bare official asset name as a cache entry
        # to delete/re-download. An absolute path keeps local inputs read-only.
        return checkpoint.resolve()

    @staticmethod
    def _classification_head(network: ClassificationModel) -> Classify:
        """Reject detection networks and other non-YOLO-classifier structures early."""
        if not isinstance(network, ClassificationModel) or not isinstance(
            network.model[-1], Classify
        ):
            raise NotImplementedError(
                "Only YOLO classification models with a Classify head are supported"
            )
        return network.model[-1]
