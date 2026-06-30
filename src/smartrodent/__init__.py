"""SmartRodent model experiment package."""

# from .detection import main
from .detection import run_speciesnet
from .utils import save_speciesnet_crops
from .utils import save_speciesnet_previews
from .utils import short_speciesnet_label

__all__ = [
    "detection",
    "run_speciesnet",
    "save_speciesnet_crops",
    "save_speciesnet_previews",
    "short_speciesnet_label",
]
