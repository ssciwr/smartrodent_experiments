"""SmartRodent model experiment package."""

from .base import DataPreprocessorBase, YoloDatasetCreatorBase
from .dataprocessing import (
    ImageFilterBiotroveClip,
    ImageFilterCLIP,
    ImageFilterYoloE,
    YoloClassifierDatasetCreatorFromSpeciesnet,
    YoloDetectorDatasetCreatorFromSpeciesnet,
)
from .detection import (
    BioTroveClip_Detector,
    DetectorBase,
    SpeciesNet_Detector,
    YOLO_Detector,
    YOLOE_Detector,
    conf_to_string,
    image_groups,
    load_experiment_config,
    resolve_lookups,
    run_detector_on_group,
)
from .training import (
    YoloClassificationTrainer,
    YoloDetectionTrainer,
)
from .utils import (
    COUNTRY_ALIASES,
    IMAGE_SUFFIXES,
    extract_crop,
    image_paths,
    path_component,
)

__all__ = [
    "BioTroveClip_Detector",
    "COUNTRY_ALIASES",
    "DataPreprocessorBase",
    "DetectorBase",
    "IMAGE_SUFFIXES",
    "ImageFilterBiotroveClip",
    "ImageFilterCLIP",
    "ImageFilterYoloE",
    "SpeciesNet_Detector",
    "YOLO_Detector",
    "YOLOE_Detector",
    "YoloDatasetCreatorBase",
    "YoloClassifierDatasetCreatorFromSpeciesnet",
    "YoloDetectorDatasetCreatorFromSpeciesnet",
    "YoloClassificationTrainer",
    "YoloDetectionTrainer",
    "conf_to_string",
    "extract_crop",
    "image_groups",
    "image_paths",
    "load_experiment_config",
    "path_component",
    "resolve_lookups",
    "run_detector_on_group",
]
