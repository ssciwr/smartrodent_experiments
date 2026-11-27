"""Name the two training stages and what differs between them."""

from dataclasses import dataclass

from smartrodent.classifier_training import ClassifierExperiment
from smartrodent.stage_configurations import (
    ClassifierConfiguration,
    StageConfiguration,
    SupConConfiguration,
)
from smartrodent.stage_experiment import StageExperiment
from smartrodent.supcon_training import SupConExperiment


@dataclass(frozen=True)
class Stage:
    """Bundle the configuration class and the experiment of a stage.

    Attributes:
        configuration: Validated configuration class of the stage.
        experiment: Class that prepares and runs one configuration.
    """

    configuration: type[StageConfiguration]
    experiment: type[StageExperiment]


SUPCON = Stage(
    SupConConfiguration,
    SupConExperiment,
)
CLASSIFIER = Stage(
    ClassifierConfiguration,
    ClassifierExperiment,
)
