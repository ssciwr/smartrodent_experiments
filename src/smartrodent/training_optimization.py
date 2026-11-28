"""Build a Lightning optimizer and its configured learning-rate phases."""

from collections.abc import Iterable, Sequence
from dataclasses import field
from itertools import accumulate
from typing import Any, Literal

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch.optim import lr_scheduler

from smartrodent.components import ComponentSpec
from smartrodent.experiment_results import Objective
from smartrodent.training_validation import Count, Nonblank


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SchedulerPhase:
    """Configure one scheduler and its place in the training schedule.

    Args:
        name: PyTorch scheduler name or an importable ``module:attr``.
        parameters: Native scheduler options other than pipeline-owned arguments.
        epochs: Phase length. The final phase runs through the end of training.
        interval: When Lightning steps a single scheduler.
    """

    name: Nonblank
    parameters: dict[str, Any] = field(default_factory=dict)
    epochs: Count | None = None
    interval: Literal["epoch", "step"] = "epoch"


def configure_optimizers(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_spec: ComponentSpec,
    phases: Sequence[SchedulerPhase],
    objective: Objective,
) -> torch.optim.Optimizer | dict[str, Any]:
    """Build the return value of a Lightning task's optimizer hook.

    Args:
        parameters: Model parameters managed by the optimizer.
        optimizer_spec: PyTorch optimizer and its native options.
        phases: Ordered learning-rate phases. An empty list keeps the rate constant.
        objective: Metric and direction supplied to a plateau scheduler.

    Returns:
        The optimizer alone, or a Lightning optimizer and scheduler mapping.

    Raises:
        ValueError: A configuration cannot be built or phases cannot be combined.
    """
    optimizer = optimizer_spec.build(
        torch.optim, torch.optim.Optimizer, params=parameters
    )
    if not phases:
        return optimizer

    specs = [ComponentSpec(phase.name, phase.parameters) for phase in phases]
    plateau = any(
        issubclass(
            spec.resolve(lr_scheduler, lr_scheduler.LRScheduler),
            lr_scheduler.ReduceLROnPlateau,
        )
        for spec in specs
    )
    if plateau and (len(phases) != 1 or phases[0].interval != "epoch"):
        raise ValueError(
            "ReduceLROnPlateau must be the only phase and run after each epoch"
        )
    if len(phases) > 1 and any(phase.interval != "epoch" for phase in phases):
        raise ValueError("Scheduler chains require epoch intervals")
    if phases[-1].epochs is not None:
        raise ValueError(
            "The final scheduler phase runs until training ends; omit epochs"
        )
    durations = [phase.epochs for phase in phases[:-1]]
    if any(epochs is None for epochs in durations):
        raise ValueError("Each scheduler phase before the last requires epochs")

    supplied = {"mode": objective.mode} if plateau else {}
    schedulers = [
        spec.build(
            lr_scheduler, lr_scheduler.LRScheduler, optimizer=optimizer, **supplied
        )
        for spec in specs
    ]
    if len(schedulers) > 1:
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=list(accumulate(durations))
        )
    else:
        scheduler = schedulers[0]
    settings = {"scheduler": scheduler, "interval": phases[0].interval}
    if plateau:
        settings["monitor"] = objective.metric
    return {"optimizer": optimizer, "lr_scheduler": settings}
