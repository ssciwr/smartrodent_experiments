"""Shared MLflow tracking and registry settings for the training stages."""

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from smartrodent.training_validation import Nonblank, parse_settings


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MlflowConfiguration:
    """Configure MLflow tracking for a standalone experiment.

    Args:
        tracking_uri: MLflow tracking server or local database URI.
        experiment_name: Experiment receiving all runs, including study trials.
        registered_model_name: Optional registry name for each run's best model.
        model_artifact_name: Name of the logged MLflow PyTorch model.
        champion_alias: Alias assigned to the best validated registered version.
        dataset_revision: Optional immutable dataset revision recorded as a tag.
    """

    tracking_uri: Nonblank
    experiment_name: Nonblank
    registered_model_name: Nonblank | None = None
    model_artifact_name: Nonblank = "classifier"
    champion_alias: Nonblank = "champion"
    dataset_revision: Nonblank | None = None

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "MlflowConfiguration | None":
        """Build optional MLflow settings from YAML-compatible values.

        Args:
            values: Optional nested ``mlflow`` configuration mapping.

        Returns:
            Validated MLflow settings, or ``None`` when tracking is disabled.

        Raises:
            ValueError: The schema rejects missing, unexpected, or invalid fields.
        """
        if values is None:
            return None
        return parse_settings(cls, values)

    def configure(self) -> None:
        """Select the configured tracking backend and experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str, *, parent_run_id: str | None = None):
        """Start an owned root run or an explicitly authorized pipeline child.

        Args:
            run_name: Display name for the new run.
            parent_run_id: Active pipeline parent, or None for standalone execution.

        Returns:
            MLflow's active-run context, which closes only this newly owned run.

        Raises:
            ValueError: Implicit resumption is requested, an unrelated run is active,
                or the requested parent is absent.
        """
        if os.environ.get("MLFLOW_RUN_ID"):
            raise ValueError(
                "Implicit run resumption via MLFLOW_RUN_ID is not supported"
            )
        active = mlflow.active_run()
        if parent_run_id is None and active is not None:
            raise ValueError("An unrelated MLflow run is already active")
        if parent_run_id is not None and (
            active is None or active.info.run_id != parent_run_id
        ):
            raise ValueError("The requested MLflow parent run must be active")
        self.configure()
        return mlflow.start_run(run_name=run_name, nested=parent_run_id is not None)

    def log_best_artifacts(
        self,
        checkpoint: Path,
        onnx_model: Path | None = None,
        native_model: Path | None = None,
    ) -> None:
        """Publish a best checkpoint, its completed local cards, and its export to the owned run.

        Args:
            checkpoint: Selected best.pt whose YAML/Markdown cards already exist.
            onnx_model: ONNX file exported from the checkpoint, or None without export.
            native_model: Native ``best.pt`` written beside the checkpoint, or None.
        """
        paths = [
            checkpoint,
            checkpoint.with_suffix(".model_card.yaml"),
            checkpoint.with_suffix(".model_card.md"),
        ]
        paths.extend(path for path in (onnx_model, native_model) if path is not None)
        for path in paths:
            mlflow.log_artifact(str(path), artifact_path="best_model")

    def run_tags(self) -> dict[str, str]:
        """Return source revisions that make a tracked run reproducible.

        Returns:
            Git commit and, when configured, dataset revision tags.
        """
        tags = {"git_commit": git_commit()}
        if self.dataset_revision is not None:
            tags["dataset_revision"] = self.dataset_revision
        return tags


class ModelPublication:
    """Publish a selected trained model with consistent cards and registry metadata.

    Args:
        configuration: Tracking/registry settings for this stage, not its pipeline.
        client: Optional existing client, preserving standalone publication callers.
    """

    def __init__(
        self, configuration: MlflowConfiguration, client: MlflowClient | None = None
    ):
        """Keep publication separate from reconstruction, optionally using an existing client."""
        self.configuration = configuration
        self.client = client

    def publish(
        self,
        model: torch.nn.Module,
        *,
        best_epoch: int,
        validated: bool,
        tags: Mapping,
        write_card: Callable[[str, dict[str, str | None]], tuple[Path, Path]],
        promote: bool = True,
    ) -> str:
        """Register the supplied best model, attach cards, then optionally promote it.

        Args:
            model: CPU/eval model restored from the stage's selected checkpoint.
            best_epoch: Epoch represented by both model and checkpoint.
            validated: Whether validation was performed; only then promote the alias.
            tags: Source revisions and selected-encoder provenance.
            write_card: Callback receiving version and model identity metadata;
                returns the completed YAML and Markdown card paths.
            promote: Assign the alias immediately for validated models. Classifier
                studies disable this until every trial has published successfully.

        Returns:
            The created registry version. Remote publication is not transactional;
            failures after version creation propagate without rollback guarantees.
        """
        config = self.configuration
        if config.registered_model_name is None:
            raise ValueError("Model publication requires registered_model_name")
        # MLflow 3.18 defaults to traced pt2 graphs. Preserve this project's
        # reconstructible wrappers and label metadata using trusted own-model pickle.
        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name=config.model_artifact_name,
            registered_model_name=config.registered_model_name,
            serialization_format="pickle",
        )
        if info.registered_model_version is None:
            raise RuntimeError("MLflow did not return a registered model version")
        version = str(info.registered_model_version)
        client = (
            self.client
            if self.client is not None
            else MlflowClient(tracking_uri=config.tracking_uri)
        )
        version_tags = {
            **tags,
            "validation_status": "passed" if validated else "not_validated",
            "best_epoch": best_epoch,
        }
        for key, value in version_tags.items():
            if value is not None:
                client.set_model_version_tag(
                    config.registered_model_name, version, key, str(value)
                )
        cards = write_card(version, {"run_id": info.run_id, "model_id": info.model_id})
        self._attach_model_cards(client, info.model_id, version, cards)
        if validated and promote:
            client.set_registered_model_alias(
                config.registered_model_name, config.champion_alias, version
            )
        return version

    def _attach_model_cards(
        self,
        client: MlflowClient,
        model_id: str,
        version: str,
        cards: tuple[Path, Path],
    ) -> None:
        """Attach completed cards and their description before alias promotion is allowed."""
        for path in cards:
            client.log_model_artifact(model_id, str(path))
        client.update_model_version(
            name=self.configuration.registered_model_name,
            version=version,
            description=cards[1].read_text(encoding="utf-8"),
        )


def git_commit() -> str:
    """Return the exact Git commit associated with a tracked run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git returned an empty commit revision")
    return commit
