"""Name training outputs and protect input data and files from those writers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunArtifacts:
    """Paths shared by the stage writers and input guard.

    Each artifact is an explicit property so IDEs and type checkers see it.
    Always-written artifacts return ``Path``; optional ones return ``None`` when
    disabled.

    Args:
        directory: The stage's output directory.
        onnx: Whether this run exports an ONNX model.
        native: Whether this run writes a native classifier checkpoint.
    """

    directory: Path
    onnx: bool = field(default=False, kw_only=True)
    native: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        """Accept string directories while keeping the instance frozen."""
        object.__setattr__(self, "directory", Path(self.directory))

    @property
    def config(self) -> Path:
        """Return the run configuration path."""
        return self.directory / "config.yaml"

    @property
    def best_checkpoint(self) -> Path:
        """Return the best checkpoint path."""
        return self.directory / "best.ckpt"

    @property
    def last_checkpoint(self) -> Path:
        """Return the last checkpoint path."""
        return self.directory / "last.ckpt"

    @property
    def metrics_csv(self) -> Path:
        """Return the metrics CSV path."""
        return self.directory / "metrics.csv"

    @property
    def hparams_yaml(self) -> Path:
        """Return the hyperparameters YAML path."""
        return self.directory / "hparams.yaml"

    @property
    def card_yaml(self) -> Path:
        """Return the YAML model card path."""
        return self.best_checkpoint.with_suffix(".model_card.yaml")

    @property
    def card_markdown(self) -> Path:
        """Return the Markdown model card path."""
        return self.best_checkpoint.with_suffix(".model_card.md")

    @property
    def onnx_model(self) -> Path | None:
        """Return the ONNX model path, or ``None`` when export is disabled."""
        return self.best_checkpoint.with_suffix(".onnx") if self.onnx else None

    @property
    def native_checkpoint(self) -> Path | None:
        """Return the native checkpoint path, or ``None`` when disabled."""
        return self.best_checkpoint.with_suffix(".pt") if self.native else None

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return every file this configuration may write."""
        # Adding an artifact means one property above plus one entry here.
        candidates = (
            self.config,
            self.best_checkpoint,
            self.last_checkpoint,
            self.metrics_csv,
            self.hparams_yaml,
            self.card_yaml,
            self.card_markdown,
            self.onnx_model,
            self.native_checkpoint,
        )
        return tuple(path for path in candidates if path is not None)


def validate_output_paths(
    runs: Sequence[RunArtifacts],
    datasets: Iterable[Path | None] = (),
    sources: Iterable[Path | None] = (),
    files: Iterable[Path] = (),
) -> None:
    """Protect the actual paths used by writers from input collisions.

    Args:
        runs: Per-run artifact paths with disjoint output directories.
        datasets: Input splits that an output may not be inside, including aliases.
        sources: Input files that no output may overwrite.
        files: Further files a writer produces, such as a pipeline's summaries;
            they are guarded like the run artifacts.

    Raises:
        ValueError: Outputs overlap, lie inside data, or collide with an input.
    """
    directories = [run.directory.resolve() for run in runs]
    for index, directory in enumerate(directories):
        if any(
            directory.is_relative_to(other) or other.is_relative_to(directory)
            for other in directories[index + 1 :]
        ):
            raise ValueError("output_directory paths must not overlap")

    inputs = [Path(path).resolve() for path in datasets if path is not None]
    if any(
        directory.is_relative_to(split) for directory in directories for split in inputs
    ):
        raise ValueError("output_directory must be outside the input datasets")

    source_paths = {Path(path).resolve() for path in sources if path is not None}
    for path in [*(path for run in runs for path in run.paths), *files]:
        resolved = path.resolve()
        if any(resolved.is_relative_to(split) for split in inputs):
            raise ValueError(f"Output artifact {path} points into an input dataset")
        if resolved in source_paths:
            raise ValueError(f"Output artifact {path} collides with source {resolved}")
        # Checking the link count protects input files without scanning entire splits.
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            raise ValueError(
                f"Output artifact {path} is hardlinked and could overwrite an input"
            )
