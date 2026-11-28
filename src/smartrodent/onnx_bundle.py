"""Export the registered classifier as a bundle for sentinel-rat-ml-pipeline (schema 1)."""

import ast
import inspect
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import onnx
import torch
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from torchvision.transforms import v2

from smartrodent.transforms import ImageTransforms, TransformSettings

TAXON_FIELDS = frozenset(
    {
        "kingdom",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species",
        "common_name",
    }
)
ARTIFACT_DIRECTORY = "best_model"


class OnnxBundleError(Exception):
    """The registered model cannot be exported as a valid bundle."""


def order_classes(names: Mapping[int, str], taxonomy: Mapping[str, Mapping]) -> list:
    """Return the taxonomy entries in the graph's output order.

    Args:
        names: Output index to label, from the ONNX metadata.
        taxonomy: Label to ``Taxon`` fields.

    Returns:
        One taxon mapping per output index.

    Raises:
        OnnxBundleError: Labels and taxonomy keys differ, or a key is not a Taxon field.
    """
    labels = set(names.values())
    if labels != set(taxonomy):
        raise OnnxBundleError(
            "taxonomy labels must equal the model labels: "
            f"missing {sorted(labels - set(taxonomy))}, "
            f"extra {sorted(set(taxonomy) - labels)}"
        )
    for label, taxon in taxonomy.items():
        unknown = set(taxon) - TAXON_FIELDS
        if unknown:
            raise OnnxBundleError(
                f"Taxon for {label!r} has unknown keys {sorted(unknown)}"
            )
    return [dict(taxonomy[names[index]]) for index in sorted(names)]


class OnnxBundleExporter:
    """Write a bundle directory for one registered classifier version.

    Args:
        tracking_uri: MLflow tracking and registry URI.
        registered_model_name: Registry name; becomes ``info.name``.
        output_directory: New or empty directory receiving the bundle.
        taxonomy: Label to ``Taxon`` fields for every class.
        alias: Registry alias to resolve when no version is given.
        version: Explicit registry version, overriding the alias.
    """

    def __init__(
        self,
        *,
        tracking_uri: str,
        registered_model_name: str,
        output_directory: Path,
        taxonomy: Mapping[str, Mapping],
        alias: str = "champion",
        version: str | int | None = None,
    ):
        """Keep the settings; nothing is contacted until ``export``."""
        self.tracking_uri = tracking_uri
        self.name = registered_model_name
        self.output_directory = Path(output_directory)
        self.taxonomy = taxonomy
        self.alias = alias
        self.version = str(version) if version is not None else None

    def export(self) -> Path:
        """Download the registered run's files and write the bundle.

        Returns:
            The bundle directory.

        Raises:
            OnnxBundleError: Any precondition of the bundle is not met; nothing is
                written in that case.
        """
        self._require_free_output()
        client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri
        )
        version = self._resolve(client)
        run_id = self._run_id(version)
        with tempfile.TemporaryDirectory() as scratch:
            files = self._download(client, run_id, Path(scratch))
            manifest = self._manifest(files, str(version.version))
            self._write(files, manifest)
        return self.output_directory

    def _manifest(self, files: Mapping[str, Path], version: str) -> dict:
        """Build the full manifest; every graph and taxonomy check happens here."""
        model = onnx.load(str(files["onnx"]), load_external_data=False)
        return {
            "schema_version": 1,
            "role": "classifier",
            "info": {"name": self.name, "version": version},
            "model_file": "model.onnx",
            **self._graph_description(model),
            "classes": order_classes(self._names(model), self.taxonomy),
            "preprocessing": self._preprocessing(files["config"]),
            "calibration": {"method": "none"},
        }

    def _write(self, files: Mapping[str, Path], manifest: Mapping) -> None:
        """Create the bundle directory from the downloaded files and the manifest."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(files["onnx"], self.output_directory / "model.onnx")
        shutil.copy(files["card"], self.output_directory / "README.md")
        (self.output_directory / "config.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    def _require_free_output(self) -> None:
        """Refuse to write into a directory that already holds something."""
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise OnnxBundleError(
                f"Output directory {self.output_directory} is not empty"
            )

    def _resolve(self, client: MlflowClient):
        """Return the explicit version, or the one behind the alias."""
        try:
            if self.version is not None:
                return client.get_model_version(self.name, self.version)
            return client.get_model_version_by_alias(self.name, self.alias)
        except MlflowException as error:
            target = (
                f"version {self.version}"
                if self.version is not None
                else f"alias {self.alias!r}"
            )
            raise OnnxBundleError(
                f"Registered model {self.name!r} {target} not found: {error}"
            ) from error

    @staticmethod
    def _run_id(version) -> str:
        """Return the run that logged the registered version."""
        if not version.run_id:
            raise OnnxBundleError(f"Version {version.version} has no source run")
        return version.run_id

    @staticmethod
    def _download(client: MlflowClient, run_id: str, scratch: Path) -> dict[str, Path]:
        """Fetch the ONNX file, model card and run config of the run's best model."""
        listed = {
            Path(artifact.path).name
            for artifact in client.list_artifacts(run_id, ARTIFACT_DIRECTORY)
        }
        required = {
            "onnx": "best.onnx",
            "card": "best.model_card.md",
            "config": "config.yaml",
        }
        files = {}
        for key, filename in required.items():
            if filename not in listed:
                reason = "ONNX export was disabled" if key == "onnx" else "missing"
                raise OnnxBundleError(
                    f"Run {run_id} has no {ARTIFACT_DIRECTORY}/{filename} ({reason}); "
                    "the ONNX file, model card and config.yaml are required"
                )
            files[key] = Path(
                client.download_artifacts(
                    run_id, f"{ARTIFACT_DIRECTORY}/{filename}", str(scratch)
                )
            )
        return files

    @staticmethod
    def _names(model: onnx.ModelProto) -> dict[int, str]:
        """Read the output index to label mapping from the ONNX metadata."""
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        if "names" not in metadata:
            raise OnnxBundleError("The ONNX file has no 'names' metadata")
        try:
            names = ast.literal_eval(metadata["names"])
        except (ValueError, SyntaxError) as error:
            raise OnnxBundleError(f"Unreadable 'names' metadata: {error}") from error
        if not isinstance(names, dict):
            raise OnnxBundleError("The 'names' metadata must be an index to label map")
        return names

    @classmethod
    def _graph_description(cls, model: onnx.ModelProto) -> dict:
        """Describe opsets, the single input and the single output of the graph."""
        graph = model.graph
        initializers = {tensor.name for tensor in graph.initializer}
        inputs = [i for i in graph.input if i.name not in initializers]
        if len(inputs) != 1 or len(graph.output) != 1:
            raise OnnxBundleError("The graph must have exactly one input and output")
        return {
            "opsets": {(i.domain or "ai.onnx"): i.version for i in model.opset_import},
            "input": cls._tensor(inputs[0]),
            "output": {
                **cls._tensor(graph.output[0]),
                "score_semantics": "probabilities",
            },
        }

    @staticmethod
    def _tensor(value: onnx.ValueInfoProto) -> dict:
        """Describe name, dtype and shape; dynamic axes become ``None``."""
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise OnnxBundleError(f"Tensor {value.name!r} must be float32")
        shape = [
            dim.dim_value if dim.WhichOneof("value") == "dim_value" else None
            for dim in tensor_type.shape.dim
        ]
        return {"name": value.name, "dtype": "float32", "shape": shape}

    @classmethod
    def _preprocessing(cls, config_path: Path) -> list[dict]:
        """Describe the run's eval transform as ordered torchvision v2 steps."""
        transforms = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
            "transforms"
        ]
        settings = TransformSettings(
            image_size=transforms["image_size"], resize=transforms.get("resize", {})
        )
        steps = ImageTransforms(settings).validation().transforms
        return [cls._step(step) for step in steps]

    @staticmethod
    def _step(step: v2.Transform) -> dict:
        """Write one transform as its class name and constructor arguments.

        Required constructor parameters become ``args``; defaulted ones become
        ``kwargs`` when they differ from the default. Values are read from same-named attributes.
        """
        entry: dict = {"name": type(step).__name__}
        args, kwargs = [], {}
        parameters = inspect.signature(type(step).__init__).parameters
        for name, parameter in parameters.items():
            if name == "self":
                continue
            if not hasattr(step, name):
                raise OnnxBundleError(
                    f"{type(step).__name__} keeps no attribute {name!r} to export"
                )
            value = _plain(getattr(step, name))
            if parameter.default is inspect.Parameter.empty:
                args.append(value)
            elif value != parameter.default:
                # An explicit None is written when the default is not None, so the
                # consumer does not silently get the default.
                kwargs[name] = value
        if args:
            entry["args"] = args
        if kwargs:
            entry["kwargs"] = kwargs
        return entry


def _plain(value):
    """Convert torch dtypes and sequences into YAML-safe data."""
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
