import base64
import gc
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import ollama
import pandas as pd
import yaml
from tqdm.auto import tqdm

from .base import Filterable


class VLMFilter(Filterable):
    """Classify dataframe photo records without modifying image files.

    Each species dataframe is handled independently. Model labels are written
    to task-specific nullable boolean columns.
    """

    def __init__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        labels: Sequence[str],
        taskname: str,
        photo_id_column: str = "photo.id",
        image_path_column: str = "image_path",
        log_level: int = logging.INFO,
    ):
        """Initialize a dataframe filter.

        Args:
            prompt: User prompt sent to the VLM for each image.
            system_prompt: System prompt sent to the VLM.
            labels: Distinct labels that the model may return.
            taskname: Suffix for label columns, e.g. ``kept_animal``.
            photo_id_column: Dataframe column identifying a photo.
            image_path_column: Dataframe column containing its on-disk path.
            log_level: Logging level for backend messages.

        Raises:
            ValueError: If labels, task name, or column names are invalid.
        """
        if (
            not isinstance(labels, Sequence)
            or isinstance(labels, str)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("labels must be a nonempty sequence of unique strings")
        if not isinstance(taskname, str) or not taskname.strip():
            raise ValueError("taskname must be a nonempty string")
        if not isinstance(photo_id_column, str) or not photo_id_column:
            raise ValueError("photo_id_column must be a nonempty string")
        if not isinstance(image_path_column, str) or not image_path_column:
            raise ValueError("image_path_column must be a nonempty string")

        self.prompt = prompt
        self.system_prompt = system_prompt
        self.labels = list(labels)
        self._normalized_labels = {label.strip().lower(): label for label in labels}
        self.taskname = taskname
        self.photo_id_column = photo_id_column
        self.image_path_column = image_path_column
        self.log_level = log_level
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(log_level)

    def __enter__(self):
        """Return this filter for use in a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release backend resources when leaving a context manager."""
        self.close()

    def close(self) -> None:
        """Release resources held by a backend, if any."""

    @classmethod
    def from_config(cls, config_path: str | Path) -> "VLMFilter":
        """Build a filter instance from YAML configuration.

        Args:
            config_path: YAML containing ``backend``, ``labels``, ``taskname``,
                prompts, and backend-specific settings.

        Returns:
            A configured Ollama- or vLLM-backed filter.

        Raises:
            ValueError: If the configured backend is unsupported.
        """
        with Path(config_path).open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        columns = config.get("columns", {})
        common = {
            "prompt": config["prompt"],
            "system_prompt": config["system_prompt"],
            "labels": config["labels"],
            "taskname": config["taskname"],
            "photo_id_column": columns.get("photo_id", "photo.id"),
            "image_path_column": columns.get("image_path", "image_path"),
        }

        backend = config["backend"]
        if backend == "ollama":
            return FilterOllama(
                **common,
                model=config["ollama"]["model"],
                pull_model=config["ollama"].get("pull_model", True),
            )
        if backend == "vllm":
            settings = config["vllm"]
            return FilterVLLM(
                **common,
                model_name=settings["model"],
                gpu_memory_utilization=settings.get("gpu_memory_utilization", 0.7),
                max_model_len=settings.get(
                    "max_model_len", settings.get("max-model-len", 8192)
                ),
                max_new_tokens=settings.get(
                    "max_new_tokens", settings.get("max-new-tokens", 300)
                ),
                batch_size=settings.get("batch_size", 64),
            )
        raise ValueError(f"Unknown backend {backend!r}")

    @property
    def model_tag(self) -> str:
        """str: Short identifier for the backend model."""
        raise NotImplementedError

    @property
    def task_columns(self) -> dict[str, str]:
        """dict[str, str]: Configured-label to output-column mapping."""
        return {label: f"{label}_{self.taskname}" for label in self.labels}

    def parse_response(self, raw: str | None) -> dict:
        """Parse a model JSON response against configured labels.

        Args:
            raw: Raw model response expected to be a JSON object with ``label``.

        Returns:
            A result mapping. Invalid JSON or unconfigured labels have a null
            label and ``parse_error=True``.
        """
        if raw is None:
            return {"label": None, "raw_response": None, "parse_error": True}
        try:
            response = json.loads(raw)
        except json.JSONDecodeError:
            response = None
        if not isinstance(response, dict):
            return {"label": None, "raw_response": raw, "parse_error": True}

        candidate = response.get("label")
        label = (
            self._normalized_labels.get(candidate.strip().lower())
            if isinstance(candidate, str)
            else None
        )
        return {
            "label": label,
            "raw_response": raw,
            "parse_error": label is None,
        }

    def filter_data(
        self, records_by_species: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return per-species copies annotated with model-label columns.

        Duplicate photo IDs are collapsed within their species only. Every row
        with the same ID must refer to the same image path and receives one
        shared classification. Input frames are not mutated and files are only
        read by the inference backend.

        Args:
            records_by_species: Species names mapped to one-row-per-photo frames.

        Returns:
            Per-species dataframe copies with nullable boolean columns. A valid
            label produces one ``True`` and ``False`` for other labels; a failed
            or unrecognized response produces ``pd.NA`` in every task column.

        Raises:
            TypeError: If the input mapping or any frame has the wrong type.
            ValueError: If columns/IDs are invalid, output columns already
                exist, or duplicate IDs have different image paths.
        """
        groups_by_species = self._validate_and_group(records_by_species)
        results = {}
        try:
            for species, (frame, groups) in groups_by_species.items():
                annotated = frame.copy(deep=True)
                for output_column in self.task_columns.values():
                    annotated[output_column] = pd.Series(
                        pd.NA, index=annotated.index, dtype="boolean"
                    )

                paths = [path for _, path in groups]
                classifications = self._classify_paths(paths)
                if len(classifications) != len(groups):
                    raise RuntimeError("Backend returned a different number of results")

                for (positions, _), classification in zip(
                    groups, classifications, strict=True
                ):
                    label = classification.get("label")
                    if classification.get("parse_error") or label not in self.labels:
                        continue
                    for configured_label, output_column in self.task_columns.items():
                        annotated.iloc[
                            positions, annotated.columns.get_loc(output_column)
                        ] = (configured_label == label)
                results[species] = annotated
        finally:
            self.close()
        return results

    def _validate_and_group(self, records_by_species):
        """Validate every frame before inference and group duplicate photos."""
        if not isinstance(records_by_species, dict):
            raise TypeError("records_by_species must be a dict of dataframes")

        grouped = {}
        required_columns = {self.photo_id_column, self.image_path_column}
        task_columns = set(self.task_columns.values())
        for species, frame in records_by_species.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{species}: expected a pandas DataFrame")
            missing = required_columns.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{species}: missing required columns {sorted(missing)}"
                )
            existing = task_columns.intersection(frame.columns)
            if existing:
                raise ValueError(
                    f"{species}: task output columns already exist: {sorted(existing)}"
                )
            if frame[self.photo_id_column].isna().any():
                raise ValueError(
                    f"{species}: {self.photo_id_column} values must not be null"
                )
            if frame[self.image_path_column].isna().any():
                raise ValueError(
                    f"{species}: {self.image_path_column} values must not be null"
                )

            species_groups = []
            for positions in frame.groupby(
                self.photo_id_column, sort=False, dropna=False
            ).indices.values():
                paths = [
                    Path(frame.iloc[position][self.image_path_column])
                    for position in positions
                ]
                if any(path != paths[0] for path in paths[1:]):
                    raise ValueError(
                        f"{species}: duplicate {self.photo_id_column} rows must have "
                        f"the same {self.image_path_column}"
                    )
                species_groups.append((positions, paths[0]))
            grouped[species] = (frame, species_groups)
        return grouped

    def _classify_paths(self, image_paths: list[Path]) -> list[dict]:
        """Classify paths through the selected backend."""
        raise NotImplementedError


class FilterOllama(VLMFilter):
    """Classify dataframe image records using a persistent Ollama connection."""

    def __init__(self, *, model: str, pull_model: bool = True, **kwargs):
        """Initialize the Ollama-backed filter.

        Args:
            model: Ollama model name.
            pull_model: Whether to download the model if it is not local.
            **kwargs: Arguments forwarded to :class:`VLMFilter`.
        """
        super().__init__(**kwargs)
        self.client = ollama.Client()
        self.model = model
        self._closed = False
        if pull_model:
            self.logger.info("Download model")
            self.logger.info(
                "Model download result: %s", self.client.pull(model, stream=False)
            )

    @property
    def model_tag(self) -> str:
        """str: The Ollama model name."""
        return self.model

    def classify(self, path: Path) -> dict:
        """Classify one image via Ollama.

        Args:
            path: On-disk path of the image to classify.

        Returns:
            Parsed response containing a configured label or a parse error.
        """
        encoded_image = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        response = self.client.generate(
            model=self.model,
            system=self.system_prompt,
            prompt=self.prompt,
            images=[encoded_image],
            format="json",
            stream=False,
            think=False,
            options={"temperature": 0},
        )
        return self.parse_response(response.response)

    def _classify_paths(self, image_paths: list[Path]) -> list[dict]:
        """Classify each unique photo, preserving input order."""
        return [
            self.classify(path)
            for path in tqdm(image_paths, desc="OLLAMA filtering images")
        ]

    def close(self) -> None:
        """Unload the model while leaving the externally managed daemon alive."""
        if self._closed:
            return
        try:
            self.client.generate(model=self.model, keep_alive=0)
        except Exception:
            self.logger.warning(
                "Could not unload Ollama model %s", self.model, exc_info=True
            )
        else:
            self._closed = True


class FilterVLLM(VLMFilter):
    """Classify dataframe image records in batches using vLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 8192,
        max_new_tokens: int = 300,
        batch_size: int = 64,
        **kwargs,
    ):
        """Initialize the vLLM-backed filter.

        Args:
            model_name: Name or path of the vLLM model to load.
            gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
            max_model_len: Maximum context length for the model.
            max_new_tokens: Maximum generated tokens per response.
            batch_size: Number of unique photos to classify in each batch.
            **kwargs: Arguments forwarded to :class:`VLMFilter`.
        """
        super().__init__(**kwargs)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._llm: Any = None
        self._sampling_params: Any = None
        self._closed = False

    @property
    def model_tag(self) -> str:
        """str: The vLLM model name, without any path prefix."""
        return self.model_name.split("/")[-1]

    @property
    def response_json_schema(self) -> dict:
        """dict: Structured-output schema containing this filter's labels."""
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": self.labels},
                "visible_animal": {"type": "boolean"},
                "evidence_kept": {"type": "array", "items": {"type": "string"}},
                "evidence_rejected": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "image_quality": {
                    "type": "string",
                    "enum": ["clear", "poor", "unusable"],
                },
                "needs_human_review": {"type": "boolean"},
            },
            "required": [
                "label",
                "visible_animal",
                "evidence_kept",
                "evidence_rejected",
                "image_quality",
                "needs_human_review",
            ],
            "additionalProperties": False,
        }

    def _ensure_engine(self) -> None:
        """Lazily construct the vLLM engine and sampling parameters."""
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._llm = LLM(
            model=self.model_name,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=self.response_json_schema),
        )

    @staticmethod
    def image_to_data_url(path: Path) -> str:
        """Encode an image file as a base64 data URL.

        Args:
            path: Image path.

        Returns:
            A ``data:image/<suffix>;base64,...`` URL.
        """
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        if suffix == "jpg":
            suffix = "jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/{suffix};base64,{encoded}"

    def build_conversation(self, image_path: Path) -> list[dict]:
        """Build a vLLM chat conversation for one image.

        Args:
            image_path: On-disk image path to include in the conversation.

        Returns:
            System and user messages for the vLLM chat endpoint.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self.image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]

    def _classify_batch(self, image_paths: list[Path]) -> list[dict]:
        """Classify paths in one vLLM request, preserving their input order."""
        self._ensure_engine()
        conversations = [self.build_conversation(path) for path in image_paths]
        outputs = self._llm.chat(
            cast(Any, conversations),
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )
        return [self.parse_response(output.outputs[0].text) for output in outputs]

    def classify(self, path: Path) -> dict:
        """Classify one image through the vLLM engine.

        Args:
            path: On-disk path of the image to classify.

        Returns:
            Parsed response containing a configured label or a parse error.
        """
        return self._classify_batch([path])[0]

    def _classify_paths(self, image_paths: list[Path]) -> list[dict]:
        """Classify unique photos in batches, without initializing for no work."""
        results = []
        for start in tqdm(
            range(0, len(image_paths), self.batch_size), desc="vLLM filtering images"
        ):
            results.extend(
                self._classify_batch(image_paths[start : start + self.batch_size])
            )
        return results

    def close(self) -> None:
        """Shut down the vLLM engine and release GPU references."""
        if self._closed:
            return
        try:
            engine = getattr(self._llm, "llm_engine", None)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            self.logger.warning("Could not shut down vLLM engine", exc_info=True)
        finally:
            self._llm = None
            self._sampling_params = None
            self._closed = True
            gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
