"""Annotate per-species image records with the answers of a vision-language model."""

import base64
import gc
import json
import logging
import sys
from abc import abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import ollama
import pandas as pd
import torch
from pydantic import ConfigDict, validate_call
from tqdm.auto import tqdm

from .base import Filterable
from .components import ComponentSpec
from .config_loading import load_section
from .training_validation import (
    Count,
    Flag,
    MappingInput,
    Name,
    Nonblank,
    parse_settings,
)


@runtime_checkable
class VLMBackend(Protocol):
    """A vision-language model that answers one prompt per image."""

    @abstractmethod
    def generate(
        self,
        image_paths: Sequence[Path],
        *,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> list[str | None]:
        """Answer the prompt for each image, constrained to the JSON schema.

        Args:
            image_paths: On-disk images, each answered separately.
            system_prompt: System prompt sent with every image.
            prompt: User prompt sent with every image.
            schema: JSON schema the backend enforces on every answer.

        Returns:
            One raw answer per image, in input order; None if none was generated.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the model; a later ``generate`` may load it again."""
        ...


class VLMFilter(Filterable):
    """Annotate per-species photo records with a VLM's answers; images are only read.

    The answer schema is configured once: the backend enforces it, the prompt cites
    it through the ``{answer_schema}`` placeholder, and its properties define the
    output columns. The ``label`` property's enum is the set of labels. Each label
    gets a nullable boolean column ``<label>_<taskname>``; every other property gets
    a column ``vlm_<taskname>_<field>``, so no answer can be mistaken for a label
    column by its name prefix.

    Args:
        prompt: User prompt sent with every image; contains ``{answer_schema}``,
            which is replaced by the schema as JSON.
        system_prompt: System prompt sent with every image.
        answer_schema: JSON schema of an answer; its ``label`` property is a
            required string with a nonempty enum of distinct labels.
        taskname: Suffix of the label columns, e.g. ``animal`` for ``kept_animal``.
        backend: The model that generates the answers.
        photo_id_column: Column identifying a photo; duplicates are answered once.
        image_path_column: Column containing each photo's on-disk path.

    Raises:
        ValueError: The prompt lacks the placeholder, the schema has no valid
            label enum, or two output columns would have the same name.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        prompt: Nonblank,
        system_prompt: str,
        answer_schema: dict[Name, Any],
        taskname: Nonblank,
        backend: VLMBackend,
        photo_id_column: Name = "photo.id",
        image_path_column: Name = "image_path",
    ):
        if "{answer_schema}" not in prompt:
            raise ValueError("prompt must contain the {answer_schema} placeholder")
        self.labels = self._labels(answer_schema)
        self.label_columns = {label: f"{label}_{taskname}" for label in self.labels}
        self.answer_columns = {
            field: f"vlm_{taskname}_{field}"
            for field in answer_schema["properties"]
            if field != "label"
        }
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError(f"Output columns collide: {self.output_columns}")
        # str.replace rather than str.format: prompts may contain other braces.
        self.prompt = prompt.replace(
            "{answer_schema}", json.dumps(answer_schema, indent=2)
        )
        self.system_prompt = system_prompt
        self.answer_schema = answer_schema
        self.required_fields = tuple(answer_schema["required"])
        self.backend = backend
        self.photo_id_column = photo_id_column
        self.image_path_column = image_path_column

    @staticmethod
    def _labels(answer_schema: dict) -> tuple[str, ...]:
        """Return the label enum after checking it can define the label columns."""
        label = answer_schema.get("properties", {}).get("label")
        labels = label.get("enum") if isinstance(label, dict) else None
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(value, str) and value.strip() for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "answer_schema needs a label property with an enum of distinct labels"
            )
        if "label" not in answer_schema.get("required", []):
            raise ValueError("answer_schema must require the label")
        return tuple(labels)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "VLMFilter":
        """Build a filter from the ``filter`` section of a YAML configuration.

        Args:
            config_path: YAML file whose ``filter`` section holds the constructor
                arguments; ``backend`` is ``{name, parameters}``, where the name is
                a backend of this module (``OllamaBackend``, ``VllmBackend``) or a
                ``module:attr`` class.

        Returns:
            A filter with its backend constructed. No model is loaded into
            memory yet; ``OllamaBackend`` does download its model when created
            with ``pull_model`` on.

        Raises:
            ValueError: The section, the backend, or a setting is invalid.
        """
        settings = load_section(config_path, "filter")
        backend = parse_settings(
            MappingInput[ComponentSpec], settings.pop("backend", None)
        )
        return cls(**settings, backend=backend.build(sys.modules[__name__], VLMBackend))

    @property
    def output_columns(self) -> list[str]:
        """list[str]: Label columns, then answer columns, as added to each frame."""
        return [*self.label_columns.values(), *self.answer_columns.values()]

    def filter_data(
        self, records_by_species: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return per-species copies annotated with the answers.

        Duplicate photo IDs are collapsed within their species only. Every row
        with the same ID must refer to the same image path and receives one
        shared answer. Input frames are not mutated and files are only read by
        the backend, which is released afterwards.

        Args:
            records_by_species: Species names mapped to one-row-per-photo frames.

        Returns:
            Per-species copies with the output columns. A usable answer sets one
            label column ``True``, the others ``False``, and each answer column to
            its value (lists and objects as JSON text). A missing or unusable
            answer leaves every output column ``NA``.

        Raises:
            TypeError: If the input mapping or any frame has the wrong type.
            ValueError: If columns/IDs are invalid, output columns already
                exist, or duplicate IDs have different image paths.
            RuntimeError: If the backend returns a different number of answers.
        """
        groups_by_species = self._validate_and_group(records_by_species)
        try:
            return {
                species: self._annotate(frame, groups)
                for species, (frame, groups) in groups_by_species.items()
            }
        finally:
            self.backend.close()

    def _annotate(self, frame: pd.DataFrame, groups: list) -> pd.DataFrame:
        """Copy one species frame and fill its output columns from the answers."""
        annotated = frame.copy(deep=True)
        for column in self.label_columns.values():
            annotated[column] = pd.Series(pd.NA, index=annotated.index, dtype="boolean")
        for column in self.answer_columns.values():
            annotated[column] = pd.Series(pd.NA, index=annotated.index, dtype=object)
        if not groups:
            return annotated

        responses = self.backend.generate(
            [path for _, path in groups],
            system_prompt=self.system_prompt,
            prompt=self.prompt,
            schema=self.answer_schema,
        )
        if len(responses) != len(groups):
            raise RuntimeError("Backend returned a different number of results")
        for (positions, _), response in zip(groups, responses, strict=True):
            answer = self._parse_answer(response)
            if answer is None:
                continue
            for label, column in self.label_columns.items():
                annotated.iloc[positions, annotated.columns.get_loc(column)] = (
                    label == answer["label"]
                )
            for field, column in self.answer_columns.items():
                if field in answer:
                    value = answer[field]
                    # Lists and objects are stored as JSON so they survive the CSV.
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)
                    annotated.iloc[positions, annotated.columns.get_loc(column)] = value
        return annotated

    def _parse_answer(self, raw: str | None) -> dict | None:
        """Return a usable answer, or None.

        Usable means a JSON object with a known label and every required field.
        Value types are not re-checked: the backends' constrained decoding enforces
        the schema, so a Python validator would only duplicate it.
        """
        if raw is None:
            return None
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(answer, dict)
            or answer.get("label") not in self.labels
            or any(field not in answer for field in self.required_fields)
        ):
            return None
        else:
            return answer

    def _validate_and_group(self, records_by_species):
        """Validate every frame before inference and group duplicate photos."""
        if not isinstance(records_by_species, dict):
            raise TypeError("records_by_species must be a dict of dataframes")

        grouped = {}
        required_columns = {self.photo_id_column, self.image_path_column}
        output_columns = set(self.output_columns)
        for species, frame in records_by_species.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{species}: expected a pandas DataFrame")
            missing = required_columns.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{species}: missing required columns {sorted(missing)}"
                )
            existing = output_columns.intersection(frame.columns)
            if existing:
                raise ValueError(
                    f"{species}: output columns already exist: {sorted(existing)}"
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
            image_paths = frame[self.image_path_column]
            for positions in frame.groupby(
                self.photo_id_column, sort=False
            ).indices.values():
                # Compared as Paths, so spellings of the same path are one image.
                paths = image_paths.iloc[positions].map(Path)
                if paths.nunique() > 1:
                    raise ValueError(
                        f"{species}: duplicate {self.photo_id_column} rows must have "
                        f"the same {self.image_path_column}"
                    )
                species_groups.append((positions, paths.iloc[0]))
            grouped[species] = (frame, species_groups)
        return grouped


class OllamaBackend(VLMBackend):
    """Generate answers through a running Ollama server.

    Args:
        model: Ollama model name.
        pull_model: Whether to download the model when the backend is created.
        num_workers: Number of concurrent requests.
        options: Ollama generation options (e.g. ``temperature``), passed as given.
    """

    @validate_call
    def __init__(
        self,
        *,
        model: Nonblank,
        pull_model: Flag = True,
        num_workers: Count = 1,
        options: dict[Name, Any] | None = None,
    ):
        self.model = model
        self.num_workers = num_workers
        self.options = {} if options is None else dict(options)
        self.client = ollama.Client()
        self.logger = logging.getLogger(__name__)
        self._closed = False
        if pull_model:
            self.logger.info(
                "Model download result: %s", self.client.pull(model, stream=False)
            )

    def generate(
        self,
        image_paths: Sequence[Path],
        *,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> list[str | None]:
        """Answer each image with one request, keeping the input order.

        Args:
            image_paths: On-disk images, each answered separately.
            system_prompt: System prompt sent with every image.
            prompt: User prompt sent with every image.
            schema: JSON schema, passed as Ollama's structured-output ``format``.

        Returns:
            One raw answer per image, in input order.
        """
        # A later close() must unload the model again after it was reloaded here.
        self._closed = False

        def answer(path: Path) -> str:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            return self.client.generate(
                model=self.model,
                system=system_prompt,
                prompt=prompt,
                images=[encoded],
                format=schema,
                stream=False,
                think=False,
                options=dict(self.options),
            ).response

        progress = {"total": len(image_paths), "desc": "Ollama filtering images"}
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                return list(tqdm(pool.map(answer, image_paths), **progress))
        else:
            return [answer(path) for path in tqdm(image_paths, **progress)]

    def close(self) -> None:
        """Unload the model, leaving the externally managed server running."""
        if self._closed:
            return
        try:
            self.client.generate(model=self.model, keep_alive=0)
        except (ollama.ResponseError, ConnectionError):
            # Unloading is cleanup; failing it must not hide the filter's result.
            self.logger.warning(
                "Could not unload Ollama model %s", self.model, exc_info=True
            )
        else:
            self._closed = True


class VllmBackend(VLMBackend):
    """Generate answers in batches with a local vLLM engine, loaded on first use.

    Args:
        model: Name or path of the vLLM model.
        batch_size: Number of images per engine request.
        engine: Keyword arguments for ``vllm.LLM`` (e.g. ``gpu_memory_utilization``,
            ``max_model_len``), passed as given.
        sampling: Keyword arguments for ``vllm.SamplingParams`` (e.g.
            ``temperature``, ``max_tokens``), passed as given.

    Raises:
        ValueError: ``engine`` or ``sampling`` sets an argument the backend supplies.
    """

    @validate_call
    def __init__(
        self,
        *,
        model: Nonblank,
        batch_size: Count = 64,
        engine: dict[Name, Any] | None = None,
        sampling: dict[Name, Any] | None = None,
    ):
        self.engine = {} if engine is None else dict(engine)
        self.sampling = {} if sampling is None else dict(sampling)
        supplied = {
            "engine": {"model", "limit_mm_per_prompt"} & set(self.engine),
            "sampling": {"structured_outputs"} & set(self.sampling),
        }
        for section, collisions in supplied.items():
            if collisions:
                raise ValueError(
                    f"{section} may not set {sorted(collisions)}; the backend does"
                )
        self.model = model
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self._llm: Any = None

    def generate(
        self,
        image_paths: Sequence[Path],
        *,
        system_prompt: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> list[str | None]:
        """Answer the images in batches, keeping the input order.

        Args:
            image_paths: On-disk images; no engine is loaded when there are none.
            system_prompt: System prompt sent with every image.
            prompt: User prompt sent with every image.
            schema: JSON schema enforced through vLLM's structured outputs.

        Returns:
            One raw answer per image, in input order.
        """
        if not image_paths:
            return []
        # Imported here: vLLM is an optional, GPU-only dependency.
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if self._llm is None:
            self._llm = LLM(
                model=self.model, limit_mm_per_prompt={"image": 1}, **self.engine
            )
        sampling_params = SamplingParams(
            **self.sampling, structured_outputs=StructuredOutputsParams(json=schema)
        )
        answers = []
        for start in tqdm(
            range(0, len(image_paths), self.batch_size), desc="vLLM filtering images"
        ):
            conversations = [
                self._conversation(path, system_prompt, prompt)
                for path in image_paths[start : start + self.batch_size]
            ]
            outputs = self._llm.chat(
                cast(Any, conversations),
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            answers.extend(output.outputs[0].text for output in outputs)
        return answers

    @staticmethod
    def _conversation(path: Path, system_prompt: str, prompt: str) -> list[dict]:
        """Build the chat messages for one image, sent inline as a data URL."""
        suffix = Path(path).suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix == "jpg" else suffix
        encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def close(self) -> None:
        """Shut down the engine and release its GPU memory."""
        if self._llm is None:
            return
        try:
            # Older vLLM engines have no shutdown(); dropping the reference suffices.
            shutdown = getattr(self._llm.llm_engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            # Cleanup must not hide the filter's own result or error; vLLM raises
            # backend-specific errors here that this module cannot enumerate.
            self.logger.warning("Could not shut down vLLM engine", exc_info=True)
        finally:
            self._llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
