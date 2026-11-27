# Refactoring plan: smartrodent

Status: Part 1 committed (`4e70493`). Part 3 committed (`a174eed`, `6b42463`). Part 2a implemented and approved, uncommitted (`uv lock` pending for scikit-learn). Next: 2b+2c. `datasets/` untouched.

## Decisions

| # | Decision |
|---|---|
| D1 | No Hydra. |
| D2 | Grid sweeps become Optuna studies; categorical choices go to Optuna as indices. |
| D3 | Both two-stage stages move to Lightning. A custom model is a plain `nn.Module` encoder; the pipeline owns the LightningModules. |
| D4 | The standalone Ultralytics classifier trainer (deleted in `2431935`) comes back, separate from the pipeline. |
| D5 | SupCon label metrics are removed. SupCon tracks its loss only. |
| D6 | Freezing is set by parameter-name prefixes. |
| D7 | Any `torch.optim.lr_scheduler` class can be configured, including chains (warmup → cosine). |
| D8 | Order: Part 1, then Part 3, then Part 2. |
| D9 | SupCon still computes and stores loss min/mean/std/max. The mean is the objective and the only value in the progress bar. |
| D10 | All ONNX exports use Ultralytics' `best_onnx_opset` (currently 18), recorded in the file metadata. |
| D11 | The metric identity stored in the MLflow registry keeps its current format, so existing champions stay comparable. |
| D12 | Configs may name any Python class or function (`module:attr`). The allowlist in `create_yolo_dataset.py` goes away. |
| D13 | The VLM filters' extra answers are saved as columns, not discarded. |
| D14 | A classifier trained on a YOLO encoder (standalone or two-stage, including SupCon encoders) is still saved as a native Ultralytics checkpoint that `YOLO(path)` loads. |

## Rules for every refactor step

1. Remove dead code and tests.
2. Composition over inheritance. Subclass only a library extension point (`LightningModule`, `Dataset`).
3. Modules are built adhering to the single responsibility principle.
4. Config is validated data. Library parameters are passed through under their own names. No copied defaults, no hand-written key checks where they are not necessary to prevent silent bugs or undefined behavior.
5. Every "algorithm with parameters" is chosen by `{name, parameters}`, built by one `ComponentSpec`.
6. Each fact appears once in a config. Example: the direction (min/max) comes only from the objective, not from scheduler `mode` or `early_stopping.comparison` as well.
7. Defaults live only in the schema. Nothing is inferred from names (e.g. `effective_tracked_mode`).

## Module map

| Now | After |
|---|---|
| `config_utils.py` | `config_loading.py`: tagged YAML loader, `load_section` |
| `training_validation.py` | `settings.py` (pydantic field types), `output_paths.py` (`RunArtifacts` plus the guard that protects inputs) |
| `experiment_support.py` | `tracking.py` (MLflow, publication, `git_commit`), `transforms.py`, `metrics.py`; the rest is replaced by Lightning |
| new | `components.py`: `ComponentSpec` |
| `experiment_results.py` | adds `Objective(metric, mode)` with `check` and `select_best`, replacing ~6 duplicated comparability checks |
| `supcon_model.py`, `classifier_model.py` | `encoders.py` (Encoder protocol, `FreezePolicy`, `Reinitialization`), `yolo_encoder.py` |
| new | `stage_runner.py`: builds the Lightning trainer and runs export → cards → MLflow for both stages |
| `classifier_sweeps.py`, `base.py`, `utils.py` | removed; contents move to their only users |
| `training.py` | `yolo_training.py` (standalone Ultralytics trainers) |
| `eval.py`, `filter.py` | `representation_eval.py`, `vlm_filter.py` |

Each rename also updates imports, `__init__.py`, `dvc.yaml` deps, README, test file names and `notebooks/supcon_training.ipynb`.

**`ComponentSpec`** is `{name, parameters}`.
- A bare name is looked up in the caller's default namespace (`torch.optim`, `torchvision.transforms.v2`, `sklearn.metrics`, `optuna.samplers`, `smartrodent.encoders`, …). A `module:attr` string imports custom code.
- Arguments the pipeline supplies (e.g. `params`, `mode`) must not also appear in `parameters`.
- Every spec is built once when the config is parsed, so a bad name fails before training starts.

## Part 1: sweeps → Optuna (includes Part 4)

- `search_space.py`:
  - `!suggest-categorical` accepts any YAML value; Optuna sees the indices `0..n-1`.
  - Study identity and reports use the real values.
  - Existing stored studies with categoricals need a new `study_name`.
- Tags:
  - Delete `!sweep`. A grid is `!suggest-categorical` plus `sampler: {name: BruteForceSampler}`.
  - Rename `!coupled-sweep` to `!coupled`.
  - Delete `!reference`, `!pyobject`, `convert_to_pyobject_tags` and `ConfigHandler`.
  - Use a private `SafeLoader` subclass, because `get_loader()` currently registers the tags on PyYAML's global `SafeLoader`.
- Every stage is either one run or a study. Delete:
  - `classifier_sweeps.py`;
  - `run_training` and `load_training_configurations`;
  - all sweep branches in `two_stage_training.py`.
- `TwoStageConfiguration` becomes plain data; the derived candidates move to a separate `TwoStagePlan`.
- `create_yolo_dataset.py`: a `datasets:` list of input/output pairs plus shared `creator: {name, parameters}`, instead of a sweep.
- Configs and README are converted.
- Accepted changes:
  - Invalid grid combinations become failed trials instead of upfront errors.
  - Grids without validation data are no longer possible.
  - Output folders become `trial_NNNN/` with a `study.yaml` report.
- Tests:
  - Delete `test_classifier_sweeps.py` and the `ConfigHandler` and dead-tag tests.
  - Add:
    - index-encoding round trip (Hypothesis);
    - a 2×2 BruteForce grid runs exactly 4 trials;
    - `!coupled` follows its categorical;
    - tags don't leak into `yaml.safe_load`;
    - the dataset list is processed.

## Part 3: library swaps

| Where | Replace with |
|---|---|
| NMS loop in `yolo_dataset_creation` | `torchvision.ops.nms` + `box_convert` (use a temporary old-vs-new comparison test) |
| `collect_metrics` in `yolo_training` | `DetMetrics.summary()` |
| Hand-written retry loops in `inaturalist`; no retries in `lila` | One `requests.Session` with urllib3 `Retry` |
| `MlflowTrackingConfiguration` | pydantic dataclass |

Details of the swaps (approved):
- Renames now: `filter.py` → `vlm_filter.py`, `training.py` → `yolo_training.py` (the detect/classify merge stays in 2a).
- `_plain_values` stays (dropped from the swaps): `to_jsonable_python` turns integer keys into strings, and the run's saved `config.yaml` must describe the run exactly.
- NMS: keep the original detection indices. The temporary old-vs-new test uses distinct scores; equal-score ties may now resolve differently (torch's sort is not stable).
- `collect_metrics`: the history format is unchanged; values are rounded to 5 decimals by `summary()`.
- Retries: `retrying_session(max_retries, backoff_factor)` retries 429/5xx; a persistent failure still raises `HTTPError`, and 404 is not retried. iNat records keep pyinaturalist's `ClientSession` (it already retries), so the manual loops go. LILA gains `max_retries` and `backoff_factor`.
- `ImageDataset` stays hand-written (dropped from the swaps): keeping the `(label, image)` order, the attribute names and flat indexing meant overriding most of `ImageFolder`, which was more complex than the original. It now sorts each class's files (`iterdir()` order depends on the filesystem), converts images to RGB, keeps its case-sensitive `.jpg`/`.png` extensions, raises `FileNotFoundError` for a missing class directory, and loses the dead `size=` argument and no-transform `torch.tensor(img)` branch; `MultiViewImageDataset` requires a transform.

Refactor items in the touched modules:
- **`vlm_filter.py`:**
  - One `VLMFilter` with a backend object (`OllamaBackend` / `VllmBackend`), chosen through `ComponentSpec`. This replaces the subclass hierarchy.
  - Remove the duplicated (and conflicting) defaults and the alias keys in `from_config`. vLLM settings pass through as `engine` (`vllm.LLM`) and `sampling` (`SamplingParams`). No injected temperature.
  - Define the answer schema once, in the filter YAML; the labels are its `label` enum. The prompt holds an `{answer_schema}` placeholder that the filter fills in; a missing placeholder is an error. Filter keys move under `filter:`.
  - Both backends enforce the schema. Every answer field is saved as a column, so `needs_human_review` and the other extra answers are kept (D13).
  - Columns: labels stay `{label}_{taskname}` booleans; other answers are `vlm_{taskname}_{field}`, so they never match a label prefix such as materialize's `rejected`. Lists are JSON strings; a parse error gives NA everywhere.
  - An answer is usable if it is a JSON object with a known label and every required field. Value types are not re-checked in Python; the backends' constrained decoding enforces them (no `jsonschema` dependency).
- **`yolo_dataset_creation.py`:** replace the three-level class hierarchy plus mixin with one `YoloDatasetCreator` made of:
  - a `SpeciesNetDetector` (`{name, parameters}`), one instance shared by all datasets;
  - a layout (`detector` or `classifier`);
  - a record-loading function.
  - Config: `detector: {name, parameters}`, `layout`, `class_names`, `batch_size` (source rows per inference batch) and `datasets`.
  - Drop the `conf`/`confidence` and `label`/`class` key fallbacks.
- **`inaturalist.py`:**
  - remove the unused `seed` (and from the configs);
  - remove the duplicate existence check;
  - stop adding logging handlers per instance; the script configures them.
- **`base.py`:** keep the `Configurable`, `Filterable` and `DatasetLoader` protocols and mark their methods `@abstractmethod`, so an incomplete implementation fails when it is created. `YoloDatasetCreatorBase` is replaced by `YoloDatasetCreator`.

Do not re-run any DVC data stage.

## Part 2: Lightning, custom encoders, standalone Ultralytics trainer

The workflow runs per sub-part (2a, 2b+2c, 2d+2e).

**2a. Restore the standalone Ultralytics trainer.**
- Restore the trainer, both scripts, its config, the DVC stages and its tests from `2431935^`.
- Scripts become single-run; search uses `YOLO.tune`.
- Drop the `/tmp` config round trip and the hard-coded `CUDA_VISIBLE_DEVICES`.
- Per-class metrics come from sklearn.
- One `UltralyticsTrainer(task: detect | classify)` replaces the classifier-subclasses-detector design.
- `tune_kwargs` lists only the overrides of `train_kwargs`.

**2b. Shared pieces.**
- Add `lightning`.
- `Encoder` contract: an `nn.Module` with `feature_dim` that returns `(B, feature_dim)` features.
  - Input is RGB in [0, 1]; normalization happens inside the encoder.
  - Built via `ComponentSpec`.
- `FreezePolicy(prefixes, batchnorm: eval | adapt)`. A prefix that matches nothing is an error.
- `Reinitialization(prefixes, source: random | pretrained)` replaces head `replace` / `source` / donor checkpoint.
- `yolo_encoder.py`: the YOLO backbone as an `Encoder`, without forward hooks. It is the only Ultralytics import on the pipeline path.
- Transforms: fixed preprocessing, then an ordered list of torchvision augmentations configured by `{name, parameters}`.
- Metrics: `{alias: {name, parameters}}`. The stored identity format stays the same (D11).
- `Objective` is the single source of min/max for checkpointing, early stopping, plateau schedulers, the Optuna direction and the champion comparison.
  - Remove `validate_stopping`, `effective_tracked_*`, `comparison` and scheduler `mode`.
- `scheduler` is a list of phases.
  - Several phases run in sequence via `SequentialLR`; each phase except the last sets `epochs`.
  - `ReduceLROnPlateau` may only be used on its own.
  - `[]` means a constant learning rate.
- `RunArtifacts` is the one list of output file names, used by both the writers and the guard.

**2c. SupCon.**
- `SupConTask` holds the encoder, the projector, `SupConLoss` and `FreezePolicy`.
- Samplers are unchanged.
- The objective is the SupCon loss mean, mode `min` (D5, D9).
- The unused classifier head is removed.
- `stage_runner.py`:
  - `trainer:` passes keyword arguments straight to `L.Trainer`. This replaces `epochs`, `device` and `amp`.
  - Callbacks: `ModelCheckpoint`; `EarlyStopping` (`patience` and `min_delta` passed through); a guard that raises `FloatingPointError` on NaN; an epoch observer that feeds Optuna pruning.
  - Loggers: `MLFlowLogger` attached to the active run, and `CSVLogger`.
- Checkpoints are Lightning `.ckpt` files. The encoder is restored from the saved hyperparameters.
- Delete `CheckpointStore`, `EarlyStopping`, `SchedulerConfiguration` and the hand-written training loop.

**2d. CE classifier.**
- `ClassifierTask` = encoder + `Linear` head, plus `FreezePolicy`, optional `Reinitialization`, CE loss and sklearn metrics.
- `ce_loss_mean` stays a whole-split mean (log it with `batch_size`).
- `encoder`:
  - standalone: `{checkpoint}` or `{name, parameters}`;
  - two-stage: a reserved key the pipeline fills in, which removes the `selected_supcon` placeholder.
- Same `optimizer` / `scheduler` / `early_stopping` / `trainer` sections as SupCon; this replaces `train_kwargs`.
- Native YOLO checkpoint (D14): an encoder may offer an optional native export, `to_native(head, label_names) -> nn.Module`. The YOLO encoder implements it by putting the trained encoder weights and the `Linear` head back into an Ultralytics `ClassificationModel` with `names`, saved in Ultralytics' checkpoint format as `best.pt` next to `best.ckpt`.
  - `native_checkpoint: true` in the classifier config requests it; an encoder without `to_native` is rejected when the configuration is validated, not after training.
  - The native `best.pt` is what MLflow registration logs for YOLO classifiers, so registered models stay loadable as before.
- Delete `classifier_model.py`, `supcon_model.py`, the Ultralytics trainer and validator adapters, the 40 protected setting keys, `native_freeze` and the old head schemas.

**2e. Pipeline.**
- Validate freeze and reinitialization prefixes against each encoder's parameter names. This replaces counting YOLO layers.
- One `StageStudy` parameterized by a `Stage` value replaces the `_StageStudy` ABC.
- ONNX export uses one `torch.onnx` path for the encoder and the classifier, at opset D10.
- `representation_eval` calls the encoder directly.
- Document that custom encoder modules must be importable wherever registered models are loaded.

**Accepted changes:**
- The two-stage classifier loses Ultralytics' EMA, `optimizer=auto`, plots and `results.csv`; warmup and cosine come from scheduler phases.
- Ties in the best checkpoint go to the earliest epoch.
- No atomic hard-link checkpoints.
- Old SupCon `best.pt` files can't be used as encoders.
- Native YOLO classifiers are an export of the Lightning model (D14); Ultralytics' trainer no longer trains them in the pipeline.

**Tests:**
- Replace `test_classifier_model`, most of `test_classifier_training`, `test_supcon_model`, the support-class tests, and the Ultralytics test helpers.
- Use a tiny CPU test encoder to test:
  - each stage trains end to end;
  - the checkpoint restores;
  - frozen parameters don't change, and BatchNorm `eval` / `adapt` behaves as configured;
  - reinitialization resets exactly the matched modules;
  - NaN raises;
  - pruning propagates;
  - the ONNX output matches torch;
  - the stored metric identity format is unchanged;
  - with the real YOLO encoder, the native `best.pt` loads with `YOLO(path)` and gives the same probabilities as the Lightning model (opt-in test).
- Hypothesis: prefix matching (`model.1` ≠ `model.10`) and `ComponentSpec` rejects colliding arguments.
- One opt-in test with the real YOLO encoder.

## Workflow

Each sub-part follows the workflow in AGENTS.md. When the refactorer runs, its report lists everything removed or flagged. The review runs the line and branch coverage commands. Record a baseline (tests, coverage) before each sub-part. No commits; your uncommitted README and config changes are preserved.

## Dead code register

Removed in Part 1: `!sweep`, `!coupled-sweep` (now `!coupled`), `!reference`, `!pyobject`,
`convert_to_pyobject_tags`, `ConfigHandler`, `config_utils.py`, `utils.py` (including
`resolve_data_path`), `classifier_sweeps.py`, `load_training_configurations`,
`run_training`, `TwoStageResult.classifier_result`, `StudyComponent`, and the dataset
creators' `from_config`.

| Item | Action |
|---|---|
| `ImageDataset(size=…)` and the no-transform `torch.tensor(img)` branch | remove, Part 3 |
| `InaturalistDataset(seed=…)` (unused) | remove, Part 3 |
| `Configurable`, `Filterable`, `DatasetLoader` protocols | keep; methods become `@abstractmethod`, Part 3 |
| SupCon classifier logits and `validation_metrics` | remove, Part 2c |
| `dvc.yaml` `create_yolo_classifier_dataset` outs don't match the config's output paths (pre-existing) | flag, needs your decision |
