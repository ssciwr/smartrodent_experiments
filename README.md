# SmartRodent experiments

Python tools for acquiring, reviewing, preparing, and learning from wildlife
images, with a focus on rodents, snakes, birds, and other small animals. The
repository contains an iNaturalist/LILA data workflow, SpeciesNet-assisted YOLO
dataset creation, Ultralytics training wrappers, and independent PyTorch
supervised-contrastive (SupCon) and cross-entropy (CE) training workflows.

## Contents

- [Installation](#installation)
- [Repository layout and library overview](#repository-layout-and-library-overview)
- [CLI reference](#cli-reference)
- [Configuration guide](#configuration-guide)
- [Dataset workflow](#dataset-workflow)
- [Training workflows](#training-workflows)
- [Checkpoints, results, and model cards](#checkpoints-results-and-model-cards)
- [DVC workflow](#dvc-workflow)
- [MLflow workflow](#mlflow-workflow)
- [Testing](#testing)
- [Troubleshooting and limitations](#troubleshooting-and-limitations)

## Installation

The project declares **Python >=3.12** in
[`pyproject.toml`](pyproject.toml) and uses [uv](https://docs.astral.sh/uv/).
Run commands from the repository root with the project installed in editable
mode in an activated virtual environment. `uv sync` installs this project in
editable mode by default.

For the CLI workflows documented here, install the `experiments` extra, which
provides DVC and MLflow:

```bash
uv sync --extra experiments
source .venv/bin/activate
```

For development and coverage reporting:

```bash
uv sync --extra experiments --extra dev
source .venv/bin/activate
```

The documented environment includes MLflow even when tracking is disabled:
some package imports depend on it. A base-only `uv sync` does not select the
`experiments` extra.

| Extra | Purpose |
| --- | --- |
| `experiments` | DVC and MLflow. |
| `dev` | pytest, Hypothesis, coverage, and test helpers. |
| `jupyter` | IPython kernel support for notebooks. |
| `ollama` | Ollama client extra; the client is also a runtime dependency. The Ollama server is separate. |
| `vllm` | Pinned local vLLM inference backend; needs a compatible GPU/software stack. |
| `all` | All declared extras, including the heavyweight inference backends. |

Add extras together when syncing an environment, for example
`uv sync --extra experiments --extra dev --extra jupyter`. The vLLM pin has a
resolver workaround documented in `pyproject.toml`:

```bash
# Optional: only for a workflow using the vLLM backend.
uv pip install "vllm==0.28.0" --torch-backend=auto
```

This may change the PyTorch stack; use a suitable environment rather than
installing every optional backend indiscriminately. Ollama-backed filtering
requires a running Ollama service. SpeciesNet, Ultralytics, and VLM workflows
may download model weights if they are not already available locally. CUDA
settings in the sample training configs require an available CUDA device;
choose `device: cpu` for CPU execution of the modern training workflows.

[`requirements_processing.txt`](requirements_processing.txt) and
[`requirements_ammico.txt`](requirements_ammico.txt) contain optional legacy
notebook/processing tools. They are not part of the main locked environment;
install one only when its separate workflow needs it.

## Repository layout and library overview

```text
configs/                 YAML parameters, training examples, model-card template
scripts/                 Executable workflow entry points
src/smartrodent/          Reusable Python library
notebooks/               Exploration and representation evaluation
tests/                   Unit, property-based, and opt-in integration tests
datasets/                Original images and intermediate/final datasets
dvc.yaml                 Declared data-processing and Ultralytics stage graph
pyproject.toml, uv.lock   Package metadata, extras, and dependency lockfile
```

Dataset directories and existing experiment outputs are not scratch space.
Choose output paths deliberately; direct scripts perform real writes, and
rerunning training in an existing directory can replace its active artifacts.

### Library map

Import reusable APIs from `smartrodent` modules, not executable scripts.

| Module(s) under `src/smartrodent/` | Responsibility and main APIs |
| --- | --- |
| [`inaturalist.py`](src/smartrodent/inaturalist.py) | `InaturalistDataset`: observation selection, licensed image acquisition, metadata, caching, and retries. |
| [`lila.py`](src/smartrodent/lila.py) | `LilaDataset`: license-aware camera-trap selection using LILA metadata and taxonomy. |
| [`licenses.py`](src/smartrodent/licenses.py) | License managers, Creative Commons handling, and license-manager registration. |
| [`filter.py`](src/smartrodent/filter.py) | `VLMFilter`, `FilterOllama`, `FilterVLLM`: label per-species image records without modifying originals. |
| [`dataset_splitting.py`](src/smartrodent/dataset_splitting.py) | `DatasetSplitter` and `TrainingDatasetSampler`: split metadata and sample training rows. |
| [`yolo_dataset_creation.py`](src/smartrodent/yolo_dataset_creation.py) | SpeciesNet-backed detector/classifier dataset creators. |
| [`datasets.py`](src/smartrodent/datasets.py) | `ImageDataset`, `MultiViewImageDataset`, `collate_views`: class mappings, image indexing, and two-view batching. |
| [`config_utils.py`](src/smartrodent/config_utils.py) | Custom YAML loader and `ConfigHandler` Cartesian/coupled sweep expansion. |
| [`training_validation.py`](src/smartrodent/training_validation.py) | Shared configuration schemas, parsing, serialization, and output/input path protections. |
| [`training.py`](src/smartrodent/training.py) | `YoloDetectionTrainer`, `YoloClassificationTrainer`: wrappers around Ultralytics train/tune/export. |
| [`supcon_model.py`](src/smartrodent/supcon_model.py), [`supcon_training.py`](src/smartrodent/supcon_training.py) | `YoloSupConModel`, `TrainingConfiguration`, `SupConExperiment`, `load_training_configurations()`, `run_training()`. |
| [`classifier_model.py`](src/smartrodent/classifier_model.py), [`classifier_training.py`](src/smartrodent/classifier_training.py) | `YoloClassifierModel`, `ClassifierConfiguration`, `ClassifierExperiment`: explicit encoder/head initialization and fresh CE training. |
| [`classifier_sweeps.py`](src/smartrodent/classifier_sweeps.py) | `ClassifierSweepConfiguration`, `ClassifierSweepExperiment`: prepare/run candidates and select the validated winner. |
| [`two_stage_training.py`](src/smartrodent/two_stage_training.py) | `TwoStageConfiguration`, `TwoStageExperiment`: SupCon selection followed by a classifier sweep. |
| [`experiment_support.py`](src/smartrodent/experiment_support.py), [`experiment_results.py`](src/smartrodent/experiment_results.py) | Transforms, schedulers, stopping, checkpoint storage, tracking/publication, and structured results. |
| [`classifier_champions.py`](src/smartrodent/classifier_champions.py) | `ClassifierChampionRegistry`: provenance-aware historical comparison and promotion. |
| [`model_card.py`](src/smartrodent/model_card.py) | Template loading, training-card population, YAML/Markdown rendering and writing. |
| [`eval.py`](src/smartrodent/eval.py) | `EvaluationImages`, `FeatureExtractor`, `EvaluationFeatures`, `TsneComparison`: fixed-subset representation evaluation. |
| [`base.py`](src/smartrodent/base.py), [`utils.py`](src/smartrodent/utils.py) | Shared protocols/base classes and path/configuration utilities. |
| [`__init__.py`](src/smartrodent/__init__.py) | Convenience exports for filters, Ultralytics trainers, and path resolution. Modern experiment APIs are imported from their modules. |

Some data-processing scripts also contain workflow-specific pandas/file
operations. The modern training scripts delegate training to the library.

## CLI reference

### Invocation and arguments

Every script below takes **one required positional YAML path**, plus argparse's
`-h`/`--help` option:

```text
python scripts/<script>.py [-h] CONFIGURATION
```

The positional argument is named `config` in most help messages and
`configuration` in `train_classifier.py` and `train_two_stage.py`; its usage is
the same. There is no `--config` flag, installed `smartrodent` command, or general
command-line parameter-override interface. Set experiment parameters in YAML.

For help without starting a workflow:

```bash
python scripts/train_supcon.py --help
python scripts/train_classifier.py --help
python scripts/train_two_stage.py --help
```

Configuration paths and directory/weight paths inside configs are interpreted
relative to the **working directory**, not the YAML file's directory. The
examples assume the repository root. Metadata CSV image references follow a
different convention described under [portable dataset paths](#portable-dataset-paths).

### Acquisition and data preparation

For each row, invoke `python scripts/<script> configs/<configuration>`.
These commands download, infer, or write dataset artifacts; they are not
validation-only commands.

| Script | Configuration(s) under `configs/` | Input, operation, and output |
| --- | --- | --- |
| [`download_inaturalist_data.py`](scripts/download_inaturalist_data.py) | [`data_config_full.yaml`](configs/data_config_full.yaml), [`dataset_config_srilanka.yaml`](configs/dataset_config_srilanka.yaml) | Reads `data.inaturalist`; selects observations and downloads per-species `records.csv` and `imgs/` into `output_path`. |
| [`download_lila_data.py`](scripts/download_lila_data.py) | [`data_config_lila.yaml`](configs/data_config_lila.yaml) | Reads `data.lila`; caches index/taxonomy/dataset metadata and downloads selected camera-trap images into class directories. |
| [`filter_data_vlm.py`](scripts/filter_data_vlm.py) | [`filter_data_vlm_config_animal.yaml`](configs/filter_data_vlm_config_animal.yaml), [`filter_data_vlm_config_birds.yaml`](configs/filter_data_vlm_config_birds.yaml), [`filter_data_vlm_config_snakes.yaml`](configs/filter_data_vlm_config_snakes.yaml), [`filter_data_vlm_config_thermal.yaml`](configs/filter_data_vlm_config_thermal.yaml) | Reads species CSVs and source images; runs the selected Ollama/vLLM backend and writes task-labelled CSVs under `paths.output_root`. |
| [`materialize_filtered_data.py`](scripts/materialize_filtered_data.py) | [`materialize_filtered_data.yaml`](configs/materialize_filtered_data.yaml) | Partitions records by rejection flags; writes `kept/` and `rejected/` metadata and relative image symlinks for review. |
| [`reconcile_filtered_data.py`](scripts/reconcile_filtered_data.py) | [`reconcile_filtered_data.yaml`](configs/reconcile_filtered_data.yaml) | Reads original metadata and an already prepared human-corrected image tree; validates associations and writes `records_corrected.csv`. Does not perform the human review itself. |
| [`split_dataset.py`](scripts/split_dataset.py) | [`dataset_splitter.yaml`](configs/dataset_splitter.yaml) | Discovers per-species CSVs using `paths.records_glob`; assigns train/validation/test rows and writes per-species split metadata. |
| [`sample_training_dataset.py`](scripts/sample_training_dataset.py) | [`training_dataset_sampler.yaml`](configs/training_dataset_sampler.yaml) | Oversamples training rows using the script's custom rule; preserves non-training splits and saves metadata plus config/script snapshots. |
| [`downsample_dataset.py`](scripts/downsample_dataset.py) | [`downsample_dataset.yaml`](configs/downsample_dataset.yaml) | Alternative metadata operation: caps each species CSV at `max_images_per_class` using `rng_seed`; does not copy images. Not a declared DVC stage. |
| [`create_yolo_dataset.py`](scripts/create_yolo_dataset.py) | [`create_yolo_detector_dataset.yaml`](configs/create_yolo_detector_dataset.yaml), [`create_yolo_classifier_dataset.yaml`](configs/create_yolo_classifier_dataset.yaml) | Selects the class-named creator under `data`; runs SpeciesNet to write detector labels/source symlinks or classifier image crops into a new destination. |

Examples—choose the relevant command, rather than executing this block as an
unattended pipeline:

```bash
python scripts/download_inaturalist_data.py configs/data_config_full.yaml
python scripts/download_lila_data.py configs/data_config_lila.yaml
python scripts/filter_data_vlm.py configs/filter_data_vlm_config_animal.yaml
python scripts/materialize_filtered_data.py configs/materialize_filtered_data.yaml
python scripts/reconcile_filtered_data.py configs/reconcile_filtered_data.yaml
python scripts/split_dataset.py configs/dataset_splitter.yaml
python scripts/sample_training_dataset.py configs/training_dataset_sampler.yaml
python scripts/downsample_dataset.py configs/downsample_dataset.yaml
python scripts/create_yolo_dataset.py configs/create_yolo_classifier_dataset.yaml
```

### Training and tuning

These commands initialize models and start actual training. They do not expose
`--dry-run`, `--validate-only`, `--resume`, `--epochs`, or `--device` options.

| Command | Workflow and reported output |
| --- | --- |
| `python scripts/train_supcon.py configs/train_supcon_config.yaml` | Fresh standalone SupCon runs; prints each final `last.pt` path. |
| `python scripts/train_classifier.py configs/train_classifier_config.yaml` | Fresh standalone CE classifier candidates; prints final checkpoints and, with validation, the selected `best.pt`. |
| `python scripts/train_two_stage.py configs/train_two_stage_config.yaml` | SupCon sweep, encoder selection, then CE sweep; prints final classifier checkpoints and the selected classifier's `best.pt`. |
| `python scripts/train_yolo_detector.py configs/train_yolo_detector_config.yaml` | Ultralytics detector training followed by the configured export; prints the export location. |
| `python scripts/train_yolo_classifier.py configs/train_yolo_classifier_config.yaml` | Expands custom classifier sweeps, trains with Ultralytics, then exports each run. |
| `python scripts/tune_yolo_classifier.py configs/train_yolo_classifier_config.yaml` | Expands configurations, runs Ultralytics tuning using `tune_kwargs`, then invokes export. |

The last three scripts are separate from the modern SupCon/CE experiment
implementation. They currently set `CUDA_VISIBLE_DEVICES=0` in their process;
keep that GPU masking in mind when selecting native Ultralytics devices.

## Configuration guide

The three modern training examples contain inline comments listing node
choices, types, and ranges:

- [`train_supcon_config.yaml`](configs/train_supcon_config.yaml): standalone SupCon.
- [`train_classifier_config.yaml`](configs/train_classifier_config.yaml): standalone CE and classifier sweeps.
- [`train_two_stage_config.yaml`](configs/train_two_stage_config.yaml): unified two-stage experiment.

These are different schemas from the older
[`train_yolo_classifier_config.yaml`](configs/train_yolo_classifier_config.yaml)
and [`train_yolo_detector_config.yaml`](configs/train_yolo_detector_config.yaml),
which use `train_kwargs`, `tune_kwargs`, and `export_kwargs`. Do not interchange
the files merely because both workflows train a YOLO-based classifier.

### Configuration families

| Family | Main structure/options |
| --- | --- |
| iNaturalist | `data.inaturalist`: `output_path`, species, years or year range, observation/license filters, caps, seed, cache/retry settings. |
| LILA | `data.lila`: output path, index/taxonomy URLs, dataset-to-license mapping, allowed licenses, classes, cap, seed. |
| VLM filters | `backend: ollama` or `vllm`, `taskname`, labels/columns, input/output roots, prompts, and backend options. |
| Review materialization/reconciliation | Input/output paths, rejection-column prefix or corrected partition/column lists, accepted image suffixes. |
| Splitting/sampling/downsampling | Input/output metadata roots, record discovery, split/sampling rules, seeds, or per-class caps. |
| YOLO dataset creators | Exactly one key under `data`: `YoloDetectorDatasetCreatorFromSpeciesnet` or `YoloClassifierDatasetCreatorFromSpeciesnet`. |
| Modern standalone training | Top-level data/output settings plus transforms, AdamW options, validation, tracking, and model/initialization settings. |
| Two-stage training | Root `output_directory`, shared `data`, `supcon`, `classifier`, and optional shared `mlflow`. |
| Ultralytics wrappers | Dataset/model, trainer options, native train/tune/export kwargs, and optional wrapper-specific MLflow settings. |
| Model cards | [`model_card_template.yaml`](configs/model_card_template.yaml): model, classes, validation, provenance, and editable narrative. |

### Sweeps and coupled values

The modern training CLIs and Ultralytics classifier train/tune scripts use the
custom loader in `smartrodent.config_utils`. Independent `!sweep` nodes form a
Cartesian product. `!coupled-sweep` pairs values with another sweep by position,
without adding an independent dimension:

```yaml
# Fragment within a standalone classifier configuration:
initialization:
  head:
    replace: !sweep
      values: [linear, full]
    source: !sweep
      values: [random, yolo_checkpoint]
    checkpoint: !coupled-sweep
      target: initialization.head.source
      values: [null, /path/to/species_classifier.pt]
```

This fragment describes four head combinations. Coupled lists have the same
length as their target, and targets are full dotted paths from the document
root. In a two-stage document, the corresponding target is
`classifier.initialization.head.source`. Replace a sweep node with a scalar to
choose one value. Stage sweeps in the two-stage workflow expand independently;
classifier choices do not multiply SupCon training runs.

The loader also provides `!range`, `!random_uniform`, `!reference`, and
`!pyobject` helpers. These serve the broader configuration/tuning machinery;
they are not all expanded as Cartesian training dimensions by `ConfigHandler`.
In particular, `!random_uniform` draws values while loading and `!pyobject`
imports Python objects. Only load trusted configuration files. Plain
`yaml.safe_load()` does not provide these custom tags without registering the
project loader.

### Shared modern training options

| Node | Choices/meaning |
| --- | --- |
| `transforms` | torchvision v2 keyword dictionaries; `image_size` is a positive square size. Augmentation `p` values range from 0 to 1. |
| `optimizer` | Native AdamW kwargs, including a positive `lr`; this is not an optimizer-name selector. |
| `epochs`, `workers`, `seed`, `device` | Maximum epoch count, nonnegative loader worker count, integer seed in `[0, 2**32)`, and PyTorch device such as `cpu`/`cuda:0`. |
| `validation_dataset` | Existing class-folder validation split, or `null` for standalone training-only operation. Two-stage selection uses validation. |
| `validation_metrics` | Named sklearn metric specifications, or `null`. Each receives true/predicted labels, not probabilities. `function` names the metric; other entries are native kwargs. |
| `tracked_metric` | SupCon native loss statistic/custom alias, or CE's `ce_loss_mean`/custom alias. SupCon defaults to `supcon_loss_mean`; CE defaults to `ce_loss_mean`. |
| `tracked_mode` | `min`, `max`, or `null`/omitted for inferred loss-min/score-max direction. |
| `scheduler` | `null`/omitted for constant LR, or `name: ReduceLROnPlateau` with `parameters`. |
| `early_stopping` | `null`/omitted, or positive `patience`, nonnegative absolute `tolerance`, and `comparison: less`/`greater`. |
| `mlflow` | `null`/omitted, or tracking/registry settings described below. |
| `model_card_template` | Optional template path; defaults to `configs/model_card_template.yaml`. |

Native SupCon validation statistics are `supcon_loss_min`,
`supcon_loss_mean`, `supcon_loss_std`, and `supcon_loss_max`; custom aliases
cannot replace them. CE reserves `ce_loss_mean`. Metric specs can include, for
example, `function: f1_score`, `average: macro`, and `zero_division: 0`.

Use `min`/`less` for loss and `max`/`greater` for scores where larger is better.
Align scheduler mode and early-stopping comparison with the selected direction.
Scheduler `patience: 2` tolerates two bad epochs and reduces LR after the third;
early-stopping patience counts consecutive insufficient improvements. Scheduler
thresholds and stopping tolerance are independent. Reducing LR does not reset
early-stopping patience. Best-checkpoint selection uses strict improvement;
tolerance controls stopping, not whether a slightly better checkpoint is saved.

Validation-driven metrics, scheduling, and stopping need a validation split.
For standalone training without validation, disable custom validation metrics,
scheduling, and stopping. SupCon also leaves `tracked_metric`/`tracked_mode`
omitted or null; CE uses `ce_loss_mean` rather than a custom validation alias.

Both modern training loops use FP32 and start fresh optimizer state; their
scripts do not expose AMP or training resumption. Saved optimizer/scheduler
state does not imply that a resume CLI exists.

### Inspect configuration without training

There is no validation-only CLI flag, but the library configuration loaders
can expand and validate settings without constructing models or starting runs:

```python
from pathlib import Path
from smartrodent.supcon_training import load_training_configurations
from smartrodent.classifier_sweeps import ClassifierSweepConfiguration
from smartrodent.two_stage_training import TwoStageConfiguration

supcon_runs = load_training_configurations(Path("configs/train_supcon_config.yaml"))
classifier_sweep = ClassifierSweepConfiguration.from_yaml("configs/train_classifier_config.yaml")
pipeline = TwoStageConfiguration.from_yaml("configs/train_two_stage_config.yaml")
print(len(supcon_runs), len(classifier_sweep.configurations))
```

Parsing is not a complete donor/architecture/device compatibility check. Creating
an experiment additionally prepares datasets, transforms, scorers, and templates;
model-dependent checks occur when actual models are initialized.

## Dataset workflow

### Acquisition and filtering

The iNaturalist downloader reads `data.inaturalist` and stores one species
folder with `records.csv` and `imgs/`. Choose species, observation years,
quality grade, allowed licenses, and download caps in its config. Configurations
are copied into outputs for provenance.

LILA uses dataset-level licenses and a taxonomy mapping to match configured
Latin taxa or raw labels such as `empty` and `vehicle`. Its image cap is per
class **and per LILA dataset**. Sequence-aware sampling uses the configured seed;
metadata is cached under the LILA output. LILA has its own DVC download stage but
is not automatically merged into the iNaturalist filtering branch.

The animal, flying-bird, and snake VLM filters annotate metadata in sequence.
The thermal-filter config is an additional direct-script workflow, not a stage
in the current DVC graph. Backend selection, prompts, model names, batching, and
input/output paths come from each filter config.

Materialization exposes kept/rejected images through symlinks. Human review is
a separate step: prepare the corrected partition tree expected by the
reconciliation config, then reconcile it against the original records.
Reconciliation does not manufacture decisions or initialize that corrected tree.
Splitting consumes corrected metadata; oversampling operates on training rows
rather than resampling validation/test data. Keep held-out test data out of model
selection and champion comparisons.

### Portable dataset paths

Keep original image roots and stage directories as direct children of one
shared dataset root, normally `datasets/`. CSV `image_path` values are relative
to that root, for example `inaturalist/Mus musculus/imgs/123_0.jpg`, **not** to
the CSV or working directory.

Materialized review images and YOLO detector images use symlinks whose targets
are relative to each link's directory. Moving/copying the whole dataset tree
while preserving symlinks preserves those references; relocating one stage or
rearranging the internal tree does not. Update configured input/output paths
when relocating the tree.

Legacy absolute image references remain readable. Existing CSVs and symlinks
are not automatically converted, and there is no path-prefix migration utility.

### YOLO dataset creation and label order

Dataset creators accept `path_to_image_data`, `dataset_output_path`, optional
`class_names` (`null` discovers species), and SpeciesNet inference options.
The classifier creator writes animal crops; the detector creator writes
labels and links to source images. Both preserve existing split assignments and
oversampled rows and require a new destination directory. Model weights load
lazily when inference is first needed, not during configuration loading.

```python
from smartrodent.yolo_dataset_creation import YoloClassifierDatasetCreatorFromSpeciesnet

creator = YoloClassifierDatasetCreatorFromSpeciesnet.from_config(
    "configs/create_yolo_classifier_dataset.yaml"
)
# Explicitly writes dataset outputs and may download/load SpeciesNet weights:
# creator.create()
```

Ultralytics classification derives IDs from **alphabetically sorted species
folders under `train/`**, not `data.yaml`'s `names` order. Its classification
wrapper uses a YAML's `path` to locate the dataset. Detection instead uses the
class mapping in `data.yaml`. The modern image datasets also assign alphabetical
IDs for a supplied species list and reuse the established mapping in validation.
Interpret predictions using the model/checkpoint mapping, not the order in an
arbitrary input list.

## Training workflows

### Standalone SupCon

```bash
python scripts/train_supcon.py configs/train_supcon_config.yaml
```

`classes_per_batch` (P) selects distinct classes and `samples_per_class` (M)
selects source images per class: a batch contains P*M source selections and
2*P*M independently augmented views. `samples_per_epoch` counts source
selections and rounds down to complete batches. `dropout` and `output_features`
configure the projection head; `temperature` controls SupCon loss, and `freeze`
lists native YOLO layer indices to freeze.

Validation uses a fixed seeded P-K cohort with deterministic single-view
preprocessing and complete batches. Its loss statistics summarize validation
batches, not an exhaustive CE-style traversal of every image. SupCon optimizes
contrastive representations, not a CE classification objective; optional
classification metrics are diagnostics from the classifier outputs.

Each configured run trains independently. Temperature must remain fixed within
a SupCon sweep. A single run writes directly to `output_directory`; multiple
runs receive `run_001`, `run_002`, etc. under their configured output roots.

### Standalone CE classifier

```bash
python scripts/train_classifier.py configs/train_classifier_config.yaml
```

The encoder format is explicit:

| `initialization.encoder.source` | `checkpoint` format |
| --- | --- |
| `supcon` | Standalone SmartRodent SupCon checkpoint. |
| `ultralytics` | Local native Ultralytics classification checkpoint. |
| `classifier` | Portable SmartRodent CE classifier checkpoint. |

`head.replace: linear` replaces only the final linear layer.
`head.replace: full` replaces the whole native `Classify` block, including its
convolution/BatchNorm, pooling, dropout, and linear layer. `head.source: random`
creates fresh head weights; `yolo_checkpoint` transfers them from an explicit
native classifier donor. A random head accepts an omitted/null donor; transfer
needs compatible dimensions and the exact target class names/IDs. A stock
ImageNet head is not a species-matched donor, even if its backbone is a valid
starting encoder.

`freeze_encoder: true` freezes parameters before the chosen head boundary;
`false` enables fine-tuning. `encoder_batchnorm: eval` fixes encoder running
statistics, while `adapt` updates them during training, including when encoder
parameters are frozen. The selected trainable head follows normal train mode.

CE uses ordinary shuffled, single-view batches and retains the final partial
batch. Validation traverses the full split, computes sample-weighted mean CE,
and evaluates configured label metrics. Each candidate owns a fresh model and
optimizer. With validation, candidates are ranked by the configured objective;
first-in-expansion-order wins a local tie. Without validation, standalone
training is allowed but no validated sweep winner/champion is selected.

A single candidate writes checkpoints at the output root; multiple candidates
use `run_001`, etc. The sweep also writes root `config.yaml` and `selection.yaml`.

### Two-stage SupCon → CE

```bash
python scripts/train_two_stage.py configs/train_two_stage_config.yaml
```

The root owns shared data, output root, and MLflow settings. `supcon` and
`classifier` contain their respective stage settings rather than copies of the
shared paths/classes. The classifier encoder source in this schema is the
placeholder `selected_supcon`, not the three standalone encoder-format choices.

Execution:

1. Expand and validate both stage configurations; prepare each candidate's
   datasets, transforms, scorers, and templates before starting training.
2. Train every SupCon candidate. Selection uses the configured validation metric
   and direction, with a consistent scorer definition/options and temperature.
3. Select the winning **best-epoch** checkpoint, not its final `last.pt`.
4. Bind that same selected encoder/checksum/provenance to every classifier
   candidate, then train the CE sweep with fresh model/optimizer state per run.
5. Select the CE winner and, if registration is enabled, compare it with the
   historical champion only after all candidates finish publishing.

SupCon and classifier sweeps are not a cross-stage Cartesian product. With S
SupCon candidates and C classifier candidates, execution trains S SupCon models
and C classifiers, not S*C classifiers. Stage failures propagate rather than
being skipped. Two-stage selection uses existing validation data, never a test
split.

### Ultralytics detector/classifier wrappers

The `train_yolo_*`/`tune_yolo_classifier.py` commands delegate training and tuning
to Ultralytics through `smartrodent.training`. Their configs contain native
`train_kwargs`, `tune_kwargs` (including tuner `space`/`iterations`), and
`export_kwargs` (for example `format: onnx`). Outputs follow native
`project`/`name` settings, not the modern experiment directory convention.
These wrappers use a different MLflow schema and do not implement the modern
classifier champion-comparison workflow.

## Checkpoints, results, and model cards

### Modern experiment artifacts

Each SupCon/CE run stores:

| File | Meaning |
| --- | --- |
| `config.yaml` | Concrete configuration for that run. |
| `history.yaml` | Completed-epoch loss/validation history and available learning-rate fields. |
| `last.pt` | Latest completed epoch. Replaced after each completed epoch. |
| `best.pt` | Best epoch by strict improvement of the tracked objective. |
| `best.model_card.yaml`, `best.model_card.md` | Selected best snapshot's metadata, validation results, and provenance. |

There are no automatically retained `epoch_0001.pt`, `epoch_0002.pt`, etc.
Checkpoints embed architecture, model state, label mapping, configuration,
optimizer state, and optional scheduler state; CE also records encoder
provenance. The scheduler steps before saving, so saved optimizer/scheduler
state reflects the next epoch's learning rate. SupCon history/tracking exposes
`learning_rate` and `next_learning_rate`; CE history records
`next_learning_rate`.

For a two-stage experiment with multiple classifier candidates:

```text
<output_directory>/
  config.yaml
  selection.yaml
  supcon/
    run_001/                 # SupCon always uses numbered candidate directories
      config.yaml
      history.yaml
      last.pt
      best.pt
      best.model_card.yaml
      best.model_card.md
  classifier/
    config.yaml
    selection.yaml
    run_001/                 # Numbered for multiple CE candidates
      ...                    # Same per-run artifact names
    run_002/
      ...
```

With one CE candidate, its artifacts live directly under `classifier/`.
Selection summaries identify the results of the current invocation, including
winning paths, scores, and selected-encoder identity; selection does not scan
old checkpoints to infer a winner. Completed artifacts/registry versions may
remain after a later failure. Use a fresh output root for a new experiment and
do not write concurrently to one run directory.

Checkpoint publication is atomic per file, not a transaction across all
artifacts. The storage implementation uses hard links for `best.pt`; the
filesystem must support them. It does not prune arbitrary old outputs or provide
power-loss durability guarantees.

### Reloading without the original starting weights

Portable checkpoints reconstruct their architecture and trained state without
the original pretrained file or notebook session:

```python
from smartrodent.supcon_model import YoloSupConModel
from smartrodent.classifier_model import YoloClassifierModel

supcon = YoloSupConModel.from_checkpoint(
    "runs/supcon_standalone_xl_srilanka_lr-00001/best.pt", device="cpu"
)
classifier = YoloClassifierModel.from_checkpoint(
    "runs/classifier_xl_srilanka/run_001/best.pt", device="cpu"
)
print(classifier.label_names)  # Species name -> integer ID.
# classifier(images) returns raw logits, including in evaluation mode.
# Release the SupCon feature hook when finished using that wrapper:
supcon.finalize()
```

Use the appropriate loader for each format. In particular, a modern portable
CE checkpoint is not a native YOLO head-donor checkpoint. Native Ultralytics
and logged pickled models must come from trusted sources. Portability does not
remove the need for compatible installed PyTorch/Ultralytics/library versions.

For library-driven execution, use `SupConExperiment(configuration).run_with_result()`,
`ClassifierExperiment(configuration).run_with_result()`,
`ClassifierSweepExperiment(sweep).run()`, or
`TwoStageExperiment(configuration).run()`. These **execute training**. The
single-experiment `run()` methods return `last.pt`; structured results expose
best/last paths, best epoch/value, tracked metric/direction, validation identity,
and tracking information.

### Model cards and representation evaluation

[`configs/model_card_template.yaml`](configs/model_card_template.yaml) supplies
editable narrative and the `model`, `validation`, and `provenance` mappings.
Training populates class mappings, selected checkpoint identity/epoch/SHA-256,
validation results, Git commit, optional dataset revision, encoder provenance,
and available MLflow run/model/version IDs. Optional tracking/data identifiers
remain `null` when unavailable. Narrative fields are preserved; `model_card_template` selects another
compatible template.

Cards are saved beside `best.pt`. Tracking uploads best-model artifacts; registry
publication also attaches cards to the logged model and uses the Markdown as
the model-version description. A Git commit tag identifies HEAD, not a snapshot
of uncommitted source changes.

Generic rendering is also available independently:

```python
from pathlib import Path
from smartrodent.model_card import load_card_template, render_model_card

card = load_card_template(Path("configs/model_card_template.yaml"))
card["model"]["name"] = "rodent-species"
markdown = render_model_card(card)
```

[`notebooks/supcon_training.ipynb`](notebooks/supcon_training.ipynb) uses the
library evaluation helpers to compare pretrained pooled, trained pooled, and
trained projected representations on the same balanced image subset. Set its
checkpoint/data paths to the experiment being inspected.

## DVC workflow

### Declared graph versus current example paths

[`dvc.yaml`](dvc.yaml) declares the iNaturalist processing branch and an
independent LILA download:

```text
download_inaturalist
  -> filter_animal -> filter_flying_bird -> filter_snake
  -> materialize_filtered_data -> reconcile_filtered_data
  -> split_dataset -> sample_training_dataset
       -> create_yolo_detector_dataset -> train_yolo_detector
       -> create_yolo_classifier_dataset -> train_yolo_classifier
                                         -> tune_yolo_classifier

download_lila  (independent)
```

Both classifier training and tuning depend on the classifier dataset; tuning
is not declared to depend on completion of `train_yolo_classifier`.

**The graph is not currently a guarantee that all supplied config paths line
up.** Several configs target Sri Lanka directories ending in `_srilanka`, while
DVC declares unsuffixed paths. For example, `filter_animal` declares
`datasets/inaturalist`/`datasets/stage1_filtered_animal`, but its YAML reads
`datasets/inaturalist_srilanka` and writes
`datasets/stage1_filtered_animal_srilanka`. The detector-creation config
currently reads stage 6 and writes `stage8_yolo_detector_not_oversampled`,
whereas its DVC stage declares a stage-7 dependency and `stage8_yolo_detector`
output. Classifier training/tuning project paths also differ from declared DVC
outputs. Align configs and stage `deps`/`outs` for the intended dataset before
reproduction; DVC does not rewrite paths inside YAML for you.

The three modern training scripts (`train_supcon.py`, `train_classifier.py`,
`train_two_stage.py`), downsampling, and thermal filtering **do not currently
have DVC stages**. Execute those directly or explicitly add a suitable stage
when integrating them; do not assume `dvc repro` runs them.

### Inspect, reproduce, and share

Inspect the declared graph/status without executing its stages:

```bash
dvc dag
dvc status
```

Once paths, inputs, backends, and the manual-review outputs are ready, reproduce
one desired target:

```bash
# Executes the target and required out-of-date upstream dependencies.
dvc repro train_yolo_detector
# Or just the configured acquisition target and its dependencies:
dvc repro download_lila
```

`dvc repro` is not a dry run. It can download large datasets/weights, run VLM or
SpeciesNet inference, replace managed outputs, and train/export models. A bare
`dvc repro` considers the whole declared pipeline, including both download
branches and classifier tuning. Human decisions needed by reconciliation must
already exist; automatic reproduction cannot replace review.

DVC uses declared dependencies, parameters, output hashes, and a generated
`dvc.lock` to decide what needs reproduction or restoration from cache. The
download stages declare YAML parameter sections; most other stages list their
whole config as a dependency. Only declared dependencies are tracked, so code
changes outside a stage's listed files are not automatically complete
invalidation signals.

With an intentionally configured remote, `dvc push` uploads cached outputs and
`dvc pull` restores them. These are explicit transfer/write operations, not
inspection commands. Keep remote credentials out of Git. Version the intended
source/configuration/DVC metadata separately from large image/model artifacts;
DVC caching and MLflow tracking have distinct roles. Shared remote definitions
can live in `.dvc/config`; keep machine-specific credentials in environment or
local configuration such as `.dvc/config.local`, not committed metadata. This
README does not assume a remote is configured.

## MLflow workflow

### Configure tracking and open the UI

Modern SupCon, CE, and two-stage configs accept:

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: rodent-experiments
  registered_model_name: rodent-classifier  # null disables registry publication
  model_artifact_name: classifier
  champion_alias: champion
  dataset_revision: null  # Optional immutable input-data revision
```

Omit `mlflow` or set it to `null` to disable tracking/publication. With tracking
on and `registered_model_name: null`, run metrics and best-model artifacts are
still logged. Use an HTTP(S) tracking-server URI instead of SQLite when working
with a remote service. Keep authentication in the environment/service setup,
not committed configuration files.

From the same working directory used for a relative SQLite URI, start the local
UI in another activated terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>. A local SQLite backend does not require a separate
server for training to log to it. Starting the UI/server can initialize backend
files. `mlflow.db` holds tracking/registry metadata; artifact payloads live in
the backend's artifact store, not inside the database. Preserve both when moving
or backing up experiments. Relative database/artifact locations depend on the
working directory.

### Run ownership and logged data

- **Standalone SupCon:** one root run per expanded training candidate.
- **Standalone CE sweep:** one root run per classifier candidate; no sweep-parent
  MLflow run is created by this CLI.
- **Two-stage:** one pipeline parent with SupCon and classifier child runs.
  The parent logs configuration/selection summaries; children log stage metrics
  and selected best-model artifacts. SupCon children are encoder candidates,
  not registered CE classifier versions.

Runs record stage-specific parameters and epoch metrics, selected best values
and epochs, source Git/data tags, and best checkpoint/cards. CE metrics use
`training/ce_loss_mean` and `validation/<metric>`; SupCon records native training
loss statistics, validation metrics, and current/next learning rates. Local
`config.yaml` and `history.yaml` remain useful independently of the tracking UI.

Modern experiments own their MLflow run contexts. An unrelated active run or
`MLFLOW_RUN_ID` requesting implicit resumption is rejected rather than silently
reused. A supplied `dataset_revision` is recorded, not automatically discovered
from DVC; set it deliberately when comparing versioned input data.

### Registration and classifier champion promotion

Modern CE publication and ranking are separate:

1. Each candidate's selected `best.pt` is restored and published as a registered
   model version, with completed cards and comparison metadata.
2. A sweep defers alias movement until **all** candidates have trained and
   published successfully. A candidate failure prevents promotion of a partial
   sweep winner; already created runs/versions are not rolled back.
3. The local validated winner is compared with the version currently referenced
   by `champion_alias`. If no incumbent exists, the validated winner can become
   the champion.
4. Historical comparison requires matching metric name, exact scorer/options,
   direction, resolved validation-split path, canonical class population,
   validation status, and dataset revision (including whether it is absent).
5. Promotion requires **strict improvement**. A historical tie retains the
   incumbent. Missing/legacy/incompatible metadata raises an error instead of
   silently overwriting the alias. Choose another model/alias or explicitly
   resolve the historical metadata when that is intended.

Training-only CE runs can be registered but are not promoted as validated
champions. `selection.yaml` distinguishes the local winner from the retained or
promoted registered champion; the best model in this invocation need not beat
the historical one. Remote publication is not transactional across model
version creation, artifact upload, metadata, and alias updates.

Standalone **SupCon registration has different semantics**: it publishes each
run's best model and assigns the alias for validated runs, without CE's strict
historical comparison or sweep-wide deferred promotion. Use separate registry
names for encoders and final CE classifiers. Two-stage orchestration disables
SupCon registration and applies the CE policy to classifier candidates only.

### Ultralytics wrapper tracking

The older `train_yolo_*`/`tune_yolo_classifier.py` wrappers use this distinct
schema:

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db
  train_experiment_name: yolo-training
  tune_experiment_name: yolo-tuning
```

The wrapper selects the operation's experiment, sets MLflow process settings,
and enables Ultralytics' MLflow integration. Its native callbacks/logging and
output/export behavior are separate from the modern model-card/champion
workflow; the modern registry fields do not belong in this mapping.

## Testing

Install the development and experiment extras, activate the environment, and
run from the repository root. For the ordinary isolated suite with line and
branch coverage, excluding live integration markers:

```bash
pytest -q -m 'not network and not vlm_integration and not speciesnet_integration' \
  --cov=smartrodent --cov-report=term-missing --cov-branch
```

For a line-coverage-only report, omit `--cov-branch`. Hypothesis is a first-class
part of the tests. A bare `pytest` includes network-marked tests unless they are
explicitly deselected; it does not execute the full DVC production pipeline.

| Marker | Execution requirements |
| --- | --- |
| `network` | Live API/network tests; select with `pytest -m network`. |
| `vlm_integration` | Real Ollama/vLLM inference; opt in with `SMARTRODENT_RUN_VLM_INTEGRATION=1 pytest -m vlm_integration`. Use `-k ollama` or `-k vllm` to select a backend. |
| `speciesnet_integration` | Real SpeciesNet inference; opt in with `RUN_SPECIESNET_INTEGRATION=1 pytest -m speciesnet_integration`. `SPECIESNET_MODEL` can select local weights. |

VLM tests accept `SMARTRODENT_OLLAMA_MODEL`, `SMARTRODENT_VLLM_MODEL`,
`SMARTRODENT_VLLM_GPU_MEMORY_UTILIZATION`, `SMARTRODENT_VLLM_MAX_MODEL_LEN`,
`SMARTRODENT_VLLM_MAX_NEW_TOKENS`, `SMARTRODENT_INTEGRATION_RODENT_IMAGE`, and
`SMARTRODENT_INTEGRATION_EMPTY_IMAGE` overrides. Integration tests can download
weights and use substantial GPU resources; enable them intentionally.

## Troubleshooting and limitations

| Symptom | Check |
| --- | --- |
| Package/module import fails | Activate the editable project environment; include `experiments` for the documented CLI environment, even if tracking is disabled. |
| Config or input path is not found | Run from the repository root or use appropriate absolute paths; paths are not rebased to the YAML directory. Sample configs refer to existing local datasets/checkpoints, not bundled assets. |
| Unknown YAML tag | Use a sweep-aware CLI/library loader; not every script accepts custom training tags. |
| Dataset class is missing/empty | Verify exact species-directory names, selected population, split paths, and indexed suffixes. Modern datasets default to `.jpg` and `.png`. |
| Head transfer or checkpoint loading fails | Match source format, donor dimensions, and exact labels/IDs. Starting ImageNet weights are not automatically valid species-head donors. |
| CUDA/backend failure | Check installed hardware/runtime and modern `device` settings; legacy YOLO scripts mask visible GPUs to device 0. |
| No validated winner or champion | Check validation configuration; training-only results are not validated candidates. |
| Champion metadata rejected | Compare scorer/options, split identity, labels, direction, and revision; do not erase provenance simply to force promotion. |
| DVC reports missing outputs or ignores a changed input | Compare the script's configured paths with declared `deps`/`outs` and listed code dependencies; current example configs and graph are not fully aligned. |
| Rerun replaced outputs or failed after partial publication | Use a new output root; inspect the current summary, completed artifacts, and tracking runs. There is no automatic resume or cross-artifact/registry rollback. |
| Checkpoint best-state publication fails on a filesystem | Modern checkpoint storage requires hard-link support and a single writer per run directory. |
