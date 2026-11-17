# SmartRodent experiments

Tools for building a licensed, species-organized wildlife image dataset from
iNaturalist observations. The maintained package currently provides the
iNaturalist downloader and its DVC pipeline.

## Repository layout

```text
config/dataset_full.yaml              Download configuration and DVC parameters
dvc.yaml                              iNaturalist download pipeline
scripts/download_inaturalist_data.py  Command-line download entry point
src/smartrodent/inaturalist.py        Downloader implementation
tests/                                Pytest suite
requirements_processing.txt           Optional legacy notebook/processing tools
requirements_ammico.txt               Optional AMMICO/VLM tools
```

## Installation

The project requires Python 3.13 or newer and uses [uv](https://docs.astral.sh/uv/)
for environment and package management.

Install the runtime dependencies from the lockfile:

```bash
uv sync
```

For development, including the test tools:

```bash
uv sync --extra dev
```

`requirements_processing.txt` and `requirements_ammico.txt` are deliberately
not part of the main environment. Install one only when working on its optional
workflow:

```bash
uv pip install -r requirements_processing.txt
# or
uv pip install -r requirements_ammico.txt
```

## iNaturalist download configuration

The downloader reads a YAML configuration. [`config/dataset_full.yaml`](config/dataset_full.yaml)
contains the current full-dataset configuration. Before running it, set
`inaturalist.output_path` to a directory on your machine.

The important settings are:

- `output_path`: destination for the downloaded dataset.
- `species`: scientific names to download.
- Either `years`, or the inclusive `first_year` / `last_year` range.
- `quality_grade`, `allowed_licences`, `seed`, and `max_img_num`: observation
  filtering and download-limit settings.

The downloader creates one directory per species, containing `records.csv` and
an `imgs/` directory. It copies the YAML configuration into the output
directory for provenance. Downloaded data stays local and is not committed.

## Portable dataset paths

Keep the original `inaturalist/` directory and every stage directory as direct
children of one shared dataset root, such as `datasets/`. CSV `image_path` values
are relative to that root, for example `inaturalist/Mus musculus/imgs/123_0.jpg`.
They are not relative to the CSV or the working directory, and keep the same
meaning across stages.

Materialized review images and YOLO detector images use symlinks whose targets
are relative to each link's directory. Moving or copying the entire dataset tree
while preserving symlinks keeps these links and CSV references valid. Moving an
individual stage or rearranging the internal tree is not supported. Configuration
input/output directory settings still need to point at the relocated stages.

Legacy absolute image paths remain readable, but newly written CSVs use relative
paths. Existing data is not automatically converted; old absolute CSV references
and symlinks are not portable until their outputs are regenerated. No path-prefix
migration utility is needed for datasets produced with this convention.

## DVC usage

[`dvc.yaml`](dvc.yaml) declares the complete RGB pipeline: iNaturalist download,
three VLM filtering steps, materialization, human-decision reconciliation,
train/validation/test splitting, oversampling, YOLO detector and classifier
dataset creation, detector/classifier training, and classifier tuning. The
detector and classifier branches share the stage-7 sampled records, then run
independently from their respective stage-8 datasets.

Inspect the graph or status without running any stage:

```bash
uv run dvc dag
uv run dvc status
```

`dvc repro` executes the requested stage and its missing dependencies; it can
download data, run VLM/SpeciesNet inference, and train models. Use it only when
those writes and compute costs are intended, for example:

```bash
uv run dvc repro train_yolo_detector
```

To run only the downloader without DVC:

```bash
uv run python scripts/download_inaturalist_data.py configs/data_config_full.yaml
```

## YOLO dataset creation

Both dataset creators implement the `Configurable` protocol and provide
`from_config(config_path)`. See
[`configs/create_yolo_detector_dataset.yaml`](configs/create_yolo_detector_dataset.yaml)
and
[`configs/create_yolo_classifier_dataset.yaml`](configs/create_yolo_classifier_dataset.yaml).
Under `data`, the configuration key is the exact creator class name:
`YoloDetectorDatasetCreatorFromSpeciesnet` or
`YoloClassifierDatasetCreatorFromSpeciesnet`. Input/output paths are interpreted
relative to the working directory, not the YAML file's directory.

`path_to_image_data` and `dataset_output_path` are required. Optional `class_names`
selects species directories; omit it or use `null` to discover all species.
Optional `model_name` selects SpeciesNet weights; omit it or use `null` for the
default model. Injected model objects are supported through the Python constructor,
not YAML. Loading a configuration reads metadata but does not load model weights
or create output files.

```python
from smartrodent.yolo_dataset_creation import YoloDetectorDatasetCreatorFromSpeciesnet

creator = YoloDetectorDatasetCreatorFromSpeciesnet.from_config(
    "configs/create_yolo_detector_dataset.yaml"
)
# Explicitly writes the configured dataset; only run after checking its paths.
creator.create()
```

Alternatively, run the shared script with a **positional configuration path**.
It selects the creator from the class-named key; exactly one creator is allowed.
Check the configured paths before running: these commands write dataset outputs.

```bash
uv run python scripts/create_yolo_dataset.py configs/create_yolo_detector_dataset.yaml
uv run python scripts/create_yolo_dataset.py configs/create_yolo_classifier_dataset.yaml
```

Use `YoloClassifierDatasetCreatorFromSpeciesnet` with the classifier configuration
for crop-based classification output. The example destinations are
`datasets/stage8_yolo_detector` and `datasets/stage8_yolo_classifier`; stage numbering
is a convention, not an enforced restriction. Creation preserves existing splits
and oversampled rows and requires a new destination directory.

## YOLO classification: class-index ordering

`YoloClassificationTrainer` passes the classification dataset directory to
Ultralytics, which derives class indices from **alphabetically sorted species
folder names under `train/`**. It does not use the `names` ordering in `data.yaml`.
Even when a YAML file is passed to the trainer, only its `path` is used to locate
the classification dataset.

An explicit dataset-creator `class_names` order can therefore differ from the
trained classifier's index order. **Use the trained model's `names` mapping to
interpret prediction indices**, rather than the creator's class list or YAML.
This differs from detection training, which uses the class-index mapping in
`data.yaml`.

## Standalone SupCon experiment

The independent two-view experiment runs outside the notebook and uses only the
supervised contrastive objective. Run from the repository root in the installed
project environment:

```bash
python scripts/train_supcon.py configs/train_supcon_config.yaml
```

The YAML config specifies the source species, augmentation size, P-K batch,
workers, optimizer settings, and output directory. A batch contains
`classes_per_batch * samples_per_class` source images and twice as many augmented
views. `samples_per_epoch` counts source-image selections and is rounded down to
complete batches. The default experiment uses FP32, without AMP or validation.

Each epoch saves `epoch_0001.pt`, etc., along with `config.yaml` and `history.yaml`.
Checkpoints contain architecture, encoder/classifier/projector weights, class
mapping, optimizer state, and experiment settings. Evaluation can reload a model
without the original pretrained weights or the original Python session:

```python
from smartrodent.supcon_model import YoloSupConModel

model = YoloSupConModel.from_checkpoint(
    "runs/supcon_standalone_xl/epoch_0005.pt", device="cpu"
)
```

The plotting and evaluation workflow remains in `notebooks/supcon_training.ipynb`.
Its helpers are imported from `smartrodent.supcon_datasets`,
`smartrodent.supcon_model`, and `smartrodent.eval`. The notebook compares the same
balanced image subset in pretrained pooled, trained pooled, and trained projected
spaces. Set its checkpoint path to the experiment you want to inspect.

This script is separate from the older Ultralytics trainer integration in
`scripts/train_yolo_classifier_supcon.py` and `smartrodent.yolo_supcon`.

### Model cards

Standalone SupCon training fills `configs/model_card_template.yaml` with the
selected best checkpoint's model name, registry version, class mapping, and
validation results. Set `model_card_template` in the training config to use a
different template. Editable narrative fields are preserved.

The `provenance` section records the checkpoint filename, SHA-256, epoch,
MLflow run and model IDs, Git commit, and dataset revision. Unavailable IDs or
revisions remain `null`. Provenance is included in both YAML and Markdown.

Both `epoch_0007.model_card.yaml` and `epoch_0007.model_card.md` are saved beside
`epoch_0007.pt`. With MLflow registration enabled, both files are uploaded to the
logged model and the Markdown becomes the registered version's description.
Without registration, the version is unset; cards are saved locally and uploaded
to the run when tracking is enabled.

The generic helpers in `smartrodent.model_card` also work independently:

```python
from pathlib import Path
from smartrodent.model_card import load_card_template, render_model_card

card = load_card_template(Path("configs/model_card_template.yaml"))
card["model"]["name"] = "rodent-species"
markdown = render_model_card(card)
```

## Tests

Install the development extra, then run the test suite from the repository
root:

```bash
uv sync --extra dev
uv run pytest
```

Tests that require external services or special hardware are identified with
pytest markers:

- `network` makes live network requests. Run only these tests with
  `uv run pytest -m network`, or exclude them with
  `uv run pytest -m "not network"`.
- `vlm_integration` runs real Ollama and vLLM inference. These tests are
  opt-in and require both the marker selection and an environment variable:

  ```bash
  SMARTRODENT_RUN_VLM_INTEGRATION=1 uv run pytest -m vlm_integration
  ```

  Add `-k ollama` or `-k vllm` to run only one backend. The model and runtime
  defaults can be overridden with `SMARTRODENT_OLLAMA_MODEL`,
  `SMARTRODENT_VLLM_MODEL`, `SMARTRODENT_VLLM_GPU_MEMORY_UTILIZATION`,
  `SMARTRODENT_VLLM_MAX_MODEL_LEN`, and `SMARTRODENT_VLLM_MAX_NEW_TOKENS`.
  Custom test images can be selected with
  `SMARTRODENT_INTEGRATION_RODENT_IMAGE` and
  `SMARTRODENT_INTEGRATION_EMPTY_IMAGE`.

The test suite does not run the complete iNaturalist download pipeline. Use
`dvc repro` explicitly when you want to execute that pipeline and write a
dataset.
