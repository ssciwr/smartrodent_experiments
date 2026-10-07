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

## DVC usage

The `download_inaturalist` DVC stage runs the downloader with
`config/dataset_full.yaml`. The `inaturalist` configuration mapping is tracked
as DVC parameters, while the downloader script and implementation are tracked
as code dependencies.

Check whether the stage is up to date:

```bash
uv run dvc status
```

Run or reproduce the configured download:

```bash
uv run dvc repro download_inaturalist
```

The stage deliberately has no declared DVC outputs. This lets `output_path`
refer to an external or large local dataset without DVC attempting to cache or
version the images. Edit the YAML to change the destination or dataset
parameters, then rerun the stage.

To run the downloader without DVC:

```bash
uv run python scripts/download_inaturalist_data.py config/dataset_full.yaml
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
