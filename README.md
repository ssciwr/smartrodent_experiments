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

## Inference script

The `inference` command runs SpeciesNet detection and then classifies up to the
five highest-confidence animal crops with the YOLO classifier. Configure the
models in [`configs/inference_pipeline.yaml`](configs/inference_pipeline.yaml):

```yaml
speciesnet_model: kaggle:google/speciesnet/pyTorch/v4.0.3a/1
classifier_weights: path/to/classifier/weights.pt
path: /path/to/images # one image or a non-recursive image directory
output: /path/to/output
imgs: [.jpg, .jpeg, .png, .webdav]
```

`classifier_weights` is a filename relative to the default Hugging Face
repository, `MaHaWo/Yolo26Rodent`. The classifier is downloaded and cached by
`huggingface_hub` on first use.

Run inference with a configuration file only:

```bash
uv run inference --config configs/inference_pipeline.yaml
```

`path` may name one image or a directory; directory scanning is not recursive.
The command creates `<output>/results.json`, keyed first by image filename and
then by detection index. Each detection contains the normalized `bbox`, all class
`probabilities`, the top result and confidence, and the top-five labels and
confidences. Images without detections produce an empty mapping.

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

## Tests

Install the development extra, then run the test suite from the repository
root:

```bash
uv sync --extra dev
uv run pytest
```

The test command does not run the iNaturalist download pipeline. Use `dvc repro`
explicitly when you want to make network requests and write a dataset.
