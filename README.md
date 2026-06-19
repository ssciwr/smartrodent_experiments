# Rodent Experiments

Experimental code for building rodent and rodent-adjacent wildlife image datasets and trying baseline computer-vision models for detection, cropping, and species classification.

The repository is currently exploratory. It combines:

- BioTrove metadata filtering for region-specific target species,
- image download and image/text-pair generation from filtered BioTrove metadata,
- CLIP-based visual filtering of unsuitable images,
- YOLO object-detection experiments,
- Google SpeciesNet species prediction experiments,
- helper code for saving boxed preview images and object crops for manual inspection.

See `notes/notes.md` for dataset notes, target-species rationale, references, and open modeling/deployment questions.

## Repository layout

```text
configs/                                  BioTrove workflow configs and target species lists
notebooks/process_biotrove.ipynb          notebook workflow for BioTrove processing and CLIP filtering
src/smartrodent/biotrove_process/         local modified BioTrove processing package
src/smartrodent/dataprocessing.py         CLIP-based image filtering and plotting helpers
src/smartrodent/main.py                   YOLO and SpeciesNet experiment entry points
src/smartrodent/utils.py                  image-path, country-code, crop, and visualization helpers
src/smartrodent/yolo26*.pt                local YOLO weights used by experiments
requirements_processing.txt               notebook/processing dependencies, including OpenAI CLIP
datasets/                                 local data directory; large datasets are not meant for git
notes/notes.md                            dataset notes, species lists, references, open questions
```

## What it does

### BioTrove dataset processing

The notebook `notebooks/process_biotrove.ipynb` uses the local package `smartrodent.biotrove_process` to:

1. load a JSON config from `configs/`,
2. filter BioTrove parquet metadata by taxonomic category or scientific name,
3. compute per-species/category counts,
4. shuffle and chunk filtered metadata,
5. download source images,
6. generate image/text pairs and optional tar files,
7. organize classifier-style image folders by scientific name,
8. run CLIP prompt-based filtering to separate likely useful live-animal images from non-animal/specimen/dead-animal images.

The BioTrove processing code under `src/smartrodent/biotrove_process/` is a local modified copy of the original BioTrove processing utilities. The notebook imports it through `smartrodent.biotrove_process`, not from an external checkout.

### Target species configs

The currently maintained region configs are:

- `configs/config_central_europe.json`
  - `Rattus norvegicus`, `Rattus rattus`, `Mus musculus`, `Myodes glareolus`, `Apodemus agrarius`, `Apodemus flavicollis`, `Apodemus sylvaticus`, `Microtus arvalis`, `Microtus agrestis`, `Arvicola amphibius`, `Crocidura leucodon`
- `configs/config_srilanka.json`
  - `Rattus norvegicus`, `Rattus rattus`, `Suncus murinus`, `Bandicota indica`, `Bandicota bengalensis`, `Mus booduga`, `Vandeleuria`, `Mus musculus`

The config paths are currently machine-local absolute paths. Before running the notebook on another machine, update `source_folder`, output folders, and intermediate directories in the selected config.

### Model experiments

`src/smartrodent/main.py` exposes two main experiment functions:

- `main(...)`: run local YOLO weights on one image or a batch of images. Ultralytics saves boxed previews and optional crops under `runs/`.
- `run_speciesnet(...)`: run SpeciesNet with optional country geofencing, write prediction JSON, save boxed previews, and optionally save detector crops.

`src/smartrodent/utils.py` contains shared helpers for image path expansion, country-code normalization, SpeciesNet label formatting, preview rendering, and crop extraction.

`src/smartrodent/dataprocessing.py` contains the `ImageFilter` class used by the BioTrove notebook for CLIP-based prompt filtering and visualization.

## Dependencies

The package is configured in `pyproject.toml` and requires Python **3.13 or newer**.

Main package dependencies:

- `datasets` / `hf-datasets`
- `polars`, `numpy`, `pyaml`
- `matplotlib`, `Pillow`
- `torch`, `torchvision`
- `ultralytics`
- `speciesnet`

Additional notebook/processing dependencies are listed in `requirements_processing.txt`. Important ones include:

- `git+https://github.com/openai/CLIP.git`
- `aiohttp`
- `pandas`, `pyarrow`, `fastparquet`, `polars`
- `ipykernel`, `ipython`
- `seaborn`, `scikit-image`, `tqdm`

Development dependencies are available through the `dev` extra:

- `pytest`
- `pytest-cov`
- `coverage`
- `pre-commit`
- `pytest-mock`

## Data prerequisites

Download BioTrove before running the BioTrove notebook:

- <https://huggingface.co/datasets/BGLab/BioTrove>

Place or symlink the dataset under `datasets/BioTrove/BioTrove`, or update the `source_folder` value in the relevant config file to point to the local BioTrove parquet directory.

Large datasets, generated parquet chunks, downloaded images, model outputs, and `runs/` outputs are local working data and should not be committed.

## Installation

This repository has a `uv.lock`, so `uv` is the preferred installer.

```bash
# from the repository root
uv sync
```

For development tools:

```bash
uv sync --extra dev
uv run pre-commit install
```

For the notebook/BioTrove processing environment, install the additional processing requirements into the same environment:

```bash
uv pip install -r requirements_processing.txt
```

Without `uv`, use a Python 3.13+ virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements_processing.txt   # optional: notebook processing
python -m pip install -e '.[dev]'                       # optional: development tools
```

## Basic usage

### Run YOLO

```python
from smartrodent.main import main

main(
    "path/to/image.jpg",
    batchsize=1,
    project="runs/yolo",
    crop=True,
)
```

### Run SpeciesNet

```python
from smartrodent.main import run_speciesnet

predictions = run_speciesnet(
    "path/to/images",
    output_json="runs/speciesnet/predictions.json",
    preview_dir="runs/speciesnet/boxed",
    crop_dir="runs/speciesnet/crops",
    batch_size=16,
    country="LKA",  # examples: DEU, LKA, USA
)
```

### Use the local BioTrove processing package

```python
from smartrodent.biotrove_process import (
    MetadataProcessor,
    GenShuffledChunks,
    GetImages,
    GenImgTxtPair,
    load_config,
)

config = load_config("configs/config_central_europe.json")
processor = MetadataProcessor(**config["metadata_processor_info"])
processor.process_all_files()
```

For the full workflow, open `notebooks/process_biotrove.ipynb`, choose one of the configs in `configs/`, verify all paths, and run the notebook cells in order.

### Use CLIP image filtering helpers

```python
from smartrodent.dataprocessing import ImageFilter

image_filter = ImageFilter(
    model="RN50x16",
    prompts=["not an animal at all", "a mouse, rat or other rodent"],
    id_tol=0.02,
)
```

## Tests and development status

The project has pytest/coverage configuration in `pyproject.toml`, but the current repository is primarily notebook- and experiment-driven. Some generated caches and local data may exist in working trees; avoid committing large data, generated outputs, or `__pycache__` files.

Run tests, when present and applicable, with:

```bash
uv run pytest
```

or, without `uv`:

```bash
python -m pytest
```

## Notes and caveats

- The config files currently contain absolute local paths and must be edited for other machines.
- SpeciesNet and YOLO may download or load large model files and may require substantial compute.
- CLIP filtering uses PyTorch and is much faster with CUDA, but the helper code currently assumes CUDA in `compute_similarity`.
- Species lists in `notes/notes.md` are dataset targets, not complete regional fauna lists.
- This repository is exploratory and should be treated as research code rather than a stable application API.
