# Rodent Experiments

Experimental code for building and evaluating image datasets and computer-vision models for rodent and rodent-adjacent wildlife detection. The repository currently focuses on:

- filtering BioTrove metadata into regional/taxonomic datasets,
- downloading and organizing species-labeled image data,
- using CLIP to remove non-animal, dead-animal, specimen, or otherwise unsuitable images,
- running baseline object detection with YOLO,
- running species classification/detection with Google SpeciesNet,
- saving boxed preview images and detection crops for manual inspection.

The work is exploratory. The notes in `notes/notes.md` document candidate datasets, target species for Central Europe and Sri Lanka, and open modeling questions around thermal/RGB fusion, uncertainty, and deployment.

## Repository layout

```text
configs/                    BioTrove filtering/download configuration files
notebooks/process_biotrove.ipynb
                            notebook workflow for BioTrove processing and CLIP filtering
src/smartrodent/main.py     YOLO and SpeciesNet experiment entry points
src/smartrodent/utils.py    image-path, country-code, visualization, and crop helpers
requirements_processing.txt extra processing/notebook dependencies, including BioTrove and CLIP
datasets/                   local datasets; large data is expected to live here locally
```

## What it depends on

The Python package is configured in `pyproject.toml` and requires Python **3.13 or newer**. Main runtime dependencies are:

- `datasets` / `hf-datasets` for Hugging Face dataset access,
- `polars`, `numpy`, `pyaml` for metadata processing,
- `matplotlib` and `Pillow` for plotting/preview image output,
- `ultralytics` for YOLO inference,
- `speciesnet` for wildlife detection/classification and geofenced species predictions.

The BioTrove processing notebook also uses the packages listed in `requirements_processing.txt`, notably:

- `git+https://github.com/baskargroup/BioTrove`,
- `git+https://github.com/openai/CLIP.git`,
- `torch`, `torchvision`, `pandas`, `pyarrow`, `fastparquet`, `seaborn`, `scikit-image`, `ipykernel`, and `tqdm`.

Development tools are available through the optional `dev` extra: `pytest`, `pytest-cov`, `coverage`, `pre-commit`, and `pytest-mock`.

## Installation

This repository has a `uv.lock`, so `uv` is the preferred installer.

```bash
# from the repository root
uv sync
```

For development tools as well:

```bash
uv sync --extra dev
uv run pre-commit install
```

If you need the notebook/BioTrove processing environment, install the additional processing requirements into the environment:

```bash
uv pip install -r requirements_processing.txt
```

If you are not using `uv`, create a Python 3.13+ virtual environment and install the project with pip:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements_processing.txt   # optional: BioTrove notebook processing
pip install -e '.[dev]'                       # optional: development tools
```

## Basic usage

Run YOLO on one image or a list of images from Python:

```python
from smartrodent.main import main

main("path/to/image.jpg", batchsize=1, project="runs/yolo", crop=True)
```

Run SpeciesNet with optional country geofencing and save JSON predictions, boxed previews, and crops:

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

The BioTrove workflow is currently notebook-driven. Open `notebooks/process_biotrove.ipynb`, choose one of the configuration files in `configs/`, and run the cells to filter metadata, download images, generate image/text pairs, organize classifier folders, and experiment with CLIP-based filtering.

## Data and model files

Large datasets and generated outputs are not expected to be committed. The `configs/*.json` files contain local paths under `datasets/` and may need to be adjusted for your machine. Model outputs are written under `runs/` by default.
