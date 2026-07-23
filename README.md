# SmartRodent experiments

Experimental machine-learning code for the SENTINEL-RAT/SmartRodent project.
The repository currently focuses on running and comparing image detectors for
rodents and other small wildlife, normalizing their predictions, and recording
ideas and results for future model development.

This is research code, not a finished application or stable Python API.

## Current status

The current implementation provides:

- a shared `DetectorBase` interface and normalized `detections.json` output;
- an Ultralytics YOLO detector wrapper;
- an Ultralytics YOLOE open-vocabulary detector wrapper with text prompts;
- batching that bounds the number of images passed to Ultralytics at once;
- a `DetectionExperiment` helper for running one detector configuration over
  folder-based image groups;
- YAML experiment configuration with anchors and aliases for reusable class
  lists;
- an AMMICO notebook and script for experimental VLM inference through an
  OpenAI-compatible API;
- unit tests for detector configuration, batching, output normalization, and
  experiment orchestration.

There is currently **no command-line interface or complete end-to-end pipeline**.
The detector classes are the main maintained code. Dataset preparation,
fine-tuning, evaluation aggregation, and deployment integration remain
experimental or documented as notes.

## Repository layout

```text
.
├── configs/
│   └── detector_experiments.yaml           detector experiment matrix and settings
├── notebooks/
│   ├── ammico_demo_getting_started.ipynb  AMMICO image-summary/VQA walkthrough
│   └── yolo26n.pt                          local model artifact, ignored by git
├── notes/
│   ├── notes.md                            datasets, model findings, architecture ideas
│   ├── species.md                          target species and references
│   ├── yolo26_dataset_preparation.md       notes for preparing YOLO fine-tuning data
│   ├── smartrodent architecture.drawio     draft system architecture
│   └── overview_faunanet*.html             saved architecture/reference material
├── scripts/
│   └── species_script.py                   experimental batch AMMICO VQA script
├── src/smartrodent/
│   ├── __init__.py
│   ├── base.py                             detector interface and shared helpers
│   └── detection.py                        YOLO, YOLOE, and experiment wrappers
├── tests/
│   ├── conftest.py
│   ├── test_detectorbase.py
│   ├── test_detectorexperiment.py
│   ├── test_yolodetector.py
│   └── test_yoloedetector.py
├── pyproject.toml                          project and dependency metadata
├── uv.lock                                reproducible Python 3.14 dependency lock
└── requirements_processing.txt            legacy processing/notebook requirements
```

Large datasets, generated runs, virtual environments, and model weights are
excluded through `.gitignore` and should not be committed.

## Requirements and installation

The main environment requires Python **3.14.x**. The project is locked with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Install the development and test tools with:

```bash
uv sync --extra dev
uv run pre-commit install
```

The repository currently uses a `src/` layout but does not define a packaging
build backend, so `smartrodent` is not installed into the environment by
`uv sync`. When running a script or interactive session directly from this
checkout, expose `src/` explicitly:

```bash
PYTHONPATH=src uv run python your_script.py
```

The test configuration already adds `src/` to `sys.path`, so no additional
setting is needed for `pytest`.

`requirements_processing.txt` is retained for older data-processing and
notebook work. It is not part of the current locked environment and should not
be installed into the main `.venv` without reviewing its packages first.

## Detector API

### YOLO

```python
from pathlib import Path

from smartrodent.detection import YOLO_Detector

images = sorted(Path("datasets/example").glob("*.jpg"))
out = Path("runs/yolo/example")

detector = YOLO_Detector(
    name="boxed",
    crop=True,
    batchsize=8,
    img_outname="boxed",  # retained for backend interface compatibility
    conf=0.25,
    model_name="/path/to/yolo-model.pt",
    project=str(out),
)
results = detector.detect(images, out)
```

### YOLOE with text-prompt classes

```python
from pathlib import Path

from smartrodent.detection import YOLOE_Detector

images = sorted(Path("datasets/example").glob("*.jpg"))
out = Path("runs/yoloe/example")

detector = YOLOE_Detector(
    name="boxed",
    crop=True,
    batchsize=8,
    img_outname="boxed",
    conf=0.20,
    model_name="/path/to/yoloe-model.pt",
    classes=["animal", "rat", "mouse", "shrew"],
    project=str(out),
)
results = detector.detect(images, out)
```

Both wrappers accept a single image path, a directory supported by
Ultralytics, or a list of image paths. List inputs are chunked according to
`batchsize` to avoid sending an unbounded in-memory batch to the model.

Each run writes a normalized summary to `<out>/detections.json`:

```json
{
  "example.jpg": [
    {"class": "mouse", "conf": 0.873},
    {"class": "animal", "conf": 0.641}
  ],
  "empty.jpg": []
}
```

Ultralytics-native previews and crops are written under the configured
`project` and run name.

## Grouped experiments

`DetectionExperiment` builds a fresh detector for each immediate subdirectory
of a dataset. Its output path is:

```text
<run_dir>/detect<confidence>/<experiment_name>/<dataset_name>/<group_name>/
```

Example construction:

```python
from pathlib import Path

from smartrodent.detection import DetectionExperiment

experiment = DetectionExperiment(
    run_dir="runs",
    dataset_name="biotrove-small-mammals",
    experiment_name="yoloe-prompts",
    conf=0.20,
    detector_type="YOLOE_Detector",
    detector_kwargs={
        "batchsize": 8,
        "crop": True,
        "model_name": "/path/to/yoloe-model.pt",
        "classes": ["animal", "rat", "mouse", "shrew"],
    },
)

for group_name, images in experiment.image_groups(Path("datasets/grouped")):
    experiment.run_detector_on_group(group_name, images)
```

The repository includes `configs/detector_experiments.yaml` for shared dataset
settings, confidence thresholds, and detector definitions. Load it without
constructing an experiment first:

```python
config = DetectionExperiment.load_experiment_config(
    "configs/detector_experiments.yaml"
)
```

YAML anchors replace the previous custom JSON lookup syntax. A reusable class
set can be declared and referenced directly:

```yaml
class_sets:
  small_mammals: &small_mammals
    - rat
    - mouse
    - shrew

experiments:
  - name: prompted
    detector: YOLOE_Detector
    kwargs:
      classes: *small_mammals
```

A full config-driven runner has not yet been implemented.

## AMMICO/VLM experiment

The AMMICO material is currently isolated from the main environment. Upstream
AMMICO pins `torch<2.9`, while the Python 3.14 environment uses a newer PyTorch,
so adding AMMICO to the main lock would downgrade PyTorch to a version that does
not support Python 3.14.

If the AMMICO notebook or `scripts/species_script.py` is needed, create a
separate Python 3.13 environment:

```bash
uv venv .venv-ammico --python 3.13
uv pip install \
  --python .venv-ammico/bin/python \
  'ammico[api] @ git+https://github.com/ssciwr/AMMICO.git'
```

AMMICO's full dependency set may also impose platform/GPU restrictions. The
notebook expects an OpenAI-compatible endpoint served by software such as
Ollama or vLLM. Configure it through `AMMICO_API_BASE_URL`, `AMMICO_API_KEY`,
and `AMMICO_MODEL_ID`.

The paths, questions, endpoint, and API key placeholders in
`scripts/species_script.py` must be adapted before use.

## Tests

Run the test suite with:

```bash
uv sync --extra dev
uv run pytest
```

The current suite contains 23 tests and covers the shared detector helpers,
YOLO/YOLOE wrappers using mocked model calls, and `DetectionExperiment`.
It does not download model weights or run real inference.

Coverage configuration lives in `pyproject.toml` and currently requires at
least 80% coverage:

```bash
uv run pytest --cov
```

## Research findings and direction

The current notes record the following provisional conclusions:

- SpeciesNet detects larger camera-trap species well but performed poorly on
  the small rodents and shrews considered here.
- YOLOE is useful for open-vocabulary animal/rodent detection, although its
  species-level performance is limited.
- Generic YOLO detection likely needs project-specific fine-tuning.
- A two-stage pipeline—animal detection followed by specialized
  classification—is a plausible direction.
- The available thermal sensor resolution is likely useful as a detection
  prior but insufficient for reliable species identification by itself.
- Confidence calibration and evidence aggregation across multiple images
  remain open research problems.

These are exploratory observations rather than benchmark-quality claims. See
`notes/notes.md` for the complete discussion and caveats.

## Known limitations

- No CLI or complete config-driven execution entry point exists yet.
- The `src/` package is not installed automatically; use `PYTHONPATH=src`.
- Real model inference requires external or downloaded weights and suitable
  compute resources.
- AMMICO must remain in a separate environment until its PyTorch constraints
  are compatible with Python 3.14.
- `scripts/species_script.py` contains machine-specific paths and placeholder
  endpoint credentials.
- Test coverage is focused on orchestration and normalized output, not model
  quality or integration inference.
- The repository remains exploratory and APIs may change without notice.

## License

See [LICENSE](LICENSE).
