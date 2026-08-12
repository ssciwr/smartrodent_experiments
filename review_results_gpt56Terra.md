# Review: PR #16 — Add filter pipeline

Reviewed `main...add-filter-pipeline` for PR #16. The PR description is the
feature contract: filter downloaded data through live-animal, thermal, flying
bird, and snake stages, with Ollama/vLLM backends and tests.

## High

### 1. Checked-in filter paths resolve to a different dataset directory than DVC

**Locations:** `configs/filter_data_vlm_config_{animal,thermal,snakes,birds}.yaml:4`, `src/smartrodent/utils.py:15-17`, `dvc.yaml:15,28,35,42`

The configs use `smartrodent_experiments/datasets/...`, but relative paths are
resolved below the repository root. At `<repo>`, that yields
`<repo>/smartrodent_experiments/datasets/...`; DVC uses `<repo>/datasets/...`.
The initial stage therefore fails on a fresh checkout, or processes an
unrelated directory.

Use an explicit data-root/base-path setting (rather than relying on the checkout
location) and resolve all configured dataset paths from it. A small unit test
can check the committed configs' resolved paths without creating image data.

### 2. The DVC graph does not safely model the intended partial parallelism

**Locations:** `dvc.yaml:10-42`, `configs/filter_data_vlm_config_{thermal,snakes,birds}.yaml:17-18`, `src/smartrodent/filter.py:143-160,176-187`

The snake and bird configs select disjoint species directories, so those two
filters are correctly independent and should be able to run in parallel. The
problem is that the thermal config has no species restriction and moves files
out of the entire shared `filtered_kept` tree while both specialised filters may
be reading their subsets. All three also derive shared destination roots and
overwrite one `filter_results.csv`. DVC sees only the whole `filtered_kept`
directory as each stage's dependency and none of their writes as outputs, so it
cannot express or protect the intended subset-level concurrency. `dvc dag`
confirms the three sibling stages.

Do **not** copy these large image trees at each stage. Make thermal an explicit
predecessor of the two specialised stages, then keep snake and bird parallel by
declaring their disjoint species subtrees/decision manifests as their inputs
and giving each a distinct results file. More robustly, let each model append
decisions to a per-stage manifest keyed by original image path, and perform one
final ordered materialization that moves each image once. Test the thermal →
{snake, bird} graph, concurrent specialised runs, and final file accounting.

### 3. DVC does not track the implementation whose changes affect filters

**Locations:** `dvc.yaml:10-42`

Stages track the CLI script but not `src/smartrodent/filter.py` or
`src/smartrodent/utils.py`, where classifier, parser, path, and placement logic
lives. Changing this code can leave DVC reporting stages up to date and retain
stale data.

Add the implementation modules/package as dependencies and test `dvc status`
after a filtering-code change.

## Medium

### 4. Model responses are not validated against the documented exact schema

**Locations:** `src/smartrodent/filter.py:211-255`, `configs/filter_data_vlm_config_animal.yaml:29-49` and equivalent configs

Prompts demand an exact schema, but `parse_response()` only catches JSON syntax
errors and checks `label`. A valid JSON list/string throws at `data.get`; missing
fields, invalid enums, wrong types, and extra fields pass through. In
particular, `bool("false")` is `True`, corrupting the review field. Ollama asks
only for generic JSON, so it has no backend schema safeguard.

Validate that input is a mapping and validate required fields, types, enums,
and extras. Invalid data should become a documented failure result. Test scalar
JSON, missing fields, invalid boolean/quality values, and extras.

### 5. vLLM batch cardinality is assumed rather than checked

**Locations:** `src/smartrodent/filter.py:526-541,592-601`

vLLM is expected to return one result per submitted conversation, so a short
batch should not occur in normal operation. Nevertheless, the local code does
not verify that invariant: `zip(chunk, self._classify_batch(chunk))` truncates
to the shorter sequence while the progress bar advances by the whole chunk.
If a backend regression, wrapper change, or mocked/test backend violates the
assumption, images are silently omitted from both placement and results.

Check that the returned count equals the submitted count and raise a clear
error if it does not. Add a unit test with a deliberately short mocked batch.
This is a defensive Medium finding, not a claim that current vLLM normally
returns short batches.

### 6. README promises a lockfile install although the project intentionally has no lockfile

**Locations:** `README.md:24-27`, `.gitignore:30`, deleted `uv.lock`

The branch deliberately treats `pyproject.toml` constraints as the dependency
source of truth and leaves versions open; that is a valid project policy and
the absence of `uv.lock` is not itself a defect. However, the README still says
runtime dependencies are installed "from the lockfile." That is now an
incorrect installation contract.

Update the README to state that dependencies are resolved from
`pyproject.toml`, and, if useful, briefly document the intentional trade-off:
current compatible dependencies over an exact frozen environment.

### 7. Per-image inference failures abort a run despite the failure-bucket contract

**Locations:** `src/smartrodent/filter.py:43-46,374-398,575-605`

`failure_root` is documented as the destination for failed classifications, but
a read, backend, model-output, or placement failure propagates and stops the
stage without a row identifying the image. Current testing verifies only
abort-and-cleanup (`tests/test_vlm_filter.py:402-418`).

Handle expected per-image failures by recording a failure decision and
continuing, or document a deliberately fail-fast contract. Add failure tests
for both backends and an end-of-run error summary.

### 8. The download stage is not a producer in the new DVC graph

**Locations:** `dvc.yaml:2-21`, `README.md:83-86`

The initial filter consumes `datasets/full_dataset/raw`, but the download stage
declares no output, so DVC cannot reproduce a fresh download followed by the
filters. The existing external-output approach is reasonable, but is
incompatible with presenting this as one reproducible pipeline.

Either declare a suitable non-cached external output and connect it, or
document download and filtering as separate commands.

## Low

### 9. CLI docs promise multi-config chaining that is not implemented

**Locations:** `scripts/filter_data_vlm.py:21-37`

`--config` accepts one `Path` and constructs one filter. Its help says "Path or
list of paths", and comments say configs can be chained. Correct the docs or
implement the promised behavior.

### 10. `copy_with_structure` has an incorrect contract

**Locations:** `src/smartrodent/filter.py:189-208`

The method says it copies an image, but removes the source in `move` mode. Name
or document it as copy-or-move/placement. `except Exception as e: raise e` is
also redundant and should be removed.

## Verification and test coverage

- `python -m pytest tests/test_vlm_filter.py tests/test_utils.py -q`: **33 passed**.
- Full collection found 55 tests. The two VLM integration tests require
  external model/backend infrastructure and are intentionally skipped by CI.
- `dvc dag` confirmed all three follow-up filters are siblings. The intended
  safe graph is thermal first, then parallel disjoint snake and bird filters.
- The useful missing coverage does **not** require a large dataset or real
  inference: use tiny temporary directories and mocks to test committed-config
  resolution, the expected DVC stage relationships, shared result-file names,
  parser schema validation, batch-cardinality checking, and per-image error
  recording.
