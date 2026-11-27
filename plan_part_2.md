# Part 2b–2e implementation plan

**Status:** Proposed, pending approval of the plan and test list. Implementation and baseline testing have yet to begin.

Scope is [Part 2b–2e of the refactoring plan](REFACTORING_PLAN.md#part-2-lightning-custom-encoders-standalone-ultralytics-trainer). Follow the test-first workflow in [AGENTS.md](AGENTS.md#workflow).

## Implementation plan

Build and review 2b+2c first, then switch the training entry points together in 2d+2e. The current classifier loads SupCon through the old `.pt` reader. Switching SupCon to `.ckpt` first would break that handoff or require a temporary converter. Move removal of the old SupCon loop from 2c to the 2d+2e cutover.

1. **2b · Shared training pieces.** Reuse the existing `ComponentSpec` and Lightning dependency. Define the plain PyTorch `Encoder` interface, implement the YOLO encoder without hooks, and share parameter-prefix matching between freezing and reinitialization. Use `ComponentSpec` for optimizers, schedulers, augmentations, and metrics. Give `Objective` ownership of metric direction and `RunArtifacts` ownership of the paths used by writers and the input guard.
2. **2c · SupCon.** Add `SupConTask` and a small `stage_runner.py` using Lightning's trainer, checkpointing, early stopping, and loggers. Keep the existing samplers and deterministic validation cohort. Store loss min/mean/std/max, use the mean to select the checkpoint and report progress, and restore the encoder from saved hyperparameters and weights. Exercise this through the new runner while the existing entry points still serve the current pipeline.
3. **2d · Classifier.** Add `ClassifierTask` using the same runner. Train an encoder plus a linear head, compute CE loss across the whole split, and preserve the existing sklearn metric identity in MLflow. Accept a configured encoder or restore one from a `.ckpt`. For YOLO configs that request native export, write a loadable `best.pt` beside `best.ckpt` and log it with the registered model.
4. **2e · Pipeline cutover.** Preflight freeze and reinitialization prefixes against all candidate encoders, then pass the winning SupCon `best.ckpt` directly to CE. Replace the study subclasses with one `StageStudy`, preserving per-encoder study identity and champion promotion. Use one Torch ONNX exporter for both stages, with the Ultralytics-selected opset recorded in metadata. Call the encoder directly in representation evaluation. Update the affected scripts, configs, README, and notebook, then remove the superseded loops, adapters, and tests.

## Config details for approval

- **Scheduler cadence.** Default to epoch stepping and keep multi-phase schedules in epochs. Allow explicit `interval: step` for a single scheduler so batch-based schedulers such as `OneCycleLR` can work as intended.
- **Pretrained reinitialization.** Restore matched encoder parameters from the weights that initialized the encoder before fine-tuning. Reject the setting during preflight if those weights are unavailable. For random reinitialization, reset the matched modules and verify that parameters outside the requested prefixes stay intact.

## Test list for approval

### 2b+2c

- Configured encoders accept RGB tensors in `[0, 1]` and return `(B, feature_dim)`.
- Training transforms apply configured augmentations in order, and validation uses deterministic preprocessing.
- **Hypothesis.** Prefix matching respects dotted name boundaries, so `model.1` cannot match `model.10`.
- Unmatched freeze or reinitialization prefixes fail during preflight.
- Frozen parameters stay fixed, and frozen BatchNorm statistics follow `eval` or `adapt`.
- An empty scheduler list keeps the learning rate constant.
- Chained schedulers switch on the configured epoch, and a single step-based scheduler advances per optimizer step.
- Plateau scheduling uses the objective's direction and rejects placement in a chain.
- A tiny CPU SupCon run preserves sampler behavior and records all four loss statistics.
- The mean loss drives checkpoint selection and early stopping, with the earliest epoch winning ties.
- The best checkpoint restores identical encoder features without the original weights file.
- Nonfinite loss raises `FloatingPointError` before publication.
- Optuna receives one observation per completed epoch, and pruning propagates.
- **Hypothesis, existing.** Keep the `ComponentSpec` argument-collision test.

### 2d+2e

- A tiny CPU classifier trains and restores from its best checkpoint.
- CE loss matches the whole-split mean with an uneven final batch.
- Random and pretrained reinitialization alter only the requested parameters.
- A native export request fails during preflight when the encoder lacks that capability.
- Pipeline preflight checks every candidate encoder, and CE receives the winning SupCon checkpoint.
- Study resume preserves encoder identity, and a failed study cannot promote a partial winner.
- Existing MLflow champions remain comparable through the unchanged metric identity and validation metadata.
- ONNX Runtime matches Torch encoder features and classifier probabilities, with the selected opset recorded.
- Representation evaluation produces ordered features and restores the encoder's training mode.
- Artifact paths reject dataset or source collisions, including symlink and hard-link aliases.
- **Opt-in, real YOLO.** The registered native `best.pt` loads through `YOLO(path)` and matches the Lightning classifier's probabilities.

## Resuming the work

Approve the plan and test list together, including the shared cutover and the two config details. After approval, record the baseline for each group, delegate the red and green steps, and have an independent reviewer check the code and tests using both coverage commands from `AGENTS.md`.
