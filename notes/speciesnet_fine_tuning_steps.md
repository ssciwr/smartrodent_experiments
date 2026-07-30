# SpeciesNet fine-tuning workflow

Stepwise summary of the SpeciesNet fine-tuning tutorial: <https://agentmorris.github.io/speciesnet-fine-tuning/>

## 1. Decide whether fine-tuning is worth it

- Fine-tuning helps when SpeciesNet does not already handle your local species well.
- Rough guidance:
  - `<100` distinct examples/class: probably too little.
  - `~1000+` examples/class: better.
  - Rare or visually similar species may need merging.

## 2. Clone the tutorial repository

```bash
git clone https://github.com/agentmorris/speciesnet-fine-tuning .
```

## 3. Create and activate a Python environment

```bash
mamba create -n speciesnet-finetuning python=3.12 pip -y
mamba activate speciesnet-finetuning
pip install -r requirements.txt
```

## 4. Prepare your image-label CSV

Create a CSV with one row per labeled image:

```csv
filename,category,location
A01/01100085.JPG,black_agouti,A01
A01/01140096.JPG,collared_peccary,A01
A01/01140097.JPG,empty,A01
```

Required columns:

- `filename`: image path
- `category`: species/class label, e.g. `rodent`, `empty`, `mouse`
- `location`: camera/site/deployment ID

Important: `location` is used to split train/validation by camera, avoiding leakage.

## 5. If labels are COCO Camera Traps JSON, convert them

```bash
cd scripts
python coco_to_csv.py path/to/labels.json path/to/output.csv
```

Useful options:

- `--multiple-label-handling omit|all`
- `--unlabeled-image-handling omit|error|include`
- `--image-verification error|omit|warning`
- `--image-folder path/to/images`

## 6. Run MegaDetector on your images

SpeciesNet fine-tuning depends on MegaDetector boxes.

You need a MegaDetector-format results JSON covering the same images in your CSV.

Options:

- Use AddaxAI GUI.
- Or run MegaDetector from the command line.

## 7. Optionally prepare a mapping CSV

This lets you rename, merge, or remove classes without editing your raw labels.

```csv
input,output
black_agouti,agouti
orinoco_agouti,agouti
empty,remove
human,remove
spotted_paca,
```

Meaning:

- `input,output`
- If `output` is a class name: rename/merge.
- If `output` is `remove`: drop that class.
- If `output` is blank: keep unchanged.

## 8. Optionally generate a mapping template from COCO labels

```bash
python coco_to_mapping_file.py path/to/labels.json mapping.csv
```

## 9. Start fine-tuning

Minimal command:

```bash
python train.py \
  --data-csv c:/path/to/your/data.csv \
  --image-root c:/path/to/your/images \
  --md-results c:/path/to/your/megadetector_results.json \
  --mapping c:/path/to/your/mapping_file.csv \
  --run-folder c:/path/to/your/output/folder
```

Notes:

- `--run-folder` is required.
- It must usually not already exist.
- The trained model will be written as:

```text
run-folder/model_best.pt
```

## 10. Understand how blanks/empty images work

SpeciesNet fine-tuning trains on **MegaDetector crops**, not whole images.

So:

- If an image has no MegaDetector animal box, it produces no training crop.
- Most true blank images may contribute nothing.
- A `blank`/`empty` class is only learned from MegaDetector false-positive crops that were labeled blank/empty.
- “No animal box found” and “classifier predicts blank” are different things.

## 11. Key fine-tuning options

Common useful options:

```text
--min-instances 100
```

Drop classes with fewer than this many training crops.

```text
--val-fraction 0.15
```

Validation fraction, split by camera/location.

```text
--conf-threshold 0.3
```

Minimum MegaDetector confidence for a box to become a crop.

```text
--max-boxes 5
```

Maximum animal boxes per image.

```text
--epochs 20
```

Training epochs.

```text
--unfreeze-blocks 2
```

How much of SpeciesNet to fine-tune:

- `0`: train only the new classification head
- `1`, `2`, etc.: unfreeze last N backbone stages
- `-1`: train the whole network

Other useful options:

```text
--weighted-loss
--batch-size 32
--lr 1e-4
--workers 8
--patience 4
--devices auto
--seed 0
```

## 12. Inspect the run folder

Important outputs:

```text
summary.md
model_best.pt
checkpoints/
metrics.csv
config.json
split.csv
image_splits.json
hparams.yaml
```

Read `summary.md` first. It tells you:

- final classes
- crop counts per class
- train/validation split
- dropped classes
- missing images
- images with no usable MegaDetector box
- final metrics

## 13. Resume if training is interrupted

```bash
python train.py --resume c:/path/to/your/run/folder
```

You do not need to repeat all the original arguments.

## 14. Run the fine-tuned model

First run MegaDetector on the new images.

Then run:

```bash
python predict.py \
  c:/path/to/your/run/folder/model_best.pt \
  c:/path/to/your/megadetector_results.json \
  c:/path/to/your/new/images \
  c:/path/to/your/output_file.json
```

## 15. Prediction options

Useful options:

```text
--csv-output
--conf-threshold 0.1
--topk 3
--batch-size 32
--device auto
```

Default output is MegaDetector-format JSON with classifications added to animal boxes.

## 16. Review/use the results

Possible downstream workflows:

- Load results into Timelapse.
- Use MegaDetector postprocessing tools.
- Use AddaxAI if supported.
- Separate images into folders by predicted species.
- Sample model predictions manually for QA.

## 17. Evaluate carefully

Key cautions:

- Validation is only meaningful if `location` reflects real camera/site IDs.
- If all images have the same `location`, validation metrics are weak.
- Check rare classes in `summary.md`.
- Merge visually similar or rare classes if needed.
- For imbalanced datasets, try `--weighted-loss`.
- Use manual review/postprocessing to sanity-check predictions.
