# Improve the strawberry detector through fine-tuning

Goal: change learned recognition of fruit versus foliage, then compare old and
new weights at the SAME inference settings. Confidence changes and temporal
filters are separate experiments; they are not model improvements.

The existing model is YOLO11s, with two classes: 0 strawberry-ripe and 1
strawberry-unripe. It can be fine-tuned on an Apple Silicon MPS GPU or CPU.
This workflow preserves the active checkpoint and writes candidates separately.

## Data we need

- Real ripe and unripe strawberry photos, including small and partly hidden fruit.
  The six smoke photos are development examples, not an independent test set.
  Additional labelled real-fruit images from a documented public dataset can
  supplement the limited local examples; check its labels and provenance first.
- APFPV frames of the replica arrangement from different angles, distances and
  lighting. Label all four replicas as ripe, not the artificial leaves.
- APFPV foliage-only frames after removing every replica. These are hard negative
  examples: their reviewed label files are empty. In mixed frames, label every
  visible fruit and leave leaves as background. Do not mark the entire image
  negative if it contains any strawberries.

Start with a small, diverse adaptation set and inspect validation. As a practical
collection target, aim for 100–200 genuinely varied APFPV frames, not hundreds of
adjacent copies. This is not a guaranteed sufficient dataset size. Keep real-fruit
positives in the training mix so fine-tuning is not exclusively on artificial red
fruit/foliage. A replica-only result cannot demonstrate improved green-fruit or
real-plant detection. Too little variety can teach the exact arrangement instead
of generalizing. Validate class balance before choosing the final mix.

## Labels and independent scenes

Use reviewed YOLO detection annotations, one text file per image:

```
class_id centre_x centre_y width height
```

Coordinates are normalized to the image width/height. One physical strawberry
gets one box and one class; do not box its red and green portions as separate
fruit. Define a consistent rule for partly coloured fruit before annotating.
There is no third foliage class: foliage is background for this detector.
Model-generated proposals must be corrected by a human, especially missed fruit
and the current false foliage boxes. Empty files mean reviewed absence of fruit,
not missing annotations. Use raw images, not screenshots with boxes drawn on them.

Place data under `datasets/strawberry-adaptation/`:

```
images/train/scene01__frame001.png
labels/train/scene01__frame001.txt
images/val/scene02__frame001.png
labels/val/scene02__frame001.txt
images/test/scene03__frame001.png
labels/test/scene03__frame001.txt
```

The prefix before `__` identifies the scene/session/source-photo group. Keep ALL
nearby frames, augmentations, and printed/camera versions of the same source photo
in the SAME split. The script checks group IDs and exact file duplicates; it
cannot detect all near duplicates or incorrectly assigned scene IDs. Use roughly
70/15/15 train/validation/test by independent group when enough groups exist,
retaining both fruit classes. Validation selects settings; reserve final test
scenes until model choices are finished. The same arrangement filmed repeatedly
is not an independent test simply because the filenames or angles differ.

## Commands

From the repository root:

```bash
conda activate ad4pt
python scripts/finetune_strawberries.py
```

This only checks the dataset and fails until images and reviewed labels exist.
It does not import the training framework or download/start a model.

Once labelled data is ready:

```bash
python scripts/finetune_strawberries.py --train --device mps --epochs 50 --batch 4
```

This starting recipe uses the existing checkpoint, 640 input, AdamW at 0.0001,
first 10 layers frozen, early stopping patience 10, workers 0 and AMP disabled.
These are starting settings, not tuned optimums. If memory is insufficient,
reduce batch to 2 or 1. Measure training time; no completion time is promised.
Ultralytics may fetch plotting fonts if they are absent. Checkpoint identity is
verified before loading. Training outputs go to a new `runs/training/` directory.
Images and weights remain ignored by Git and outside the live app's imports.

## Decide whether the MODEL improved

The script evaluates old and candidate weights on the same validation set at
640 input and the same validation settings, then saves:

- `validation-comparison.json`: per-class precision, recall, AP50 and AP50–95.
- Standard Ultralytics training/validation plots and CSV metrics.
- `finetune/weights/best.pt`: separately saved candidate weights.
- `candidate-detector.json`: candidate model path/hash; optional stricter unripe
  threshold reset to null. This does not change the active detector config.

The AP validation pass uses a 0.001 score floor to build curves; its precision
and recall are not the same as live fixed-0.25 counting accuracy. Also compare raw
frame detections at 0.25 on held-out camera scenes, with identical crop/size rules
and without using temporal confirmation to claim recognition improvement.
Look for fewer foliage false positives while maintaining or improving unripe
recall and ripe performance. Report per-class metrics, not only a total count
that could hide offsetting errors. Final test-set evaluation remains separate;
validation results must not be labelled final unseen-test performance.

Only choose the candidate if measured evidence supports improvement. A detector
improvement does not by itself solve moving-drone unique-fruit counting.

## Sources

- Ultralytics training and Apple Silicon MPS:
  https://docs.ultralytics.com/modes/train/
- YOLO detection annotation format:
  https://docs.ultralytics.com/datasets/detect/
- Background negatives, complete annotations and split discipline:
  https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/
- Additional real-fruit data sources to review:
  Strawberry-DS https://pmc.ncbi.nlm.nih.gov/articles/PMC10165176/
  and its primary repository https://data.mendeley.com/datasets/z6dtfdpzz8/1 .
  It has six maturity stages. Document and review the mapping to two classes
  before use; do not silently drop intermediate-stage fruit annotations.

## Supporting tools

The scripts below operate on your local data. No research dataset or correction
list is included in the public checkout. Run `--help` for their options.

- `prepare_strawberry_dataset.py --source PATH`: convert a YOLO export with the
  exact ordered classes `Early-Turning, Green, Late-Turning, Red, Turning, White`
  to the two-class training layout. Only Red becomes ripe; all other stages become
  unripe. This is an experimental grouping, not a universal maturity definition.
  Source-photo IDs and identical image hashes help keep duplicates out of training
  when they also appear in held-out splits.
- `apply_strawberry_label_corrections.py --corrections PATH`: apply a supplied,
  individually reviewed JSON correction list, checking original label hashes and
  keeping backups. It does not discover or approve corrections automatically.
- `evaluate_fixed_threshold.py --candidate PATH`: compare baseline and candidate
  detector JSON configs on the same split at confidence 0.25. Results default to
  `reports/evaluation/fixed-threshold.json`. Add `--split test` only for your final
  held-out evaluation.

For count-report plots, install the optional reporting dependency:

```bash
python -m pip install -e '.[reports]'
python scripts/count_report.py --trials reports/my-trials.csv
```

Create your own CSV with at least these columns, using actual manual observations
and paths to completed run JSONs (paths resolve from the repository root):

```csv
trial_id,counts_json,actual_ripe,actual_unripe
```

Add one row per trial. Generated CSVs, plots, summary JSON, and Markdown tables go
to `reports/testing/results/` by default. These are count-agreement measurements,
not object-detection precision or recall. Generated reports are ignored by Git.
