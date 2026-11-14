# Counting behavior

AD4PT offers two different estimates. Compare runs made with the same counting
mode and settings, and inspect the saved `stop_reason` before using the totals.

## Tracking mode

`python -m ad4pt demo --counting-mode tracks` is the default. BoT-SORT assigns IDs
and compensates for camera motion using sparse optical flow. The preview retains
the detector's original boxes rather than displaying the tracker's smoothed boxes.
New detections may display `pending` before receiving an ID.

An ID contributes once after three tracked observations at the applicable count
threshold. Qualifying observations vote on its ripe/unripe label. The default
threshold is 0.25; weaker detections down to the tracker's 0.10 low threshold can
help association but do not confirm counts or vote on ripeness. Lost tracks remain
available for 30 processed frames. Confirmation and retention are configured in
`config/strawberry_botsort.yaml`.

Processed-frame counts are not durations: their elapsed time depends on inference
speed. A fruit that leaves view and returns with a new ID can count again. Motion,
occlusion, missed detections, and false positives can change the result. This mode
does not establish reliable unique-fruit totals along a flight path.

## Fixed-scene mode

```bash
python -m ad4pt demo --counting-mode scene
```

This bypasses BoT-SORT. A box must retain the same class and overlap a previous
box by IoU at least 0.3 for five consecutive processed frames before display or
counting. A miss or class change resets confirmation. Adjust with
`--scene-confirmation-frames N` (1–60); `1` exposes raw detections for diagnosis.
Persistent false boxes can still pass this filter.

The result is the most frequent **joint ripe/unripe count pair** across processed
frames, provided that pair has occurred at least three times. Keep the complete
arrangement visible throughout the run. Changing views, empty frames, and prolonged
occlusion can change the winning pair. This estimates a fixed scene rather than
accumulating fruit over a camera sweep.

JSON records `static_scene_confirmed_modal_detections`, box-confirmation settings,
and `scene_mode_support_frames`. Support is the winning pair's frequency, not a
probability of correctness. Below the required support, zero counts mean there
were insufficient supporting observations.

## Detection filters

The supplied model config uses 640 input, confidence 0.25, and these camera-specific
filters before display and counting:

- `ignore_top_fraction: 0.08` excludes the original camera's OSD strip. Fruit in
  this strip is also excluded. Use `0` for footage without an overlay.
- `max_box_area_fraction: 0.15` rejects boxes covering more than 15% of the cropped
  image. Increase it for close-ups, or use `1.0` to disable the area filter.
- `--unripe-confidence 0.5` optionally raises only the unripe count/display
  threshold. This can reduce foliage false positives and miss genuine green fruit.

Temporal filtering and confidence changes do not demonstrate improved learned
recognition. Validate against manually counted scenes and distinguish replica,
printed-photo, and real-plant experiments.

## Output and captures

Normal shutdown, disconnect, Ctrl-C, and handled runtime errors save counts so far.
Very brief appearances may remain unconfirmed. Configuration failures can exit
before a run starts. Failed output writes, force-kills, and power loss can prevent
saving.

By default, no frames or per-frame detection logs are saved. To enable optional
raw-image capture:

```bash
python -m ad4pt demo --capture-dir datasets/camera-errors
```

Press **S** while an error is visible. The timestamped PNG has no detection boxes.
These images remain local because `datasets/` is ignored by Git.
