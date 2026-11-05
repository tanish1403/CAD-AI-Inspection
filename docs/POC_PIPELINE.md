# POC pipeline and reviewer notes

## Design

The registration-first path is deliberately separate from the original Grounding DINO + DINOv2 baseline. It first estimates a mapping from CAD image coordinates to real-photo coordinates, then transfers mask-derived component regions. The output is a review queue, not an automated pass/fail inspection.

## Registration modes

- Automatic: SIFT, AKAZE, and ORB are attempted with ratio filtering and RANSAC. Weak solutions are rejected.
- Manual: a JSON file contains at least four paired full-resolution pixel coordinates (`cad_points`, `real_points`). The desktop helper collects these points; the Kaggle notebook includes interactive Matplotlib point selection.
- The homography is planar. A separate homography is needed for each distinct visible plane. A single homography over front, top, and side faces is geometrically invalid.

## Model experiments

1. Original baseline: Grounding DINO Tiny proposals + DINOv2 CLS embeddings. Preserve outputs and check non-finite scores.
2. DINOv3: ViT-S/16 dense patch descriptors. Compare patch similarity and inspect heatmaps. It is a feature backbone, not an open-vocabulary detector.
3. LocateAnything-3B: optional VLM grounding experiment. Save raw structured grounding output; visually validate coordinate parsing. This model is large and its model card reports A100/H100 testing; it is not recommended for 4 GB GPUs. The model card lists NVIDIA License for non-commercial research use.

## Decision policy

- No CAD mask pixels: `NO_CAD_REFERENCE`, not absent.
- Invalid/out-of-frame projected ROI: `PROJECTED_OUT_OF_FRAME`.
- Valid projected ROI: `REVIEW_REQUIRED` until semantic evidence or human inspection.
- `PRESENT` and `ABSENT` require a validated component-specific rule and known visibility. `UNCERTAIN` is used for occlusion, blur, bad alignment, or ambiguous evidence.

## Limitations

- A homography models one plane, not an entire 3D assembly.
- The supplied dataset does not provide per-view presence/absence ground truth labels.
- DINOv3 cosine similarity is not a calibrated probability.
- LocateAnything outputs need correct coordinate interpretation and manual validation.
- Dimensional verification requires calibration and depth/scale information beyond one uncalibrated photo.
