# Experiment plan

## Baselines
1. Grounding DINO only: open-vocabulary detection recall by component family.
2. DINOv2 reference similarity: candidate ranking quality.
3. Grounding DINO + DINOv2: combined candidate ranking.
5. Few-shot DINOv2 linear probe: compare against zero-shot results.

## Dataset split
Split by physical assembly / capture session, not random crops from the same photograph. Keep validation/test images unseen during threshold selection.

## Metrics
- Component-level precision, recall, F1.
- Missing-part recall and false-absence rate.
- False acceptance rate (wrong component accepted).
- Localization IoU where boxes are labelled.
- Calibration / uncertainty coverage: accuracy on non-uncertain predictions and percentage sent to review.

## Required annotation
For each real image, label each expected component as present, absent, occluded/uncertain. If present, draw a bounding box or mask. Keep a small validation set reviewed by a human.

## Ablations
Compare aliases/prompts, thresholds, input resolution, DINOv2 backbone, and pretrained feature similarity baseline. Report actual measurements only; do not invent results.
