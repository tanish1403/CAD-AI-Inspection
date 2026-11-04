# Option A experiment protocol

## Run
`python scripts/run_all_views.py --data data/sample/AI_sample_CAD --output outputs/full_supplied_experiment --device auto`

## Experiment set
- TestCase1: four supplied real-photo captures.
- TestCase2: one supplied real-photo capture.
- Grounding DINO is run once per broad part family per view.
- DINOv2 compares CAD reference crops with candidate crops.
- Candidate assignment is one-to-one within each family.

## Reportable outputs
- Candidate detections and detector scores.
- DINOv2 cosine similarity and combined ranking score.
- PRESENT / UNCERTAIN / conditionally ABSENT status.
- Per-view component counts and annotated images.

## Do not claim accuracy without labels
The supplied data provides CAD part masks and JSON inventory, but does not provide complete per-real-view present/absent ground truth. Therefore prediction counts are not precision, recall, F1, or accuracy. Create a manually reviewed test-label file before reporting those metrics.

## Reproducibility
Record model checkpoint IDs, Transformers version, torch version, GPU name, configuration thresholds, and input image names. Do not tune thresholds on the final test images.
