# CAD-AI Inspector — Reviewer Solution

## 1. Objective

The task is to compare a real photograph of a mechanical assembly against a CAD reference package (CAD rendering, instance mask, and JSON part/stencil mapping), identify which expected components are visually supported, and provide a defensible missing-part decision. The second task is to outline how the system can verify component position and dimensions against CAD.

## 2. Proposed solution

This implementation uses pretrained models only (no custom training or fine-tuning):

- **Grounding DINO Tiny** (`IDEA-Research/grounding-dino-tiny`) proposes object bounding boxes from text prompts for broad mechanical-part families.
- **DINOv2 Small** (`facebook/dinov2-small`) extracts frozen visual embeddings from CAD-derived component crops and real-image candidate crops. Cosine similarity ranks likely correspondences.
- **OpenCV** handles image/mask processing, crop extraction, non-maximum suppression, and result visualization.
- **JSON + instance mask** provide the expected component inventory and the mapping between stencil IDs and component names.

The system is reference-guided and zero-shot: it does not learn the user's component classes from labelled training examples.

## 3. End-to-end pipeline

1. **Input validation:** load the real photograph, CAD rendering, instance mask, and JSON mapping; verify dimensions, IDs, and files.
2. **Inventory construction:** parse each expected part name and its stencil/mask ID from JSON.
3. **CAD reference extraction:** read the mask's red channel and crop the corresponding component region from the CAD render for each available ID.
4. **Prompt-family assignment:** map part names to broad families such as bolt, washer, nut, cover, or box. Aliases are configurable.
5. **Real-image proposals:** run Grounding DINO once per family prompt and collect candidate boxes and detector scores.
6. **Candidate cleanup:** apply confidence filtering and non-maximum suppression; crop each remaining candidate.
7. **Feature extraction:** run DINOv2 on CAD reference crops and real candidate crops. Cache candidate embeddings to avoid repeated computation.
8. **Reference matching:** calculate cosine similarity between each reference crop and compatible real candidates. Use one-to-one greedy assignment so one detected object cannot satisfy multiple expected instances.
9. **Decision policy:** report `PRESENT` only when a candidate is detected and its CAD-reference similarity exceeds the configured threshold. Report `ABSENT` only when a validated real-image ROI is supplied, the part is confirmed visible in that ROI, and no candidate is detected there. Otherwise report `UNCERTAIN`.
10. **Outputs:** save annotated image, CSV, JSON, and debug CAD reference crops for review.

## 4. Decision logic and scientific safeguards

A detector miss is not proof of absence. A component can be missed because it is small, occluded, poorly illuminated, outside the detector vocabulary, or visually different from its CAD rendering. Therefore, the default result for insufficient evidence is `UNCERTAIN`.

CAD mask coordinates refer to the CAD rendering. They are not automatically coordinates in the real photograph. A real-image ROI must come from manual annotation or a validated CAD-to-camera registration. The sample ROI file is a schema example, not a set of ground-truth annotations.

The combined detector/similarity score is a ranking score, not a calibrated probability. Thresholds in the configuration are starting points and must be calibrated on a labelled validation set.

## 5. Dataset audit

The supplied archive contains two cases, each with 32 JSON component entries.

| Case | Real-image views | Image dimensions | IDs with nonzero mask pixels | IDs with zero mask pixels |
|---|---:|---:|---:|---:|
| TestCase1 | 4 | 4032 × 3024 | 29 / 32 | 3 / 32 |
| TestCase2 | 1 | 2902 × 2176 | 22 / 32 | 10 / 32 |

The mask instance IDs are encoded in the red channel (OpenCV BGR index 2). A zero-pixel ID means no CAD reference crop is available for that entry; it does not establish that the component is absent from the real photograph.

## 6. Execution on Kaggle GPU

1. Upload `CAD_AI_Inspection_OptionA_Kaggle.zip` as a Kaggle dataset/input.
2. Create a Kaggle Notebook and enable GPU; enable internet for first-time model downloads.
3. Extract the archive under `/kaggle/working/`.
4. Install dependencies with `pip install -r requirements.txt`.
5. Download public checkpoints with `python scripts/download_models.py --models grounding-dino dinov2`.
6. Run the supplied notebook or execute:

```bash
python scripts/run_all_views.py \
  --data data/sample/AI_sample_CAD \
  --output outputs/full_supplied_experiment \
  --device auto
```

The notebook/project README contains the exact paths and troubleshooting guidance. Model weights are downloaded at runtime and are not bundled in the project archive.

## 7. Expected deliverables from each run

- `annotated_result.png`: real photograph with candidate boxes and inspection labels.
- `inspection_results.csv`: tabular per-component results.
- `inspection_results.json`: machine-readable per-component details and scores.
- `debug/cad_reference_crops/`: extracted CAD reference crops for traceability.
- Full experiment aggregation: `all_predictions.csv` and `view_summary.csv`.

## 8. Evaluation plan

The supplied data does not include per-component present/absent ground-truth labels for every real-image view. Consequently, accuracy, precision, recall, and F1 cannot honestly be reported from this archive alone.

To evaluate quantitatively, manually annotate a held-out set with `PRESENT`, `ABSENT`, and `OCCLUDED/UNCERTAIN` labels. Then report:

- precision, recall, and F1 for present-part recognition;
- missing-part recall and false-absence rate;
- uncertainty/review coverage;
- bounding-box IoU where localization annotations are available.

Threshold selection must use a validation set separate from the final test set. Include per-family results because bolts, washers, covers, and larger housings have different visual difficulty.

## 9. Problem 2 — position and dimensional verification architecture

A production metrology pipeline should add:

1. **Camera calibration:** estimate intrinsic parameters and lens distortion using a calibration target.
2. **CAD-to-camera registration:** estimate object pose from fiducials, known geometry, or robust 2D/3D feature correspondences.
3. **Part localization and pose:** estimate each component's translation and rotation relative to the registered assembly coordinate frame.
4. **Metric measurement:** use calibrated stereo, structured light, or depth sensing to recover 3D geometry. A single uncalibrated 2D image cannot guarantee true 3D dimensions.
5. **CAD/GD&T comparison:** compare measured position, orientation, dimensions, and surface deviations against nominal CAD values and specified tolerances.
6. **Inspection report:** output per-part deviations, tolerance pass/fail, measurement uncertainty, and evidence images.

A useful architecture is: calibrated RGB/depth capture → assembly pose registration → instance/part localization → 3D measurement → CAD coordinate transformation → GD&T tolerance engine → traceable inspection report.

## 10. Limitations and next steps

- Tiny fasteners may be missed by a general open-vocabulary detector.
- CAD-to-photo domain gap (viewpoint, lighting, material, occlusion) can reduce embedding similarity.
- Similar-looking fasteners can be confused; component-specific prompts and geometric constraints may be needed.
- Missing-part decisions require validated expected ROIs or registration; absence cannot be inferred from detector failure alone.
- The current implementation is a complete runnable inference pipeline, but model inference must be run in the target Kaggle environment. No model accuracy or experimental detection results are claimed before that run.

Recommended next steps: run all five supplied views on Kaggle, inspect annotated outputs, manually label a validation subset, calibrate thresholds, then report measured metrics with a clear test protocol.

## 11. Official model references

- Grounding DINO Transformers documentation: https://huggingface.co/docs/transformers/model_doc/grounding-dino
- Grounding DINO Tiny: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- DINOv2 Small: https://huggingface.co/facebook/dinov2-small
- PyTorch installation: https://pytorch.org/get-started/locally/
