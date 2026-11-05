# CAD-AI Inspection POC

A registration-first, model-assisted proof of concept for comparing a CAD render + instance mask + JSON part map against a real photograph. It is designed to run either locally on Windows or in Kaggle GPU.

## What this POC does

1. Reads the supplied JSON mapping and extracts per-stencil component regions from the red channel of the BMP mask.
2. Aligns the CAD render to the real photograph using local feature matching (SIFT, AKAZE, ORB), or user-provided point correspondences when automatic registration is unreliable.
3. Transfers the CAD component regions into the real-photo coordinate system.
4. Produces alignment overlays, projected component ROIs, diagnostics, and a component review CSV.
5. Keeps the original Grounding DINO + DINOv2 implementation as a baseline.
6. Adds an experimental DINOv3 dense-feature diagnostic and an optional NVIDIA LocateAnything grounding experiment.

**Important:** this is a POC, not a validated industrial inspection system. The supplied images have viewpoint/lighting differences. A single homography is only a planar approximation; use it for a visible plane (for example, the front face), not for every surface of a 3D object. Local contrast/edge values and foundation-model similarity are not calibrated probabilities. The software intentionally does not turn an unverified low score into `ABSENT`.

## Project layout

```text
CAD_AI_Inspection/
├── README.md
├── REVIEWER_SOLUTION.md
├── requirements.txt
├── setup_windows.bat
├── CAD_AI_Inspector_Kaggle.ipynb
├── CAD_AI_Benchmark_All_Models.ipynb
├── configs/
├── data/sample/AI_sample_CAD/
├── models/                         # downloaded weights/cache
├── scripts/
│   ├── run_inspection.py           # original Grounding DINO + DINOv2 baseline
│   ├── run_registration_poc.py     # registration-first ROI transfer
│   ├── collect_manual_alignment.py # desktop point-picking helper
│   ├── compare_dinov3.py           # optional DINOv3 dense-feature diagnostic
│   ├── run_locateanything.py       # optional NVIDIA LocateAnything experiment
│   ├── run_all_benchmarks.py       # one-command full model comparison
│   ├── run_florence2.py            # Florence-2 grounding experiment
│   ├── run_qwen_vl_review.py       # Qwen3-VL qualitative review
│   ├── run_all_views.py
│   └── dataset_audit.py
├── src/cad_inspector/
└── outputs/
```


## One-command GPU benchmark (recommended)

The project now includes a single orchestrator that runs the baseline and all selected model experiments across every supplied real-image view. It runs each model in a separate subprocess so GPU memory is released between models, and a failed/unsupported model is logged without stopping the remaining experiments.

### Run everything

```bash
python scripts/run_all_benchmarks.py \
  --data data/sample/AI_sample_CAD \
  --output outputs/full_benchmark \
  --device cuda \
  --max-side 1200
```

This attempts: (1) original Grounding DINO + DINOv2 baseline, (2) registration and mask projection, (3) DINOv3 dense patch diagnostic, (4) Florence-2 Large open-vocabulary grounding, (5) Qwen3-VL 4B qualitative VLM review, and (6) NVIDIA LocateAnything-3B grounding. First run downloads model weights. Internet access and Hugging Face availability are required.

**Kaggle T4 recommendation:** run all experiments once, but LocateAnything-3B may exceed available memory or fail due to its custom model requirements. To prioritize the T4-friendly experiments, use:

```bash
python scripts/run_all_benchmarks.py --data data/sample/AI_sample_CAD --output outputs/t4_benchmark --device cuda --skip-locateanything
```

You can also select a subset: `--experiments registration,dinov3,florence,qwen`. Use `--skip-baseline`, `--skip-qwen`, `--skip-florence`, `--skip-dinov3`, `--skip-registration`, or `--skip-locateanything` to omit individual experiments.

### One-command Kaggle notebook

Open `CAD_AI_Benchmark_All_Models.ipynb`, enable GPU, attach/extract the project dataset, and run all cells. It invokes the same orchestrator and creates a downloadable benchmark archive under `outputs/kaggle_full_benchmark.zip`.

### Benchmark outputs

```text
outputs/full_benchmark/
├── environment.json
├── benchmark_runs.json
├── benchmark_summary.csv
├── SUMMARY.json
├── AI_sample_CAD_BG_00/
│   ├── baseline/
│   ├── registration/
│   ├── dinov3/
│   ├── florence2/
│   ├── qwen_vl/
│   └── locateanything/
└── ... one folder per view
```

`COMPLETED` means the script executed successfully, not that the model prediction is correct. `FAILED` and `TIMEOUT` entries include logs. The Qwen VLM result is a qualitative assembly-level review, not per-part ground truth. Florence/LocateAnything boxes are candidate proposals. DINOv3 similarity is a diagnostic, not a probability. Registration failure means projected component ROIs must not be trusted.

The benchmark uses sequential processes rather than loading all models into VRAM simultaneously. This is intentional for T4 compatibility. Model caches are retained by Hugging Face between runs. For a larger GPU, the same command works; increase `--max-side` cautiously.

## A. Local Windows setup

Recommended: Python 3.10 or 3.11. A GTX 1650 4 GB is suitable for the classical registration stage and the original baseline with CPU fallback/low resolution. DINOv3 ViT-S may fit with small inputs and FP16, but memory is limited. **LocateAnything-3B is not recommended on a 4 GB GPU.**

1. Extract the ZIP.
2. Open Command Prompt in the extracted project folder.
3. Run `setup_windows.bat` or:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA driver if you want GPU inference. For a first run, CPU is safest.

### Registration-first run

```bat
python scripts\run_registration_poc.py --case data\sample\AI_sample_CAD --output outputs\case1_registration
python scripts\run_registration_poc.py --case data\sample\AI_sample_CAD\TestCase2 --output outputs\case2_registration
```

If automatic registration fails (expected when viewpoint/appearance differences are large), create manual correspondences:

```bat
python scripts\collect_manual_alignment.py --cad data\sample\AI_sample_CAD\FG_00.png --real data\sample\AI_sample_CAD\BG_00.png --output configs\case1_alignment.json --points 4
python scripts\run_registration_poc.py --case data\sample\AI_sample_CAD --manual-points configs\case1_alignment.json --output outputs\case1_manual
```

Click at least four distinctive corresponding points, in the same order, in both windows. For a front-face homography, choose points on the **same front plane** (e.g. the four front-panel corners). Do not mix front-face points with points on the top/side faces. A homography cannot correctly flatten a multi-plane 3D object.

For another real photo, add `--real-image path\to\photo.png` to `run_registration_poc.py` and collect matching points against that photo.

Outputs:
- `registration_overlay.png`
- `projected_component_rois.png`
- `warped_cad.png`
- `component_review.csv`
- `summary.json`

The component CSV uses `REVIEW_REQUIRED`, `NO_CAD_REFERENCE`, or `PROJECTED_OUT_OF_FRAME`; it does not claim component presence/absence.

## B. Original model baseline

The original Grounding DINO + DINOv2 pipeline is preserved for comparison:

```bat
python scripts\download_models.py --models grounding-dino dinov2
python scripts\run_inspection.py --case data\sample\AI_sample_CAD\TestCase2 --output outputs\baseline_testcase2 --device cpu
```

The baseline's scores are ranking signals only. Check for NaN/None values in all exported scores before interpreting results. Do not compare a baseline run against a different code revision without recording the revision/configuration.

## C. DINOv3 experiment

DINOv3 is used here as a dense visual-feature diagnostic, not as a detector. It produces patch-level features; the script reports best-patch cosine similarity and a heatmap. It does **not** prove that a particular named component exists.

```bash
pip install -U "transformers>=4.56.0" timm
python scripts/compare_dinov3.py --cad data/sample/AI_sample_CAD/FG_00.png --real data/sample/AI_sample_CAD/BG_00.png --output outputs/dinov3_case1
```

Default model: `facebook/dinov3-vits16-pretrain-lvd1689m`. The first run downloads weights. Use a small ViT-S model on Kaggle T4; avoid large variants for this POC. DINOv3 dense features are described in the official [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/dinov3) and [Meta repository](https://github.com/facebookresearch/dinov3).

## D. NVIDIA LocateAnything experiment (optional)

LocateAnything-3B is a 3B-parameter vision-language grounding model. Its official model card uses custom Transformers code and recommends high-memory NVIDIA GPUs; its model card reports A100/H100 test hardware and large memory use. Do not attempt it on a GTX 1650 4 GB. Kaggle T4 16 GB may also be insufficient depending on context/resolution/runtime. Run only if the GPU has sufficient free memory and the model loads successfully:

```bash
pip install -U "transformers==4.57.1" peft decord
python scripts/run_locateanything.py --image data/sample/AI_sample_CAD/BG_00.png --output outputs/locateanything_case1
```

The script saves the raw grounding response and any `<box>` tokens it finds. Inspect/parse/validate the coordinates visually; the experiment does not produce component-level presence/absence labels. Review the [official model card and NVIDIA license](https://huggingface.co/nvidia/LocateAnything-3B) before redistribution or use. The card states non-commercial research use only.

## E. Kaggle GPU workflow

1. Upload this ZIP as a Kaggle Dataset (or add it as notebook input).
2. Create a Kaggle notebook and enable GPU.
3. Open `CAD_AI_Inspector_Kaggle.ipynb` from this package and run cells in order.
4. The notebook installs requirements, audits the dataset, runs registration, and provides optional cells for DINOv3, LocateAnything, and the original baseline.
5. Download the generated `outputs` folder as a ZIP and share it for analysis.

If automatic registration fails, use the notebook's manual point-selection cell (matplotlib click selection) and then rerun registration with the saved correspondence JSON.

## F. How to compare experiments fairly

Record for every run: model/revision, prompt, image view, image resolution, GPU, registration method, inlier count/ratio, runtime, peak VRAM, output artifact paths, and manual reviewer labels. Compare part-level precision/recall/F1 only after creating a ground-truth table for each real image. The current supplied dataset does not include per-view component-presence labels, so no honest accuracy/F1 can be computed yet.

Suggested table:

| Run | Registration | Proposal/model | # mapped ROIs | # human-confirmed present | # human-confirmed absent | # uncertain | Runtime | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|

## G. Decision policy and limitations

- `NO_CAD_REFERENCE`: JSON part has no pixels for its stencil ID in the supplied mask. This does not mean the real component is absent.
- `PROJECTED_OUT_OF_FRAME`: projected expected region is outside/invalid.
- `REVIEW_REQUIRED`: CAD region projected; a person or separately validated detector must inspect it.
- `PRESENT`: only add after positive component-specific visual evidence is validated.
- `ABSENT`: only add if the expected region is known to be visible, registration is valid for that plane, and a validated detector/human confirms absence.
- `UNCERTAIN`: occlusion, poor registration, blur, insufficient resolution, or conflicting evidence.

A single photograph cannot guarantee 3D dimensions or GD&T. Problem 2 would require camera calibration, pose/plane registration, calibrated scale/depth (stereo/structured light preferred), and a tolerance engine.

## Research/model references

- [DINOv3 official Transformers docs](https://huggingface.co/docs/transformers/model_doc/dinov3)
- [DINOv3 official repository](https://github.com/facebookresearch/dinov3)
- [NVIDIA LocateAnything model card](https://huggingface.co/nvidia/LocateAnything-3B)
- [NVIDIA LocateAnything research page](https://research.nvidia.com/labs/lpr/locate-anything/)
- [Grounding DINO Transformers docs](https://huggingface.co/docs/transformers/model_doc/grounding-dino)

To attempt automatic registration across every supplied photo view and save a per-view success/failure log:

```bash
python scripts/run_registration_views.py --data data/sample/AI_sample_CAD --output outputs/registration_all_views
```

Automatic registration failures are expected on difficult viewpoint changes. The script logs failures and does not fabricate projected ROIs. Use manual correspondences for a selected image.
