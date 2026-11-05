# Benchmark Quickstart

## One command (Windows / Linux / Kaggle)

```bash
python scripts/run_all_benchmarks.py --data data/sample/AI_sample_CAD --output outputs/full_benchmark --device cuda
```

On Kaggle T4, use `--skip-locateanything` if the 3B model is too large:

```bash
python scripts/run_all_benchmarks.py --data data/sample/AI_sample_CAD --output outputs/t4_benchmark --device cuda --skip-locateanything
```

The script runs baseline, registration, DINOv3, Florence-2, Qwen3-VL and (unless skipped) NVIDIA LocateAnything sequentially over all discovered `BG_00*.png`/`.jpg` views. It records per-experiment status, duration, command and log. Failed models do not stop the other experiments.

## Select experiments

```bash
python scripts/run_all_benchmarks.py --experiments registration,dinov3,florence,qwen --device cuda
```

## Important interpretation

- DINOv3 is a dense-feature diagnostic, not a detector or probability.
- Florence-2 and LocateAnything provide candidate detections, not verified part identities.
- Qwen3-VL provides qualitative observations, not ground-truth labels.
- Registration must pass geometric validation before projected ROIs are trusted.
- No precision/recall/F1 can be claimed until real-photo part labels are manually annotated.
- LocateAnything is optional and its model license must be checked before use or redistribution.
