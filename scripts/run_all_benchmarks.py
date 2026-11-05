#!/usr/bin/env python3
"""Run the complete CAD inspection model benchmark from one command.

Each experiment is isolated. Missing optional packages, unavailable model revisions,
network failures, and CUDA OOMs are recorded as SKIPPED/FAILED rather than aborting
all remaining experiments. No model score is interpreted as calibrated probability.
"""
from __future__ import annotations
import argparse, csv, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label, cmd, out_dir, records, timeout=None):
    start = time.time(); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"experiment": label, "command": [str(x) for x in cmd], "output_dir": str(out_dir)}
    try:
        p = subprocess.run([str(x) for x in cmd], cwd=ROOT, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=os.environ.copy())
        (out_dir / "run.log").write_text(p.stdout or "", encoding="utf-8")
        rec.update(status="COMPLETED" if p.returncode == 0 else "FAILED",
                   returncode=p.returncode, log=str(out_dir / "run.log"),
                   tail=(p.stdout or "")[-5000:])
    except subprocess.TimeoutExpired as e:
        rec.update(status="TIMEOUT", error=str(e))
    except Exception as e:
        rec.update(status="FAILED", error=repr(e), traceback=traceback.format_exc())
    rec["elapsed_seconds"] = round(time.time() - start, 2)
    records.append(rec)
    print(f"[{rec['status']}] {label} ({rec['elapsed_seconds']}s)", flush=True)
    return rec


def discover_cases(data_root):
    root = Path(data_root)
    folders = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    cases = []
    for folder in folders:
        if (folder / "FG_00.png").exists() and (folder / "Mask_00_00.bmp").exists() and list(folder.glob("*.json")):
            images = sorted(folder.glob("BG_00*.png")) + sorted(folder.glob("BG_00*.jpg"))
            for img in images:
                cases.append((folder, img))
    return cases


def main():
    ap = argparse.ArgumentParser(description="Run all CAD-AI inspection benchmarks in one command")
    ap.add_argument('--data', default='data/sample/AI_sample_CAD')
    ap.add_argument('--output', default='outputs/full_benchmark')
    ap.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    ap.add_argument('--max-side', type=int, default=1200)
    ap.add_argument('--experiments', default='all', help='all or comma list: baseline,registration,dinov3,florence,qwen,locateanything')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--skip-locateanything', action='store_true', help='Recommended on Kaggle T4 and smaller GPUs')
    ap.add_argument('--skip-qwen', action='store_true')
    ap.add_argument('--skip-florence', action='store_true')
    ap.add_argument('--skip-dinov3', action='store_true')
    ap.add_argument('--skip-registration', action='store_true')
    ap.add_argument('--timeout-minutes', type=int, default=90)
    args = ap.parse_args()
    data = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    out = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    selected = set(['baseline','registration','dinov3','florence','qwen','locateanything'] if args.experiments == 'all' else args.experiments.split(','))
    skips = {'baseline':args.skip_baseline,'registration':args.skip_registration,'dinov3':args.skip_dinov3,
             'florence':args.skip_florence,'qwen':args.skip_qwen,'locateanything':args.skip_locateanything}
    cases = discover_cases(data)
    if not cases:
        raise SystemExit(f'No cases/views found under {data}. Expected FG_00.png, Mask_00_00.bmp, JSON and BG_00*.png.')
    try:
        import torch
        gpu = {"cuda_available":torch.cuda.is_available(), "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "vram_gb":round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else None}
    except Exception as e: gpu = {"torch":"unavailable", "error":str(e)}
    manifest = {"project":"CAD_AI_Inspection", "started_at":time.strftime('%Y-%m-%d %H:%M:%S'),
                "python":sys.version, "platform":platform.platform(), "gpu":gpu,
                "data_root":str(data), "views":[{"case":str(c),"image":str(i)} for c,i in cases],
                "selected_experiments":sorted(selected), "warning":"Benchmark outputs are diagnostics. No accuracy claims without manual ground truth."}
    (out/'environment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records=[]; timeout=args.timeout_minutes*60
    for case, image in cases:
        tag=(case.name+'_'+image.stem).replace(' ','_')
        case_out=out/tag; case_out.mkdir(parents=True,exist_ok=True)
        # Registration-first path, with safe failure reporting.
        reg_out=case_out/'registration'
        if 'registration' in selected and not skips['registration']:
            run('registration_'+tag,[PY,'scripts/run_registration_poc.py','--case',case,'--real-image',image,'--output',reg_out,'--max-side',args.max_side],reg_out,records,timeout)
        # Original detector/embedding baseline.
        if 'baseline' in selected and not skips['baseline']:
            base_out=case_out/'baseline'
            run('grounding_dino_dinov2_'+tag,[PY,'scripts/run_inspection.py','--case',case,'--output',base_out,'--device',args.device,'--real-image',image],base_out,records,timeout)
        # DINOv3 dense patch diagnostic.
        if 'dinov3' in selected and not skips['dinov3']:
            dino_out=case_out/'dinov3'
            run('dinov3_'+tag,[PY,'scripts/compare_dinov3.py','--cad',case/'FG_00.png','--real',image,'--output',dino_out],dino_out,records,timeout)
        # Florence-2 open-vocabulary grounding adapter.
        if 'florence' in selected and not skips['florence']:
            fl_out=case_out/'florence2'
            run('florence2_'+tag,[PY,'scripts/run_florence2.py','--image',image,'--output',fl_out],fl_out,records,timeout)
        # Qwen VLM performs local/whole-image qualitative review; no auto absence claims.
        if 'qwen' in selected and not skips['qwen']:
            q_out=case_out/'qwen_vl'
            run('qwen3_vl_'+tag,[PY,'scripts/run_qwen_vl_review.py','--image',image,'--cad',case/'FG_00.png','--json',next(case.glob('*.json')),'--output',q_out],q_out,records,timeout)
        # NVIDIA LocateAnything: optional, resource-intensive and license-restricted.
        if 'locateanything' in selected and not skips['locateanything']:
            la_out=case_out/'locateanything'
            run('locateanything_'+tag,[PY,'scripts/run_locateanything.py','--image',image,'--output',la_out],la_out,records,timeout)
    # Consolidated run report
    (out/'benchmark_runs.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    with (out/'benchmark_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['experiment','status','elapsed_seconds','output_dir','returncode','error']); w.writeheader()
        for r in records: w.writerow({k:r.get(k,'') for k in w.fieldnames})
    counts={s:sum(r['status']==s for r in records) for s in ['COMPLETED','FAILED','TIMEOUT']}
    summary={"views":len(cases),"experiments_attempted":len(records),"status_counts":counts,
             "outputs":str(out),"interpretation":"COMPLETED means the script ran, not that its predictions are correct. Review per-model artifacts and manually label ground truth before computing accuracy."}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nBENCHMARK FINISHED\n'+json.dumps(summary,indent=2))

if __name__=='__main__': main()
