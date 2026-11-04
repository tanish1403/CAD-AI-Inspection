#!/usr/bin/env python
"""Run the pretrained-only detector over every supplied real photograph."""
import argparse, json, sys
from pathlib import Path
import cv2, pandas as pd, yaml
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
sys.path.insert(0,str(ROOT/"src"))
from cad_inspector.dataset import load_case
from cad_inspector.models import FoundationModels
from cad_inspector.inference import inspect_case
from cad_inspector.visualization import save_overlay

def to_rgb(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(path)
    if im.ndim==2:return cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(im,cv2.COLOR_BGRA2RGB if im.shape[2]==4 else cv2.COLOR_BGR2RGB)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/sample/AI_sample_CAD")
    ap.add_argument("--output",default="outputs/full_supplied_experiment")
    ap.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    ap.add_argument("--config",default=str(ROOT/"configs/config.yaml"))
    args=ap.parse_args()
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(open(args.config,encoding="utf-8"))
    model=FoundationModels(cfg,args.device)
    case_dirs=[data]+[p for p in data.iterdir() if p.is_dir() and any(p.glob("*.json"))]
    all_rows=[]
    for folder in case_dirs:
        if not any(folder.glob("*.json")) or not any(folder.glob("Mask_*")):continue
        base=load_case(folder)
        captures=sorted(folder.glob("BG_*.png"))
        for img_path in captures:
            base["real"]=to_rgb(img_path);base["real_path"]=img_path
            run_name=f"{folder.name}__{img_path.stem}"
            run_out=out/run_name;run_out.mkdir(parents=True,exist_ok=True)
            rows=inspect_case(base,model,cfg,run_out)
            payload={"case":folder.name,"real_image":img_path.name,"results":rows}
            (run_out/"inspection_results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
            pd.DataFrame(rows).to_csv(run_out/"inspection_results.csv",index=False)
            save_overlay(base["real"],rows,run_out/"annotated_result.png")
            for row in rows:all_rows.append({"case":folder.name,"real_image":img_path.name,**row})
            print(f"{folder.name}/{img_path.name}: {len(rows)} expected components")
    df=pd.DataFrame(all_rows)
    df.to_csv(out/"all_predictions.csv",index=False)
    summary=df.groupby(["case","real_image","status"]).size().unstack(fill_value=0).reset_index()
    summary.to_csv(out/"view_summary.csv",index=False)
    print("\nVIEW SUMMARY\n",summary.to_string(index=False))
    print("\nSaved experiment outputs:",out.resolve())
    print("These are model predictions, not accuracy metrics; ground-truth labels are not supplied.")
if __name__=="__main__":main()
