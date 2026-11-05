#!/usr/bin/env python
import argparse, json, sys
from pathlib import Path
import yaml, pandas as pd
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0,str(ROOT/"src"))
from cad_inspector.dataset import load_case
from cad_inspector.models import FoundationModels
from cad_inspector.inference import inspect_case
from cad_inspector.visualization import save_overlay

def main():
    p=argparse.ArgumentParser(description="CAD-aware foundation-model inspection")
    p.add_argument("--case",required=True,help="Folder with real image, CAD mask and part JSON")
    p.add_argument("--output",default="outputs/run")
    p.add_argument("--real-image",default=None,help="Optional real image override (e.g. BG_00_TestImage2.png)")
    p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    p.add_argument("--config",default=str(ROOT/"configs/config.yaml"))
    p.add_argument("--rois",default=None,help="Optional JSON: {part_name: {bbox:[x0,y0,x1,y1], visible:true}} in REAL-image pixels")
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8"))
    case=load_case(a.case)
    if a.real_image:
        import cv2
        from pathlib import Path
        image=cv2.imread(a.real_image)
        if image is None: raise FileNotFoundError(a.real_image)
        case["real"]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        case["real_path"]=Path(a.real_image)
    models=FoundationModels(cfg,a.device)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    roi_map=json.loads(Path(a.rois).read_text(encoding="utf-8")) if a.rois else None
    if roi_map and "parts" in roi_map: roi_map=roi_map["parts"]
    results=inspect_case(case,models,cfg,out,roi_map=roi_map)
    (out/"inspection_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    pd.DataFrame(results).to_csv(out/"inspection_results.csv",index=False)
    save_overlay(case["real"],results,out/"annotated_result.png")
    print(pd.DataFrame(results)[["part_name","status","score","candidate_count"]].to_string(index=False))
    print(f"\nSaved results to {out.resolve()}")
    print("Note: this version does not automatically call ABSENT from detector non-detection.")
if __name__=="__main__": main()
