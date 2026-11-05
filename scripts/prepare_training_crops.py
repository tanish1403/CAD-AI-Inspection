#!/usr/bin/env python
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cad_inspector.dataset import load_case,extract_cad_reference
import cv2
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",required=True); p.add_argument("--output",default="data/training/cad_reference_crops")
    a=p.parse_args(); case=load_case(a.case); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    count=0
    for part in case["parts"]:
        crop=extract_cad_reference(case,part)
        if crop is None: continue
        folder=out/part["name"]; folder.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(folder/f'{part["stencil_value"]:03d}.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)); count+=1
    print(f"Wrote {count} CAD reference crops to {out}")
    print("These are CAD-domain references, not real-image labels. Collect labelled real crops separately for few-shot training.")
if __name__=="__main__": main()
