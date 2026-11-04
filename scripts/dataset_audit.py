#!/usr/bin/env python
"""Dataset-only audit: no model weights or network required."""
import json, csv, argparse
from pathlib import Path
import cv2, numpy as np

def read_rgb(p):
    im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if im is None:return None
    if im.ndim==2:return cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(im,cv2.COLOR_BGRA2RGB if im.shape[2]==4 else cv2.COLOR_BGR2RGB)

def audit_case(folder):
    folder=Path(folder)
    js=next(folder.glob("*.json"))
    obj=json.loads(js.read_text(encoding="utf-8"))
    parts=next(iter(obj.values()))
    maskp=next(folder.glob("Mask_*"))
    m=cv2.imread(str(maskp),cv2.IMREAD_COLOR)
    # Verified on supplied BMPs: stencil labels are encoded in RGB red channel.
    red=m[:,:,2]
    unique,counts=np.unique(red,return_counts=True)
    counts=dict(zip(unique.tolist(),counts.tolist()))
    rows=[]
    for p in parts:
        val=int(p.get("StencilValue",p.get("PixelValue",-1)))
        rows.append({"part_name":p.get("Name",p.get("PartName",str(val))),
                     "stencil_value":val,"mask_pixels":counts.get(val,0),
                     "mask_present":counts.get(val,0)>0})
    imgs=[]
    for p in sorted(folder.glob("BG_*.png")):
        im=read_rgb(p)
        if im is not None:imgs.append({"file":p.name,"width":im.shape[1],"height":im.shape[0],"channels":im.shape[2]})
    cad=next(iter(folder.glob("FG_*.png")),None)
    cadim=read_rgb(cad) if cad else None
    return {"folder":str(folder),"json_parts":len(parts),"mask_shape":list(red.shape),
            "distinct_mask_values":len(unique),"nonzero_mask_pixels":int(np.count_nonzero(red)),
            "parts_with_mask_pixels":sum(x["mask_present"] for x in rows),
            "parts_without_mask_pixels":sum(not x["mask_present"] for x in rows),
            "cad_shape":list(cadim.shape) if cadim is not None else None,
            "real_images":imgs,"parts":rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--output",default="outputs/dataset_audit")
    a=ap.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cases=[root]+[p for p in root.iterdir() if p.is_dir() and any(p.glob("*.json"))]
    results=[audit_case(c) for c in cases if any(c.glob("*.json")) and any(c.glob("Mask_*"))]
    (out/"dataset_audit.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    with (out/"part_mask_inventory.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["case","part_name","stencil_value","mask_pixels","mask_present"]);w.writeheader()
        for r in results:
            for p in r["parts"]:w.writerow({"case":Path(r["folder"]).name,**p})
    lines=["# Supplied dataset audit","",
           "> This is an actual dataset-structure and mask-integrity audit. It is not a pretrained-model inference result.",""]
    for r in results:
        lines += [f"## {Path(r['folder']).name}",f"- JSON components: {r['json_parts']}",
                  f"- Mask dimensions: {r['mask_shape'][1]} × {r['mask_shape'][0]}",
                  f"- Distinct red-channel mask values: {r['distinct_mask_values']}",
                  f"- Components with nonzero mask pixels: {r['parts_with_mask_pixels']}",
                  f"- Components with zero mask pixels: {r['parts_without_mask_pixels']}",
                  f"- CAD dimensions: {r['cad_shape']}","- Real captures:"]
        lines += [f"  - {x['file']}: {x['width']} × {x['height']} ({x['channels']} channels)" for x in r["real_images"]]
        lines.append("")
    (out/"DATASET_AUDIT.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
if __name__=="__main__":main()
