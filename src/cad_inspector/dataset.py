import json
from pathlib import Path
import cv2
import numpy as np

def find_one(folder, prefixes, suffixes):
    for p in Path(folder).iterdir():
        if p.is_file() and p.name.lower().endswith(tuple(s.lower() for s in suffixes)) and any(p.name.startswith(x) for x in prefixes):
            return p
    return None

def load_case(case_dir):
    folder=Path(case_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Case directory not found: {folder}")
    real_path=find_one(folder,["BG_00"],[".png",".jpg",".jpeg"])
    cad_path=find_one(folder,["FG_00"],[".png",".jpg",".jpeg"])
    mask_path=find_one(folder,["Mask_"],[".bmp",".png",".tif",".tiff"])
    json_files=list(folder.glob("*.json"))
    if real_path is None or mask_path is None or not json_files:
        raise FileNotFoundError("Expected BG_00 image, Mask_* image and a .json part map in case folder")
    real=cv2.cvtColor(cv2.imread(str(real_path)),cv2.COLOR_BGR2RGB)
    mask_bgr=cv2.imread(str(mask_path),cv2.IMREAD_COLOR)
    if real is None or mask_bgr is None:
        raise ValueError("Could not decode real image or mask")
    # Dataset convention: IDs are encoded in the RED channel of the BMP.
    mask=mask_bgr[:,:,2]
    mapping=json.loads(json_files[0].read_text(encoding="utf-8"))
    entries=next(iter(mapping.values()))
    parts=[]
    for item in entries:
        value=int(item.get("StencilValue",item.get("PixelValue",-1)))
        name=str(item.get("Name",item.get("PartName",f"part_{value}")))
        if value>=0:
            ys,xs=np.where(mask==value)
            if len(xs):
                x0,x1=int(xs.min()),int(xs.max()+1)
                y0,y1=int(ys.min()),int(ys.max()+1)
                parts.append({"name":name,"stencil_value":value,"bbox":[x0,y0,x1,y1],"mask_pixels":int(len(xs))})
            else:
                parts.append({"name":name,"stencil_value":value,"bbox":None,"mask_pixels":0})
    return {"folder":folder,"real_path":real_path,"cad_path":cad_path,"mask_path":mask_path,
            "json_path":json_files[0],"real":real,"mask":mask,"parts":parts}

def extract_cad_reference(case, part, out_dir=None):
    if case["cad_path"] is None or part["bbox"] is None:
        return None
    cad=cv2.cvtColor(cv2.imread(str(case["cad_path"])),cv2.COLOR_BGR2RGB)
    mask=case["mask"]
    if mask.shape[:2] != cad.shape[:2]:
        mask=cv2.resize(mask,(cad.shape[1],cad.shape[0]),interpolation=cv2.INTER_NEAREST)
    x0,y0,x1,y1=part["bbox"]
    # Scale mask bbox if mask and CAD render sizes differ.
    mh,mw=case["mask"].shape[:2]; ch,cw=cad.shape[:2]
    x0=int(x0*cw/mw); x1=int(x1*cw/mw); y0=int(y0*ch/mh); y1=int(y1*ch/mh)
    part_mask=(mask==part["stencil_value"])[y0:y1,x0:x1]
    crop=cad[y0:y1,x0:x1].copy()
    if crop.size==0: return None
    crop[~part_mask]=0
    if out_dir:
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(Path(out_dir)/f"{part['stencil_value']:03d}_{part['name']}.png"),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
    return crop
