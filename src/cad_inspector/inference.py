from pathlib import Path
import cv2
from .dataset import extract_cad_reference

FAMILY_PROMPTS={
 "bolt":["a metal bolt","a hex head bolt","a screw fastener"],
 "washer":["a metal washer","a flat washer","a ring-shaped metal fastener"],
 "nut":["a hex nut","a metal nut","a threaded nut"],
 "cover":["a metal side cover","a metal housing cover","a machine casing"],
 "box":["a metal mechanical housing","a rectangular metal enclosure"],
 "unknown":["a mechanical metal component","a metal machine part"]
}
def family_for(name):
    n=name.lower()
    for family in ["washer","bolt","nut","cover","box"]:
        if family in n:return family
    return "unknown"

def crop_box(image,box,pad=5):
    h,w=image.shape[:2]
    x0,y0,x1,y1=[int(round(v)) for v in box]
    x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(w,x1+pad);y1=min(h,y1+pad)
    return image[y0:y1,x0:x1] if x1>x0 and y1>y0 else None

def overlaps(box,roi):
    x0,y0,x1,y1=box; a,b,c,d=roi
    return max(0,min(x1,c)-max(x0,a))*max(0,min(y1,d)-max(y0,b))>0

def inspect_case(case,models,config,output_dir,roi_map=None):
    out=Path(output_dir);debug=out/"debug";debug.mkdir(parents=True,exist_ok=True)
    real=case["real"];inf=config["inference"]
    maxside=inf.get("max_image_side",1280)
    scale=min(1.0,maxside/max(real.shape[:2]))
    small=cv2.resize(real,(int(real.shape[1]*scale),int(real.shape[0]*scale)),interpolation=cv2.INTER_AREA) if scale<1 else real
    families=sorted({family_for(p["name"]) for p in case["parts"]})
    family_detections={}
    for fam in families:
        dets=models.detect(small,FAMILY_PROMPTS[fam],
            box_threshold=inf["grounding_box_threshold"],
            text_threshold=inf["grounding_text_threshold"])
        for d in dets:d["bbox"]=[v/scale for v in d["bbox"]]
        family_detections[fam]=dets

    refs={}
    for part in case["parts"]:
        ref=extract_cad_reference(case,part,debug/"cad_reference_crops")
        refs[part["stencil_value"]]=models.embedding(ref) if ref is not None else None

    # Cache embeddings for each real candidate once.
    candidate_embs={}
    for fam,dets in family_detections.items():
        for idx,d in enumerate(dets):
            crop=crop_box(real,d["bbox"])
            candidate_embs[(fam,idx)]=models.embedding(crop) if crop is not None else None

    # Build part-candidate score edges and greedily assign each detection to at most one expected instance.
    assignments={}
    for fam in families:
        fam_parts=[p for p in case["parts"] if family_for(p["name"])==fam]
        dets=family_detections[fam]
        edges=[]
        for p in fam_parts:
            ref_emb=refs[p["stencil_value"]]
            roi=(roi_map or {}).get(p["name"])
            for idx,d in enumerate(dets):
                if roi and roi.get("bbox") and not overlaps(d["bbox"],roi["bbox"]):continue
                emb=candidate_embs[(fam,idx)]
                if ref_emb is None or emb is None:continue
                sim=models.cosine(ref_emb,emb)
                score=0.35*d["detector_score"]+0.65*max(0.0,sim)
                edges.append((score,p["stencil_value"],idx,sim))
        used_parts=set();used_dets=set()
        for score,pid,idx,sim in sorted(edges,reverse=True):
            if pid in used_parts or idx in used_dets:continue
            if score<inf["present_threshold"]:continue
            assignments[pid]={"candidate":dets[idx],"similarity":sim,"score":score,"candidate_index":idx}
            used_parts.add(pid);used_dets.add(idx)

    results=[]
    for p in case["parts"]:
        fam=family_for(p["name"]);roi=(roi_map or {}).get(p["name"])
        assigned=assignments.get(p["stencil_value"])
        # If an expected visible ROI was specified, require candidate overlap for the part assignment.
        if assigned:
            status="PRESENT";best=assigned["candidate"];sim=assigned["similarity"];score=assigned["score"]
        else:
            best=None;sim=None;score=None
            if roi and roi.get("bbox") and roi.get("visible") is True:
                nearby=[d for d in family_detections[fam] if overlaps(d["bbox"],roi["bbox"])]
                status="ABSENT" if not nearby else "UNCERTAIN"
            else:status="UNCERTAIN"
        results.append({
            "part_name":p["name"],"stencil_value":p["stencil_value"],
            "mask_pixels":p["mask_pixels"],"reference_available":refs[p["stencil_value"]] is not None,
            "family_prompt":fam,"status":status,
            "similarity_score":round(sim,4) if sim is not None else None,
            "detector_score":round(best["detector_score"],4) if best else None,
            "combined_rank_score":round(score,4) if score is not None else None,
            "bbox":best["bbox"] if best else None,
            "candidate_count":len(family_detections[fam]),
            "note":"One-to-one candidate assignment. ABSENT requires an explicit visible expected ROI; otherwise no match is UNCERTAIN."
        })
    return results
