from pathlib import Path
import numpy as np
import cv2

def crop_box(image, box, pad=4):
    h,w=image.shape[:2]
    x0,y0,x1,y1=[int(round(v)) for v in box]
    x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(w,x1+pad); y1=min(h,y1+pad)
    if x1<=x0 or y1<=y0: return None
    return image[y0:y1,x0:x1]

def build_prompts(part, aliases):
    phrases=[part["name"].replace("_"," ").replace("-"," ")]
    phrases.extend(aliases.get(part["name"],[]))
    # Grounding DINO uses dot-separated phrase prompts.
    return list(dict.fromkeys([p.strip().lower() for p in phrases if p.strip()]))

def inspect_case(case, models, config, output_dir, roi_map=None):
    output_dir=Path(output_dir); debug=output_dir/"debug"; debug.mkdir(parents=True,exist_ok=True)
    real=case["real"]
    inf=config["inference"]; aliases=config.get("part_aliases",{})
    maxside=inf.get("max_image_side",1280)
    scale=min(1.0,maxside/max(real.shape[:2]))
    if scale<1:
        detector_img=cv2.resize(real,(int(real.shape[1]*scale),int(real.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    else: detector_img=real
    results=[]
    for part in case["parts"]:
        ref=None
        from .dataset import extract_cad_reference
        ref=extract_cad_reference(case,part,debug/"cad_refs")
        ref_emb=models.embedding(ref) if ref is not None else None
        prompts=build_prompts(part,aliases)
        try:
            candidates=models.detect(detector_img,prompts,
                box_threshold=inf["grounding_box_threshold"],
                text_threshold=inf["grounding_text_threshold"])
        except Exception as exc:
            candidates=[]; error=str(exc)
        ranked=[]
        for cand in candidates[:inf.get("candidate_limit_per_part",15)]:
            box=[v/scale for v in cand["bbox"]]
            crop=crop_box(real,box)
            emb=models.embedding(crop) if crop is not None else None
            sim=models.cosine(ref_emb,emb) if ref_emb is not None and emb is not None else 0.0
            # Guard against invalid model outputs; NaN must never enter ranking or CSV output.
            if not np.isfinite(sim):
                sim=0.0
            det=float(cand.get("detector_score",0.0))
            if not np.isfinite(det):
                det=0.0
            # Similarity is evidence, not a calibrated probability.
            combined=0.45*det+0.55*max(0.0,float(sim))
            ranked.append({**cand,"bbox":box,"similarity":sim,"combined_score":combined})
        ranked.sort(key=lambda x: float(x.get("combined_score",0.0)) if np.isfinite(x.get("combined_score",0.0)) else -1.0,reverse=True)
        # If the operator or a validated registration stage supplied an expected ROI,
        # only candidates overlapping that visible ROI count as evidence for this part.
        roi_entry=(roi_map or {}).get(part["name"])
        if roi_entry and roi_entry.get("bbox"):
            rx0,ry0,rx1,ry1=roi_entry["bbox"]
            def overlaps(b):
                x0,y0,x1,y1=b
                ix=max(0,min(x1,rx1)-max(x0,rx0))
                iy=max(0,min(y1,ry1)-max(y0,ry0))
                return (ix*iy)>0
            roi_ranked=[c for c in ranked if overlaps(c["bbox"])]
        else:
            roi_ranked=ranked
        best=roi_ranked[0] if roi_ranked else None
        score=best["combined_score"] if best else 0.0
        if best and score>=inf["present_threshold"]:
            status="PRESENT"
        elif roi_entry and roi_entry.get("visible") is True and best is None:
            status="ABSENT"
        else:
            status="UNCERTAIN"
        # No ROI or unconfirmed visibility: detector non-detection is not proof of absence.
        results.append({"part_name":part["name"],"stencil_value":part["stencil_value"],
                        "status":status,"score":round(score,4),
                        "detector_score":round(best["detector_score"],4) if best else None,
                        "embedding_similarity":round(best["similarity"],4) if best else None,
                        "bbox":best["bbox"] if best else None,
                        "candidate_label":best["label"] if best else None,
                        "candidate_count":len(ranked),"expected_roi":roi_entry.get("bbox") if roi_entry else None,
                        "note":"ABSENT requires an explicitly supplied ROI with visible=true; otherwise detector non-detection remains UNCERTAIN."})
    return results
