from pathlib import Path
import argparse, json, sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_inspector.dataset import load_case


def resize_keep(img, max_side=1200):
    h,w=img.shape[:2]; s=min(1.0,max_side/max(h,w))
    if s<1: return cv2.resize(img,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA),s
    return img.copy(),s


def register(cad_rgb, real_rgb, mask, max_side=1200):
    """Estimate CAD-render -> real-photo homography using SIFT + mutual ratio matches."""
    cad_bgr=cv2.cvtColor(cad_rgb,cv2.COLOR_RGB2BGR); real_bgr=cv2.cvtColor(real_rgb,cv2.COLOR_RGB2BGR)
    cad_s,sc=resize_keep(cad_bgr,max_side); real_s,sr=resize_keep(real_bgr,max_side)
    # Ignore the black canvas around the CAD assembly; retain a margin.
    gray_c=cv2.cvtColor(cad_s,cv2.COLOR_BGR2GRAY)
    fg=cv2.threshold(gray_c,8,255,cv2.THRESH_BINARY)[1]
    ys,xs=np.where(fg>0)
    if len(xs)<100: raise RuntimeError('CAD foreground could not be isolated')
    x0,x1=max(0,xs.min()-15),min(fg.shape[1],xs.max()+16)
    y0,y1=max(0,ys.min()-15),min(fg.shape[0],ys.max()+16)
    cad_crop=cad_s[y0:y1,x0:x1]
    # CLAHE helps reduce illumination differences; SIFT remains a geometric matcher.
    def prep(im):
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g)
    g1,g2=prep(cad_crop),prep(real_s)
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=0.015,edgeThreshold=15)
    k1,d1=sift.detectAndCompute(g1,None); k2,d2=sift.detectAndCompute(g2,None)
    if d1 is None or d2 is None: raise RuntimeError('Insufficient local features for registration')
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    m12=matcher.knnMatch(d1,d2,k=2); m21=matcher.knnMatch(d2,d1,k=2)
    good12={a.queryIdx:a.trainIdx for pair in m12 if len(pair)==2 for a,b in [pair] if a.distance<0.72*b.distance}
    good21={a.queryIdx:a.trainIdx for pair in m21 if len(pair)==2 for a,b in [pair] if a.distance<0.72*b.distance}
    mutual=[(q,t) for q,t in good12.items() if good21.get(t)==q]
    if len(mutual)<8: raise RuntimeError(f'Only {len(mutual)} mutual feature matches; registration is unreliable')
    src=np.float32([[k1[q].pt[0]+x0,k1[q].pt[1]+y0] for q,t in mutual]).reshape(-1,1,2)
    dst=np.float32([k2[t].pt for q,t in mutual]).reshape(-1,1,2)
    Hs,inliers=cv2.findHomography(src,dst,cv2.RANSAC,4.0)
    if Hs is None or inliers is None: raise RuntimeError('Homography estimation failed')
    n_in=int(inliers.sum()); ratio=n_in/max(1,len(mutual))
    if n_in<8 or ratio<0.18: raise RuntimeError(f'Weak registration: {n_in} inliers from {len(mutual)} mutual matches')
    # Hs maps downscaled CAD pixels to downscaled real pixels. Convert to full-resolution coordinates.
    S_c=np.array([[sc,0,0],[0,sc,0],[0,0,1]],np.float64)
    S_r_inv=np.array([[1/sr,0,0],[0,1/sr,0],[0,0,1]],np.float64)
    H=S_r_inv @ Hs @ S_c
    return H, {'mutual_matches':len(mutual),'inliers':n_in,'inlier_ratio':round(float(ratio),4),'cad_scale':sc,'real_scale':sr}


def polygon_for_bbox(H, bbox):
    x0,y0,x1,y1=bbox
    pts=np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]])
    return cv2.perspectiveTransform(pts,H)[0]


def region_evidence(real, poly):
    h,w=real.shape[:2]
    x0=max(0,int(np.floor(poly[:,0].min()))); x1=min(w,int(np.ceil(poly[:,0].max())))
    y0=max(0,int(np.floor(poly[:,1].min()))); y1=min(h,int(np.ceil(poly[:,1].max())))
    if x1-x0<3 or y1-y0<3: return None
    patch=real[y0:y1,x0:x1]
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    # Cheap visibility/structure signals only; they are NOT semantic proof of a component.
    contrast=float(gray.std())/64.0
    edges=cv2.Canny(gray,50,130)
    edge=float((edges>0).mean())/0.20
    return {'bbox':[x0,y0,x1,y1], 'contrast':round(min(1.0,contrast),4),
            'edge_density':round(min(1.0,edge),4),
            'visual_support':round(float(0.5*min(1,contrast)+0.5*min(1,edge)),4)}


def main():
    ap=argparse.ArgumentParser(description='CAD-conditioned registration and visual-evidence POC')
    ap.add_argument('--case',required=True); ap.add_argument('--real-image',default=None)
    ap.add_argument('--output',default='outputs/poc'); ap.add_argument('--max-side',type=int,default=1200)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    case=load_case(args.case)
    if case['cad_path'] is None: raise FileNotFoundError('FG_00 CAD render is required')
    if args.real_image:
        img=cv2.imread(args.real_image)
        if img is None: raise FileNotFoundError(args.real_image)
        case['real']=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cad=cv2.cvtColor(cv2.imread(str(case['cad_path'])),cv2.COLOR_BGR2RGB)
    try:
        H,reg=register(cad,case['real'],case['mask'],args.max_side)
        status='REGISTERED'
    except Exception as e:
        (out/'registration_error.txt').write_text(str(e),encoding='utf-8')
        raise RuntimeError(f'Registration failed safely: {e}. No part presence/absence claims were generated.')
    h,w=case['real'].shape[:2]
    overlay=case['real'].copy()
    warped=cv2.warpPerspective(cad,H,(w,h))
    warped_mask=cv2.warpPerspective((case['mask']>0).astype(np.uint8)*255,
        H,(w,h),flags=cv2.INTER_NEAREST)
    # Blend warped CAD only where CAD foreground is present, for reviewer-visible alignment QA.
    m=warped_mask>0
    overlay[m]=cv2.addWeighted(overlay,0.55,warped,0.45,0)[m]
    rows=[]; vis=case['real'].copy()
    for part in case['parts']:
        if not part.get('bbox'):
            rows.append({'part_name':part['name'],'stencil_value':part['stencil_value'],
                         'status':'UNMAPPED_IN_CAD_MASK','visual_support':None,'expected_roi':None,
                         'note':'JSON entry has no corresponding nonzero instance-mask pixels.'})
            continue
        poly=polygon_for_bbox(H,part['bbox'])
        ev=region_evidence(case['real'],poly)
        if ev is None:
            rows.append({'part_name':part['name'],'stencil_value':part['stencil_value'],
                         'status':'OUT_OF_FRAME_OR_INVALID','visual_support':None,'expected_roi':None,
                         'note':'Projected CAD region falls outside the real image or is degenerate.'})
            continue
        # Visual-support is deliberately advisory. It does not identify a bolt/washer by itself.
        label='REVIEW_VISUAL_SUPPORT' if ev['visual_support']>=0.22 else 'REVIEW_LOW_VISIBILITY'
        color=(30,190,70) if label=='REVIEW_VISUAL_SUPPORT' else (0,165,255)
        p=np.int32(poly).reshape(-1,1,2)
        cv2.polylines(vis,[p],True,color,3)
        x,y=ev['bbox'][:2]
        cv2.putText(vis,part['name'][:24],(x,max(20,y-7)),cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1,cv2.LINE_AA)
        rows.append({'part_name':part['name'],'stencil_value':part['stencil_value'],'status':label,
                     'visual_support':ev['visual_support'],'contrast':ev['contrast'],
                     'edge_density':ev['edge_density'],'expected_roi':ev['bbox'],
                     'note':'Advisory local image-structure score only; human verification required. Not a calibrated probability or proof of component identity.'})
    cv2.imwrite(str(out/'registration_overlay.png'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/'projected_component_rois.png'),cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/'warped_cad.png'),cv2.cvtColor(warped,cv2.COLOR_RGB2BGR))
    pd.DataFrame(rows).to_csv(out/'component_review.csv',index=False)
    summary={'registration_status':status,**reg,'component_count':len(rows),
             'mapped_components':sum(r['status'] not in ('UNMAPPED_IN_CAD_MASK','OUT_OF_FRAME_OR_INVALID') for r in rows),
             'warning':'This POC demonstrates CAD-to-photo registration and expected-region transfer. Local edge/contrast support is not a semantic part detector. Do not report PRESENT/ABSENT without validation.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print(pd.DataFrame(rows)[['part_name','status','visual_support']].to_string(index=False))

if __name__=='__main__': main()
