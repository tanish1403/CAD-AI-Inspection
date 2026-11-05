"""Registration-first CAD/photo POC. Presence is never inferred from texture alone."""
from pathlib import Path
import argparse, json, sys
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_inspector.dataset import load_case


def scale_matrix(s): return np.array([[s,0,0],[0,s,0],[0,0,1]], np.float64)
def resize(im, side):
    h,w=im.shape[:2]; s=min(1.0,side/max(h,w))
    return cv2.resize(im,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA) if s<1 else im.copy(),s

def auto_register(cad, real, max_side=1400, method='auto'):
    """Estimate CAD->photo homography; conservative feature checks, not guaranteed for viewpoint changes."""
    c,sc=resize(cad,max_side); r,sr=resize(real,max_side)
    cg=cv2.cvtColor(c,cv2.COLOR_RGB2GRAY); rg=cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)
    # CAD is rendered on black; crop to non-black foreground to avoid wasting keypoints.
    ys,xs=np.where(cg>8)
    if len(xs)<100: raise RuntimeError('Cannot isolate CAD foreground')
    x0,x1=max(0,xs.min()-12),min(c.shape[1],xs.max()+13); y0,y1=max(0,ys.min()-12),min(c.shape[0],ys.max()+13)
    cg=cg[y0:y1,x0:x1]
    def enhance(g): return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g)
    cg,rg=enhance(cg),enhance(rg)
    attempts=[]
    methods=['sift','akaze','orb'] if method=='auto' else [method]
    for m in methods:
        try:
            if m=='sift': detector=cv2.SIFT_create(nfeatures=10000,contrastThreshold=0.008,edgeThreshold=12); norm=cv2.NORM_L2; ratio=.78
            elif m=='akaze': detector=cv2.AKAZE_create(); norm=cv2.NORM_HAMMING; ratio=.82
            else: detector=cv2.ORB_create(nfeatures=12000,scaleFactor=1.2,nlevels=10,fastThreshold=7); norm=cv2.NORM_HAMMING; ratio=.82
            k1,d1=detector.detectAndCompute(cg,None); k2,d2=detector.detectAndCompute(rg,None)
            if d1 is None or d2 is None: attempts.append(f'{m}: no descriptors'); continue
            matcher=cv2.BFMatcher(norm)
            pairs=matcher.knnMatch(d1,d2,k=2)
            good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<ratio*b.distance]
            if len(good)<8: attempts.append(f'{m}: {len(good)} ratio matches'); continue
            src=np.float32([[k1[a.queryIdx].pt[0]+x0,k1[a.queryIdx].pt[1]+y0] for a in good]).reshape(-1,1,2)
            dst=np.float32([k2[a.trainIdx].pt for a in good]).reshape(-1,1,2)
            Hs,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
            if Hs is None or mask is None: attempts.append(f'{m}: homography failed'); continue
            nin=int(mask.sum()); ir=nin/len(good)
            attempts.append(f'{m}: {nin}/{len(good)} inliers ({ir:.2f})')
            if nin<8 or ir<.12: continue
            H=scale_matrix(1/sr) @ Hs @ scale_matrix(sc)
            # Reject projectively unstable mappings and implausible foreground coverage.
            if not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:
                attempts.append(f'{m}: non-finite/singular transform'); continue
            corners=np.float32([[[x0/sc,y0/sc],[(x1-1)/sc,y0/sc],[(x1-1)/sc,(y1-1)/sc],[x0/sc,(y1-1)/sc]]])
            proj=cv2.perspectiveTransform(corners,H)[0]
            if not np.isfinite(proj).all(): attempts.append(f'{m}: invalid projected corners'); continue
            pw=float(proj[:,0].max()-proj[:,0].min()); ph=float(proj[:,1].max()-proj[:,1].min())
            coverage=(pw*ph)/(real.shape[1]*real.shape[0])
            if pw<20 or ph<20 or coverage<0.005 or coverage>0.90 or pw>real.shape[1]*2.5 or ph>real.shape[0]*2.5:
                attempts.append(f'{m}: implausible projected CAD bounds ({pw:.0f}x{ph:.0f}, coverage={coverage:.3f})'); continue
            return H, {'mode':'automatic','method':m,'matches':len(good),'inliers':nin,'inlier_ratio':round(ir,4),'projected_bounds_fraction':round(coverage,4),'attempts':attempts}
        except Exception as e: attempts.append(f'{m}: {e}')
    raise RuntimeError('Automatic registration failed. '+ ' | '.join(attempts)+' Use the manual point tool.')

def manual_register(points_json):
    d=json.loads(Path(points_json).read_text(encoding='utf-8'))
    src=np.asarray(d['cad_points'],np.float32); dst=np.asarray(d['real_points'],np.float32)
    if src.shape!=dst.shape or src.ndim!=2 or src.shape[1]!=2 or len(src)<4: raise ValueError('Need >=4 paired [x,y] points')
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    if H is None: raise RuntimeError('Manual homography failed')
    return H, {'mode':'manual','points':len(src),'inliers':int(mask.sum()) if mask is not None else len(src),'inlier_ratio':float(mask.mean()) if mask is not None else 1.0}

def project_bbox(H,b):
    x0,y0,x1,y1=b; pts=np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]); return cv2.perspectiveTransform(pts,H)[0]
def clipped_bbox(poly,w,h):
    x0=max(0,int(np.floor(poly[:,0].min()))); y0=max(0,int(np.floor(poly[:,1].min())))
    x1=min(w,int(np.ceil(poly[:,0].max()))); y1=min(h,int(np.ceil(poly[:,1].max())))
    return [x0,y0,x1,y1] if x1-x0>=3 and y1-y0>=3 else None

def local_signals(img,bbox):
    x0,y0,x1,y1=bbox; patch=img[y0:y1,x0:x1]
    if patch.size==0:return {}
    g=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(g,45,120)
    return {'contrast_std':round(float(g.std()),3),'edge_fraction':round(float((edges>0).mean()),4)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',required=True); ap.add_argument('--real-image'); ap.add_argument('--output',default='outputs/registration_poc'); ap.add_argument('--manual-points'); ap.add_argument('--method',choices=['auto','sift','akaze','orb'],default='auto'); ap.add_argument('--max-side',type=int,default=1400)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    case=load_case(a.case)
    if a.real_image:
        bgr=cv2.imread(a.real_image)
        if bgr is None: raise FileNotFoundError(a.real_image)
        case['real']=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    cad_bgr=cv2.imread(str(case['cad_path'])); cad=cv2.cvtColor(cad_bgr,cv2.COLOR_BGR2RGB)
    if a.manual_points: H,reg=manual_register(a.manual_points)
    else: H,reg=auto_register(cad,case['real'],a.max_side,a.method)
    real=case['real']; h,w=real.shape[:2]
    # Scale mask to CAD render before warping. IDs remain integer-valued.
    mask=case['mask']; mask=cv2.resize(mask,(cad.shape[1],cad.shape[0]),interpolation=cv2.INTER_NEAREST) if mask.shape!=cad.shape[:2] else mask
    warped=cv2.warpPerspective(cad,H,(w,h)); wm=cv2.warpPerspective(mask,H,(w,h),flags=cv2.INTER_NEAREST)
    blend=real.copy(); foreground=wm>0
    blend[foreground]=np.clip(.55*real[foreground]+.45*warped[foreground],0,255).astype(np.uint8)
    ann=real.copy(); rows=[]
    for p in case['parts']:
        base={'part_name':p['name'],'stencil_value':p['stencil_value'],'cad_mask_pixels':p['mask_pixels']}
        if p['bbox'] is None:
            rows.append({**base,'status':'NO_CAD_REFERENCE','evidence':'No nonzero pixels for this stencil ID; this does not mean absent in photo.'}); continue
        # Part bbox coordinates originate in mask space; convert to CAD-render pixels before projection.
        mh,mw=case['mask'].shape[:2]; ch,cw=cad.shape[:2]
        x0,y0,x1,y1=p['bbox']; cad_bbox=[x0*cw/mw,y0*ch/mh,x1*cw/mw,y1*ch/mh]
        poly=project_bbox(H,cad_bbox); bb=clipped_bbox(poly,w,h)
        if bb is None:
            rows.append({**base,'status':'PROJECTED_OUT_OF_FRAME','evidence':'Projected CAD region is outside image or degenerate.'}); continue
        sig=local_signals(real,bb); x0,y0,x1,y1=bb
        cv2.polylines(ann,[np.int32(poly).reshape(-1,1,2)],True,(0,190,255),2)
        cv2.putText(ann,p['name'][:25],(x0,max(18,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,190,255),1,cv2.LINE_AA)
        rows.append({**base,'status':'REVIEW_REQUIRED','expected_roi':bb,**sig,'evidence':'ROI projected from CAD. Local contrast/edges are diagnostic only, not proof of component identity.'})
    cv2.imwrite(str(out/'registration_overlay.png'),cv2.cvtColor(blend,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(out/'projected_component_rois.png'),cv2.cvtColor(ann,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(out/'warped_cad.png'),cv2.cvtColor(warped,cv2.COLOR_RGB2BGR))
    pd.DataFrame(rows).to_csv(out/'component_review.csv',index=False)
    summary={'registration':reg,'components':len(rows),'projected_rois':sum(x['status']=='REVIEW_REQUIRED' for x in rows),'no_cad_reference':sum(x['status']=='NO_CAD_REFERENCE' for x in rows),'decision_policy':'No PRESENT/ABSENT labels are generated by this registration-only stage. Verify projected ROIs with a detector or human reviewer.','homography':H.tolist()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
