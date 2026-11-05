"""Desktop helper: click corresponding CAD and real-photo points in the same order."""
import argparse,json
from pathlib import Path
import cv2

def collect(path,title,n=4):
    img=cv2.imread(str(path))
    if img is None: raise FileNotFoundError(path)
    pts=[]; view=img.copy()
    def cb(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(pts)<n:
            pts.append([int(x),int(y)]); cv2.circle(view,(x,y),9,(0,0,255),-1); cv2.putText(view,str(len(pts)),(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2); cv2.imshow(title,view)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL); cv2.imshow(title,view); cv2.setMouseCallback(title,cb)
    print(f'{title}: click {n} corresponding points in order. Press ESC to cancel.')
    while len(pts)<n:
        if cv2.waitKey(30)==27: raise RuntimeError('Cancelled')
    cv2.destroyWindow(title); return pts

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cad',required=True); p.add_argument('--real',required=True); p.add_argument('--output',default='configs/manual_alignment.json'); p.add_argument('--points',type=int,default=4); a=p.parse_args()
    print('Choose widely spaced, distinctive points visible in BOTH images. Use the same physical corners/features and click in the same order.')
    c=collect(a.cad,'CAD image',a.points); r=collect(a.real,'Real photograph',a.points)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'cad_points':c,'real_points':r,'note':'Coordinates are full-resolution pixel [x,y], ordered correspondences.'},indent=2))
    print('Saved',out)
if __name__=='__main__': main()
