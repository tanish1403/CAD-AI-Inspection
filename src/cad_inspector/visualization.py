import cv2
import numpy as np
from pathlib import Path

COLORS={"PRESENT":(40,190,70),"ABSENT":(40,40,230),"UNCERTAIN":(0,190,240)}
def save_overlay(image_rgb, results, path):
    canvas=cv2.cvtColor(image_rgb.copy(),cv2.COLOR_RGB2BGR)
    for r in results:
        color=COLORS.get(r["status"],(255,255,255))
        if r.get("bbox"):
            x0,y0,x1,y1=[int(v) for v in r["bbox"]]
            cv2.rectangle(canvas,(x0,y0),(x1,y1),color,3)
            label=f'{r["part_name"]}: {r["status"]} ({r["score"]:.2f})'
            cv2.putText(canvas,label,(max(5,x0),max(22,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),canvas)
