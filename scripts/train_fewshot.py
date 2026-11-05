#!/usr/bin/env python
"""Train a lightweight classifier over frozen DINOv2 CLS embeddings."""
import argparse,sys
from pathlib import Path
import numpy as np, joblib, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor,AutoModel

def encode(folder,processor,model,device):
    ds=ImageFolder(folder)
    X=[]; y=[]
    for path,label in ds.samples:
        im=Image.open(path).convert("RGB")
        batch=processor(images=im,return_tensors="pt").to(device)
        with torch.inference_mode(): feat=model(**batch).last_hidden_state[:,0,:]
        feat=torch.nn.functional.normalize(feat,dim=-1)
        X.append(feat.cpu().numpy()[0]); y.append(label)
    return np.asarray(X),np.asarray(y),ds.classes

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--val",required=True)
    p.add_argument("--output",default="models/fewshot_classifier.joblib")
    p.add_argument("--model",default="facebook/dinov2-small")
    a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    proc=AutoImageProcessor.from_pretrained(a.model); model=AutoModel.from_pretrained(a.model).to(device).eval()
    X,y,classes=encode(a.train,proc,model,device)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced")
    clf.fit(X,y)
    Xv,yv,valclasses=encode(a.val,proc,model,device)
    print(classification_report(yv,clf.predict(Xv),target_names=classes,zero_division=0))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"classifier":clf,"classes":classes,"model_id":a.model},out)
    print("Saved:",out)
if __name__=="__main__": main()
