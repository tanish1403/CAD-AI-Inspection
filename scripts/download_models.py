#!/usr/bin/env python
import argparse, os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
os.environ.setdefault("HF_HOME",str(ROOT/"models"/"huggingface_cache"))
MODELS={"grounding-dino":"IDEA-Research/grounding-dino-tiny","dinov2":"facebook/dinov2-small"}
def main():
    p=argparse.ArgumentParser(description="Download pretrained Option-A models")
    p.add_argument("--models",nargs="+",choices=list(MODELS)+["all"],default=["grounding-dino","dinov2"])
    a=p.parse_args(); selected=list(MODELS) if "all" in a.models else a.models
    target=ROOT/"models"/"huggingface"; target.mkdir(parents=True,exist_ok=True)
    token=os.getenv("HF_TOKEN") or None
    for key in selected:
        print(f"Downloading {MODELS[key]} ...")
        saved=snapshot_download(repo_id=MODELS[key],token=token,local_dir=target/key)
        print(f"Cached at {saved}")
if __name__=="__main__":main()
