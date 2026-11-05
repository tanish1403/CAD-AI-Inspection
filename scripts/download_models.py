#!/usr/bin/env python
import argparse, os, sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
os.environ.setdefault("HF_HOME",str(ROOT/"models"/"huggingface_cache"))
MODELS={
 "grounding-dino":"IDEA-Research/grounding-dino-tiny",
 "dinov2":"facebook/dinov2-small",
 "sam2":"facebook/sam2-hiera-small",
}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",nargs="+",choices=list(MODELS)+["all"],default=["grounding-dino","dinov2"])
    args=p.parse_args()
    selected=list(MODELS) if "all" in args.models else args.models
    token=os.getenv("HF_TOKEN") or None
    target=ROOT/"models"/"huggingface"
    target.mkdir(parents=True,exist_ok=True)
    for key in selected:
        repo=MODELS[key]
        print(f"Downloading {key}: {repo}")
        try:
            path=snapshot_download(repo_id=repo,token=token,local_dir=target/key)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"FAILED {repo}: {e}",file=sys.stderr)
            raise
if __name__=="__main__": main()
