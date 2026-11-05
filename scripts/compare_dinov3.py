"""DINOv3 dense feature correspondence diagnostic. It does not declare part presence."""
import argparse,json
from pathlib import Path
import cv2,numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor,AutoModel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cad',required=True); ap.add_argument('--real',required=True); ap.add_argument('--output',default='outputs/dinov3'); ap.add_argument('--model',default='facebook/dinov3-vits16-pretrain-lvd1689m'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    proc=AutoImageProcessor.from_pretrained(a.model); model=AutoModel.from_pretrained(a.model).to(a.device).eval()
    def encode(path):
        image=Image.open(path).convert('RGB'); inp=proc(images=image,return_tensors='pt').to(a.device)
        with torch.inference_mode(): tokens=model(**inp).last_hidden_state
        nreg=int(getattr(model.config,'num_register_tokens',0)); patches=tokens[:,1+nreg:,:]
        size=proc.size; ph=size.get('height',224) if isinstance(size,dict) else 224; pw=size.get('width',224) if isinstance(size,dict) else ph
        ps=int(getattr(model.config,'patch_size',16)); gh,gw=ph//ps,pw//ps
        if patches.shape[1]!=gh*gw: gh=gw=int(round(patches.shape[1]**.5))
        f=torch.nn.functional.normalize(patches[0].float(),dim=-1).reshape(gh,gw,-1)
        return f.cpu().numpy()
    fc=encode(a.cad); fr=encode(a.real)
    c=torch.from_numpy(fc.reshape(-1,fc.shape[-1])); r=torch.from_numpy(fr.reshape(-1,fr.shape[-1])); sim=c@r.T
    best=sim.max(dim=1).values.numpy(); summary={'model':a.model,'cad_patch_grid':list(fc.shape[:2]),'real_patch_grid':list(fr.shape[:2]),'mean_best_patch_cosine':float(best.mean()),'median_best_patch_cosine':float(np.median(best)),'warning':'Unregistered patch retrieval diagnostic. Not a calibrated match probability and not evidence of component presence.'}
    heat=cv2.resize(best.reshape(fc.shape[:2]),(512,512),interpolation=cv2.INTER_CUBIC); heat=cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); cv2.imwrite(str(out/'dinov3_patch_similarity.png'),cv2.applyColorMap(heat,cv2.COLORMAP_TURBO))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
