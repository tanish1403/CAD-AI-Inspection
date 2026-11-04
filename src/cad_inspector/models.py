from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModel

class FoundationModels:
    """Pretrained-only Grounding DINO + DINOv2 inference."""
    def __init__(self, config, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Select --device cpu or enable Kaggle GPU.")
        self.device=device
        root=Path(__file__).resolve().parents[2]
        self.gd_id=self._resolve(root,config["models"]["grounding_dino_id"],"grounding-dino")
        self.dino_id=self._resolve(root,config["models"]["dinov2_id"],"dinov2")
        self.det_processor=self.det_model=self.feat_processor=self.feat_model=None

    @staticmethod
    def _resolve(root, model_id, folder):
        local=root/"models"/"huggingface"/folder
        return str(local) if (local/"config.json").exists() else model_id

    def load_detector(self):
        if self.det_model is None:
            self.det_processor=AutoProcessor.from_pretrained(self.gd_id)
            self.det_model=AutoModelForZeroShotObjectDetection.from_pretrained(self.gd_id).to(self.device).eval()
        return self.det_processor,self.det_model

    def load_features(self):
        if self.feat_model is None:
            self.feat_processor=AutoImageProcessor.from_pretrained(self.dino_id)
            self.feat_model=AutoModel.from_pretrained(self.dino_id).to(self.device).eval()
        return self.feat_processor,self.feat_model

    @torch.inference_mode()
    def detect(self, image_rgb, prompts, box_threshold=0.22, text_threshold=0.20):
        processor,model=self.load_detector()
        image=Image.fromarray(image_rgb)
        # Transformers Grounding DINO expects a batch of text-label lists.
        labels=[prompts]
        inputs=processor(images=image,text=labels,return_tensors="pt").to(self.device)
        outputs=model(**inputs)
        result=processor.post_process_grounded_object_detection(
            outputs,inputs.input_ids,threshold=box_threshold,
            text_threshold=text_threshold,target_sizes=[(image.height,image.width)])[0]
        names=result.get("text_labels",result.get("labels",[]))
        raw=[]
        for box,score,label in zip(result["boxes"],result["scores"],names):
            coords=[float(v) for v in box.tolist()]
            x0,y0,x1,y1=coords
            raw.append({"bbox":coords,"detector_score":float(score.item()),"label":str(label),
                        "xywh":[x0,y0,max(1.0,x1-x0),max(1.0,y1-y0)]})
        if not raw:return []
        keep=cv2.dnn.NMSBoxes([r["xywh"] for r in raw],
                              [r["detector_score"] for r in raw],
                              score_threshold=box_threshold,nms_threshold=0.55)
        ids=np.asarray(keep).reshape(-1).tolist() if len(keep) else []
        return [{k:v for k,v in raw[i].items() if k!="xywh"} for i in ids]

    @torch.inference_mode()
    def embedding(self, crop_rgb):
        processor,model=self.load_features()
        if crop_rgb is None or crop_rgb.size==0: return None
        inputs=processor(images=Image.fromarray(crop_rgb),return_tensors="pt").to(self.device)
        out=model(**inputs)
        return torch.nn.functional.normalize(out.last_hidden_state[:,0,:],dim=-1).squeeze(0)

    def cosine(self,a,b):
        if a is None or b is None:return 0.0
        return float(torch.nn.functional.cosine_similarity(a[None,:],b[None,:]).item())
