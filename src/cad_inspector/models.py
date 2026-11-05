import os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModel

def resolve_device(requested="auto"):
    if requested=="auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested

class FoundationModels:
    def __init__(self, config, device="auto"):
        self.device=resolve_device(device)
        root=Path(__file__).resolve().parents[2]
        self.gd_id=self._local_or_remote(root,config["models"]["grounding_dino_id"],"grounding-dino")
        self.dino_id=self._local_or_remote(root,config["models"]["dinov2_id"],"dinov2")
        self.det_processor=None; self.det_model=None
        self.feat_processor=None; self.feat_model=None

    @staticmethod
    def _local_or_remote(root, model_id, folder_name):
        local=root/"models"/"huggingface"/folder_name
        # Use a downloaded snapshot when it contains model metadata; otherwise HF downloads it.
        if (local/"config.json").exists():
            return str(local)
        return model_id

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
        # HF Grounding DINO expects a batch containing one list of text labels.
        inputs=processor(images=image,text=[prompts],return_tensors="pt").to(self.device)
        outputs=model(**inputs)
        results=processor.post_process_grounded_object_detection(
            outputs,inputs.input_ids,threshold=box_threshold,text_threshold=text_threshold,
            target_sizes=[(image.height,image.width)])
        r=results[0]
        candidates=[]
        for box,score,label in zip(r["boxes"],r["scores"],r["labels"]):
            candidates.append({"bbox":[float(v) for v in box.tolist()],
                               "detector_score":float(score.item()),"label":str(label)})
        return candidates

    @torch.inference_mode()
    def embedding(self, crop_rgb):
        processor,model=self.load_features()
        if crop_rgb is None or crop_rgb.size==0: return None
        image=Image.fromarray(crop_rgb)
        inputs=processor(images=image,return_tensors="pt").to(self.device)
        out=model(**inputs)
        # CLS token is a compact global representation.
        emb=out.last_hidden_state[:,0,:]
        return torch.nn.functional.normalize(emb,dim=-1).squeeze(0)

    def cosine(self, a, b):
        if a is None or b is None: return 0.0
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)).item())
