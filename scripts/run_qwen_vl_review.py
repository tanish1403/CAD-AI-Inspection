#!/usr/bin/env python3
"""Qwen3-VL qualitative assembly review. Produces evidence-oriented JSON/text, not ground truth."""
import argparse,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def main():
 p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--cad',required=True); p.add_argument('--json',required=True); p.add_argument('--output',default='outputs/qwen_vl'); p.add_argument('--model',default='Qwen/Qwen3-VL-4B-Instruct'); p.add_argument('--max-pixels',type=int,default=768*768); p.add_argument('--max-new-tokens',type=int,default=900); a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); start=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'; dtype=torch.float16 if device=='cuda' else torch.float32
 real=Image.open(a.image).convert('RGB'); cad=Image.open(a.cad).convert('RGB')
 # Bound image pixel count for T4-class GPUs; preserve aspect ratio.
 def limit_pixels(im, maximum):
  import math
  scale=min(1.0,(maximum/max(1,im.width*im.height))**0.5)
  if scale < 1.0: im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))))
  return im
 real=limit_pixels(real,a.max_pixels); cad=limit_pixels(cad,a.max_pixels)
 # Keep the prompt compact: use the JSON names, not the entire raw annotation document.
 raw=json.loads(Path(a.json).read_text(encoding='utf-8')); entries=next(iter(raw.values())); names=[]
 for e in entries:
  names.append({'name':e.get('Name',e.get('PartName','unknown')),'stencil':e.get('StencilValue',e.get('PixelValue'))})
 prompt=("You are assisting with a visual inspection POC. Compare the CAD render with the real assembly photograph. "
         "List visible similarities and differences at assembly level. Do not claim a specific component is absent unless its location is clearly visible. "
         "Do not invent coordinates, dimensions, part counts, or certainty. If an item cannot be verified from these images, say uncertain. "
         "Return JSON with keys: overall_observations, visible_component_families, possible_differences, occlusion_or_visibility_issues, uncertain_items, limitations. "
         "Expected part names from metadata: "+json.dumps(names[:40])+". This is qualitative assistance, not ground truth.")
 processor=AutoProcessor.from_pretrained(a.model); model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=dtype,device_map='auto' if device=='cuda' else None).eval()
 messages=[{'role':'user','content':[{'type':'image','image':cad},{'type':'text','text':'CAD reference render.'},{'type':'image','image':real},{'type':'text','text':'Real assembly photograph. '+prompt}]}]
 inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 if device=='cuda': inputs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
 with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
 new=generated[0][inputs['input_ids'].shape[-1]:]; answer=processor.decode(new,skip_special_tokens=True)
 (out/'vlm_review.txt').write_text(answer,encoding='utf-8')
 try:
  txt=answer[answer.find('{'):answer.rfind('}')+1]; parsed=json.loads(txt)
 except Exception: parsed={'raw_text':answer,'json_parse':'failed'}
 (out/'vlm_review.json').write_text(json.dumps(parsed,indent=2,ensure_ascii=False),encoding='utf-8')
 summary={'model':a.model,'device':device,'elapsed_seconds':round(time.time()-start,2),'output':'vlm_review.json','warning':'VLM qualitative observations are not verified labels. Never use its standalone ABSENT judgement as ground truth.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
