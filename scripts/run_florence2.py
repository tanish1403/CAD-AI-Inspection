#!/usr/bin/env python3
"""Florence-2 large-ft open-vocabulary/phrase grounding diagnostic."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForCausalLM

def main():
 p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--output',default='outputs/florence2'); p.add_argument('--model',default='microsoft/Florence-2-large-ft'); p.add_argument('--prompt',default='bolts. nuts. washers. screws. metal fasteners. mechanical components.'); p.add_argument('--max-new-tokens',type=int,default=1024); p.add_argument('--max-side',type=int,default=1024); a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); t=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'; dtype=torch.float16 if device=='cuda' else torch.float32
 im=Image.open(a.image).convert('RGB'); im.thumbnail((a.max_side,a.max_side))
 processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True); model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype=dtype).to(device).eval()
 task='<OPEN_VOCABULARY_DETECTION>'; prompt=task+a.prompt
 inputs=processor(text=prompt,images=im,return_tensors='pt').to(device)
 if device=='cuda': inputs['pixel_values']=inputs['pixel_values'].to(dtype)
 with torch.inference_mode(): ids=model.generate(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],max_new_tokens=a.max_new_tokens,num_beams=3,do_sample=False)
 decoded=processor.batch_decode(ids,skip_special_tokens=False)[0]
 parsed=processor.post_process_generation(decoded,task=task,image_size=im.size)
 (out/'raw_response.txt').write_text(decoded,encoding='utf-8'); (out/'predictions.json').write_text(json.dumps(parsed,indent=2,default=str),encoding='utf-8')
 vis=im.copy(); draw=ImageDraw.Draw(vis); data=parsed.get(task,{}) if isinstance(parsed,dict) else {}
 boxes=data.get('bboxes',[]) if isinstance(data,dict) else []; labels=data.get('bboxes_labels',[]) if isinstance(data,dict) else []
 for i,b in enumerate(boxes):
  x0,y0,x1,y1=map(float,b); draw.rectangle((x0,y0,x1,y1),outline='lime',width=3); draw.text((x0,max(0,y0-14)),str(labels[i] if i<len(labels) else 'candidate'),fill='yellow')
 vis.save(out/'annotated_candidates.png')
 summary={'model':a.model,'device':device,'prompt':a.prompt,'candidate_count':len(boxes),'elapsed_seconds':round(time.time()-t,2),'warning':'Open-vocabulary candidates only. Validate boxes visually; detections are not component-level presence/absence labels.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__':
 try: main()
 except Exception as e:
  Path('outputs').mkdir(exist_ok=True); raise
