"""Optional NVIDIA LocateAnything grounding experiment; saves raw response, no automatic absence claims."""
import argparse,json,re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel,AutoProcessor,AutoTokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--query',default='Locate all visible bolts, nuts, washers, and other mechanical fasteners. Return bounding boxes for each object.'); p.add_argument('--output',default='outputs/locateanything'); p.add_argument('--model',default='nvidia/LocateAnything-3B'); p.add_argument('--max-new-tokens',type=int,default=1024); p.add_argument('--mode',choices=['fast','slow','hybrid'],default='hybrid'); a=p.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('LocateAnything 3B is not recommended on CPU. Use a suitable CUDA GPU; model card evaluates on A100/H100-class hardware.')
 if torch.cuda.get_device_properties(0).total_memory < 24*1024**3: print('WARNING: GPU has under 24 GB VRAM. This 3B VLM may OOM; use DINOv3 experiment first.')
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
 processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True)
 model=AutoModel.from_pretrained(a.model,torch_dtype=dtype,trust_remote_code=True).to('cuda').eval()
 image=Image.open(a.image).convert('RGB')
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':a.query}]}]
 text=processor.py_apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 images,videos=processor.process_vision_info(messages)
 inputs=processor(text=[text],images=images,videos=videos,return_tensors='pt').to('cuda')
 with torch.inference_mode():
  response=model.generate(pixel_values=inputs['pixel_values'].to(dtype),input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],image_grid_hws=inputs.get('image_grid_hws'),tokenizer=tokenizer,max_new_tokens=a.max_new_tokens,use_cache=True,generation_mode=a.mode,temperature=0.0,do_sample=False,verbose=False)
 answer=response[0] if isinstance(response,tuple) else response
 if isinstance(answer,list): answer='\n'.join(map(str,answer))
 answer=str(answer)
 (out/'raw_grounding_response.txt').write_text(answer,encoding='utf-8')
 boxes=re.findall(r'<box>\s*([^<]+?)\s*</box>',answer)
 summary={'model':a.model,'query':a.query,'generation_mode':a.mode,'box_tokens_found':boxes,'raw_response_file':'raw_grounding_response.txt','warning':'Experimental model output. Check coordinate convention and visually validate every box; no part-level PRESENT/ABSENT decisions are produced.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
