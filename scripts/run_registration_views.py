"""Run registration POC on every BG image in the sample dataset; failed registrations are logged, not guessed."""
from pathlib import Path
import argparse,subprocess,sys,json

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='data/sample/AI_sample_CAD'); p.add_argument('--output',default='outputs/registration_all_views'); p.add_argument('--method',default='auto',choices=['auto','sift','akaze','orb']); a=p.parse_args()
 root=Path(a.data); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); script=Path(__file__).with_name('run_registration_poc.py')
 cases=[root]+[x for x in root.iterdir() if x.is_dir()]
 records=[]
 for case in cases:
  if not (case/'FG_00.png').exists() or not (case/'0000.json').exists(): continue
  images=sorted(case.glob('BG_00*.png'))
  for image in images:
   tag=(case.name+'_'+image.stem).replace(' ','_'); dest=out/tag
   cmd=[sys.executable,str(script),'--case',str(case),'--real-image',str(image),'--output',str(dest),'--method',a.method]
   r=subprocess.run(cmd,capture_output=True,text=True)
   rec={'case':str(case),'real_image':str(image),'returncode':r.returncode,'output':str(dest),'stdout':r.stdout[-3000:],'stderr':r.stderr[-3000:]}
   records.append(rec); print(tag,'OK' if r.returncode==0 else 'REGISTRATION_FAILED')
 (out/'all_views_run.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 print('Completed:',len(records),'views. See all_views_run.json; failures need manual point selection.')
if __name__=='__main__': main()
