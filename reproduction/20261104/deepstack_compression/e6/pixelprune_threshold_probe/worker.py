"""One-image author selector CPU resource probe; never opens labels or predictions."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];P=HERE/'plan001.json';PIXEL=BASE/'e6/external_sources/pixelprune_8d347eb';CAP=BASE/'e5/selector_real32/run_001/image_000/capture.pt';R=BASE/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def metadata(t):
 x=t.detach().cpu().contiguous();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}
def append(path,obj):
 with path.open('a') as f: f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def source_gate(plan):
 if plan['GT_opened'] is not False or plan['predictions_opened'] is not False or plan['gpu_authorized'] is not False or any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('source or scope changed')
 proof=R/'selector_real32_run001_review.json';started=json.loads((R/'monitor_selector_real32_run001/started.json').read_text());terminal=json.loads((R/'monitor_selector_real32_run001/terminal.json').read_text());r=json.loads(proof.read_text())
 if sha(proof)!=plan['prior_review_sha256'] or r['status']!='passed' or r['independent_review'] is not True or r['array_sha256'].get(str(CAP))!=plan['capture_sha256'] or sha(CAP)!=plan['capture_sha256'] or terminal['reason']!='completed' or terminal['worker_returncode']!=0 or terminal['remaining_process_group']!=[] or started['worker_pid']!=terminal['worker_pid']:raise ValueError('input source/exit not accepted')
def child(out,threshold,plan):
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(PIXEL));from pixelprune.core import compute_merged_keep_indices
 from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
 if torch.cuda.is_initialized():raise ValueError('unexpected CUDA initialized')
 capture=torch.load(CAP,map_location='cpu',weights_only=True);processor=capture['actual']['processor'];pixel=processor['pixel_values'];grid=processor['image_grid_thw']
 if metadata(pixel)!=plan['processor']['pixel_values'] or metadata(grid)!=plan['processor']['image_grid_thw']:raise ValueError('processor identity mismatch')
 norm=normalize_pixel_values_for_selector(pixel);t=time.monotonic();selected=compute_merged_keep_indices(norm,grid,spatial_merge_size=2,method='pred_2d',metric='max',threshold=threshold,anchored=True);duration=time.monotonic()-t
 if len(selected)!=1:raise ValueError('single image expected')
 ids=selected[0].detach().cpu().tolist()
 if not ids or ids!=sorted(set(ids)) or any(type(x) is not int or x<0 or x>=1024 for x in ids):raise ValueError('invalid selected index')
 result={'status':'completed','threshold':threshold,'selector_path':'fast' if threshold==0 else 'anchored_python','device':'cpu','selected_count':len(ids),'selected_indices':ids,'wall_seconds':duration,'pixel_metadata':metadata(pixel),'grid_metadata':metadata(grid),'cuda_initialized':torch.cuda.is_initialized(),'GT_opened':False,'predictions_opened':False}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--threshold',type=float);a=p.parse_args();plan=json.loads(P.read_text())
 if a.threshold is not None:
  if a.threshold not in plan['thresholds']:raise ValueError('threshold outside plan')
  child(a.out,a.threshold,plan);return
 source_gate(plan);a.out.mkdir(parents=True,exist_ok=False);journal=a.out/'journal.jsonl'
 for i,t in enumerate(plan['thresholds']):
  path=a.out/('threshold_%02d.json'%i);append(journal,{'event':'started','threshold':t,'index':i});start=time.monotonic()
  try:
   c=subprocess.run([sys.executable,str(HERE/'worker.py'),'--out',str(path),'--threshold',str(t)],timeout=plan['per_threshold_timeout_seconds'],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
   event={'event':'completed' if c.returncode==0 else 'failed','threshold':t,'index':i,'returncode':c.returncode,'stderr_tail':c.stderr[-2000:],'elapsed_seconds':time.monotonic()-start}
  except subprocess.TimeoutExpired as e:event={'event':'timeout','threshold':t,'index':i,'elapsed_seconds':time.monotonic()-start}
  append(journal,event)
 done={'status':'completed','thresholds_planned':len(plan['thresholds']),'events':len(plan['thresholds']),'plan_sha256':sha(P),'journal_sha256':sha(journal),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'limitations':['CPU resource/selection probe on one already archived processor input; CPU and GPU identity not established','thresholds assessed after prior DEV results; no accuracy or cost result']}
 with (a.out/'completed.json').open('x') as f:json.dump(done,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
