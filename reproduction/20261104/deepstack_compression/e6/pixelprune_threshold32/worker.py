"""Bounded, 32-image CPU author PixelPrune selector calibration; no labels/models."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];PLAN=HERE/'plan001.json';PIXEL=BASE/'e6/external_sources/pixelprune_8d347eb';R=BASE/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def meta(t):
 x=t.detach().cpu().contiguous();return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}
def append(path,obj):
 with path.open('a') as f:f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def validate_plan(plan):
 if plan.get('status')!='frozen' or plan.get('GT_opened') is not False or plan.get('predictions_opened') is not False or plan.get('gpu_authorized') is not False or len(plan.get('samples',[]))!=32 or len({x['group_id'] for x in plan['samples']})!=32 or plan['thresholds']!=[0.,.4,.45,.5,.55,.6,.65,.7,.75,.8]:raise ValueError('bad calibration plan')
 if any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('source changed')
 proof=R/'selector_real32_run001_review.json';s=json.loads((R/'monitor_selector_real32_run001/started.json').read_text());t=json.loads((R/'monitor_selector_real32_run001/terminal.json').read_text());r=json.loads(proof.read_text())
 if sha(proof)!=plan['prior_review_sha256'] or r['status']!='passed' or r['independent_review'] is not True or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('prior proof/exit mismatch')
 if any(r['array_sha256'].get(x['capture_path'])!=x['capture_sha256'] for x in plan['samples']):raise ValueError('archive binding mismatch')
 pp=R/'pixel_threshold_probe001_review.json';ps=json.loads((R/'monitor_pixel_threshold_probe001/started.json').read_text());pt=json.loads((R/'monitor_pixel_threshold_probe001/terminal.json').read_text());pv=json.loads(pp.read_text())
 if sha(pp)!=plan['one_image_probe_review_sha256'] or pv.get('status')!='passed' or pv.get('independent_review') is not True or pt['reason']!='completed' or pt['worker_returncode']!=0 or pt['remaining_process_group']!=[] or ps['worker_pid']!=pt['worker_pid']:raise ValueError('one-image feasibility review mismatch')
def child(out,index,plan):
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(PIXEL));from pixelprune.core import compute_merged_keep_indices
 from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 sample=plan['samples'][index];capture=Path(sample['capture_path'])
 if sha(capture)!=sample['capture_sha256']:raise ValueError('input archive changed')
 x=torch.load(capture,map_location='cpu',weights_only=True)['actual']['processor'];pixels=x['pixel_values'];grid=x['image_grid_thw']
 if meta(pixels)!=sample['processor']['pixel_values'] or meta(grid)!=sample['processor']['image_grid_thw']:raise ValueError('processor identity mismatch')
 norm=normalize_pixel_values_for_selector(pixels);out.mkdir(exist_ok=False)
 for j,threshold in enumerate(plan['thresholds']):
  append(out/'journal.jsonl',{'event':'started','threshold_index':j,'threshold':threshold});t=time.monotonic()
  try:
   ids=compute_merged_keep_indices(norm,grid,spatial_merge_size=2,method='pred_2d',metric='max',threshold=threshold,anchored=True)
   if len(ids)!=1:raise ValueError('expected single image')
   ids=ids[0].detach().cpu().tolist()
   if not ids or ids!=sorted(set(ids)) or any(type(z) is not int or not 0<=z<1024 for z in ids):raise ValueError('bad selector output')
   event={'event':'completed','threshold_index':j,'threshold':threshold,'selected_count':len(ids),'selected_indices':ids,'selector_path':'fast' if threshold==0 else 'anchored_python','wall_seconds':time.monotonic()-t}
  except Exception as e:event={'event':'failed','threshold_index':j,'threshold':threshold,'error_type':type(e).__name__,'error':str(e)[:1000],'wall_seconds':time.monotonic()-t}
  append(out/'journal.jsonl',event)
 with (out/'complete.json').open('x') as f:json.dump({'status':'completed','thresholds':len(plan['thresholds']),'sample':sample['identity'],'cuda_initialized':torch.cuda.is_initialized(),'GT_opened':False,'predictions_opened':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--image-index',type=int);a=p.parse_args();plan=json.loads(PLAN.read_text())
 if a.image_index is not None:
  if a.image_index not in range(32):raise ValueError('image out of range')
  child(a.out,a.image_index,plan);return
 validate_plan(plan);a.out.mkdir(parents=True,exist_ok=False);journal=a.out/'journal.jsonl'
 for i,sample in enumerate(plan['samples']):
  out=a.out/('image_%03d'%i);append(journal,{'event':'started','image_index':i,'sample':sample['identity']});t=time.monotonic()
  try:
   c=subprocess.run([sys.executable,str(HERE/'worker.py'),'--out',str(out),'--image-index',str(i)],timeout=plan['per_image_timeout_seconds'],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});event={'event':'completed' if c.returncode==0 else 'failed','image_index':i,'returncode':c.returncode,'stderr_tail':c.stderr[-2000:],'elapsed_seconds':time.monotonic()-t}
  except subprocess.TimeoutExpired:event={'event':'timeout','image_index':i,'elapsed_seconds':time.monotonic()-t}
  append(journal,event)
 validate_plan(plan)
 with (a.out/'completed.json').open('x') as f:json.dump({'status':'completed','images_planned':32,'thresholds_planned':len(plan['thresholds']),'plan_sha256':sha(PLAN),'journal_sha256':sha(journal),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'scope':'completion of attempt does not mean all image/threshold cases succeeded'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
