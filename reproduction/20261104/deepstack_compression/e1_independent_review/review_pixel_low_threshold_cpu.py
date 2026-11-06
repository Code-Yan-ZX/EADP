"""Independent rerun of positive-threshold author dispatch on saved CPU pixels."""
import argparse,hashlib,importlib.util,json,os,struct,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_low_threshold_cpu';I=B/'e1_independent_review';PIXEL=B/'e6/external_sources/pixelprune_8d347eb'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=I/'pixel_low_threshold_cpu_review.json'
 if a.out.resolve()!=out or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU output/visibility')
 s=read(D/'monitor_check001/started.json');t=read(D/'monitor_check001/terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s['worker_pid']!=t['worker_pid'] or (cwd/cmd[1]).resolve()!=D/'check.py':raise ValueError('original supervisor')
 plan=read(D/'plan001.json');result=read(D/'result001.json')
 if plan['thresholds']!=[.1,.2,.3] or plan['GT_opened'] is not False or plan['predictions_opened'] is not False or result.get('status')!='completed' or result.get('plan_sha256')!=sha(D/'plan001.json') or result.get('GT_opened') is not False or result.get('predictions_opened') is not False or result.get('cuda_initialized') is not False:raise ValueError('prior CPU scope')
 for name,h in plan['source_sha256'].items():
  if sha(name)!=h:raise ValueError('source mismatch')
 sys.path.insert(0,str(PIXEL));import torch
 from pixelprune.methods import pred_2d
 from pixelprune.patches import qwen3_vl_hf
 if torch.cuda.is_initialized() or Path(pred_2d.__file__).resolve()!=PIXEL/'pixelprune/methods/pred_2d.py' or Path(qwen3_vl_hf.__file__).resolve()!=PIXEL/'pixelprune/patches/qwen3_vl_hf.py':raise ValueError('actual module source/CUDA')
 torch.set_num_threads(1);path=B/'e6/pixel_wavefront_cpu/candidate.py';spec=importlib.util.spec_from_file_location('_independent_low_wavefront',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 samples=read(B/'e6/pixelprune_threshold32/plan001.json')['samples'];old=result['rows'];rows=[]
 if len(samples)!=32 or len(old)!=96 or result.get('comparisons')!=96:raise ValueError('case count')
 for i,sample in enumerate(samples):
  capture=Path(sample['capture_path']);
  if sha(capture)!=sample['capture_sha256']:raise ValueError('input capture changed')
  proc=torch.load(capture,map_location='cpu',weights_only=True)['actual']['processor'];pixels=proc['pixel_values'];grid=proc['image_grid_thw']
  if grid.tolist()!=[[1,64,64]]:raise ValueError('wrong grid')
  normal=qwen3_vl_hf.normalize_pixel_values_for_selector(pixels);tokens=normal.reshape(1024,-1)
  if not torch.isfinite(tokens).all().item():raise ValueError('nonfinite input')
  for j,threshold in enumerate(plan['thresholds']):
   author=pred_2d.Pred2DSelector(method='max',threshold=threshold,anchored=True);anchored=author._select_2d_loco_anchored;count=[]
   def instrument(*x,**kw):count.append(1);return anchored(*x,**kw)
   def no_fast(*_x,**_kw):raise ValueError('unexpected fast dispatch')
   author._select_2d_loco_anchored=instrument;author._select_2d_loco_fast=no_fast
   actual=author.select(normal,grid,spatial_merge_size=2)
   if len(count)!=1 or len(actual)!=1:raise ValueError('actual author dispatch')
   ref=actual[0].tolist();candidate=m.select(tokens,32,32,threshold).tolist();oldrow=old[i*3+j]
   if oldrow['index']!=i or oldrow['identity']!=sample['identity'] or oldrow['threshold']!=threshold or oldrow['selected_indices']!=ref or oldrow['candidate_indices']!=candidate or oldrow['ordered_equal'] is not (ref==candidate) or oldrow['count']!=len(ref):raise ValueError('prior row mismatch')
   if ref!=candidate:raise ValueError('author/candidate ordered mismatch')
   rows.append({'index':i,'identity':sample['identity'],'threshold':threshold,'selected_count':len(ref),'ordered_indices_sha256':hashlib.sha256(struct.pack('<'+'q'*len(ref),*ref)).hexdigest(),'actual_author_anchored_dispatch':True})
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 proof={'status':'passed','independent_review':True,'result_sha256':sha(D/'result001.json'),'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(__file__),'ordered_matches':len(rows),'case_count':len(rows),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'cases':rows,'errors':[],'limitations':['post-hoc reuse of 32 DEV processor captures; no accuracy conclusion','author actual positive-threshold selector dispatch and candidate rerun only on CPU','GPU/full-model identity and speed remain unverified at these thresholds']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print('passed',len(rows))
if __name__=='__main__':main()
