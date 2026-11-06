"""Independently rerun the anchored author and diagonal selector, CPU only."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_wavefront_cpu';OLD=B/'e6/pixelprune_threshold32';I=B/'e1_independent_review';P=B/'e6/external_sources/pixelprune_8d347eb';sys.path[:0]=[str(D),str(P)]
from candidate import select
from pixelprune.methods.pred_2d import Pred2DSelector
from pixelprune.patches.qwen3_vl_hf import normalize_pixel_values_for_selector
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def terminal(m,source):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];cwd=Path(s['cwd']);assert [(cwd/x).resolve() for x in s['command'] if Path(x).name==source.name]==[source.resolve()];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'pixel_wavefront_cpu_review.json';assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized();torch.set_num_threads(1)
s=terminal(D/'monitor_check001',D/'check.py');assert (Path(s['cwd'])/s['command'][s['command'].index('--out')+1]).resolve()==D/'result001.json'
source=read(D/'plan001.json');assert source['thresholds']==[0.4,0.45,0.5] and all(sha(k)==v for k,v in source['source_sha256'].items());r=read(D/'result001.json');assert r['status']=='completed' and r['plan_sha256']==sha(D/'plan001.json') and r['images']==32 and r['comparisons']==96 and r['synthetic_cases']==126 and r['matches']==96 and r['GT_opened'] is False and r['predictions_opened'] is False and r['gpu_used'] is False and r['cuda_initialized'] is False
prior=read(I/'pixel_threshold32_review.json');terminal(I/'monitor_pixel_threshold32',I/'review_pixel_threshold32.py');assert prior['status']=='passed' and prior['independent_review'] is True and prior['images']==32 and prior['threshold_cases']==320
rng=torch.Generator(device='cpu').manual_seed(20261105);cases=0
for h,w in ((1,1),(1,7),(7,1),(2,3),(3,2),(5,5),(8,9)):
 for dtype in (torch.float32,torch.bfloat16):
  for thr in source['thresholds']:
   v=torch.rand((h*w,5),generator=rng).to(dtype)
   for x in (v,torch.zeros_like(v),torch.ones_like(v)):
    x1=Pred2DSelector(method='max',threshold=thr,anchored=True)._select_2d_loco_anchored(x,h,w,x.device);x2=select(x,h,w,thr);assert torch.equal(x1,x2);cases+=1
assert cases==126
# Test strict boundary around a threshold using same dtype and direct author comparison.
boundary=0
for dtype in (torch.float32,torch.bfloat16):
 for th in source['thresholds']:
  t=torch.tensor(th,dtype=dtype);v=torch.stack([torch.tensor(0.,dtype=dtype),torch.nextafter(t,torch.tensor(float('-inf'),dtype=dtype)),t,torch.nextafter(t,torch.tensor(float('inf'),dtype=dtype))]).reshape(4,1)
  for h,w in ((1,4),(4,1),(2,2)):
   x=Pred2DSelector(method='max',threshold=th,anchored=True)._select_2d_loco_anchored(v,h,w,v.device);assert torch.equal(x,select(v,h,w,th));boundary+=1
old=read(OLD/'plan001.json');assert len(old['samples'])==32 and len(r['rows'])==96;checks=[]
for i,sample in enumerate(old['samples']):
 cap=Path(sample['capture_path']);assert sha(cap)==sample['capture_sha256'];processor=torch.load(cap,map_location='cpu',weights_only=True)['actual']['processor'];assert processor['image_grid_thw'].tolist()==[[1,64,64]];tokens=normalize_pixel_values_for_selector(processor['pixel_values']).reshape(1024,-1);assert torch.isfinite(tokens).all()
 journal=[json.loads(x) for x in (OLD/f'run001/image_{i:03d}/journal.jsonl').read_text().splitlines()]
 for j,thr in enumerate(source['thresholds']):
  oldrow=[x for x in journal if x.get('event')=='completed' and x['threshold']==thr];assert len(oldrow)==1;row=r['rows'][i*3+j];author=Pred2DSelector(method='max',threshold=thr,anchored=True)._select_2d_loco_anchored(tokens,32,32,tokens.device).tolist();candidate=select(tokens,32,32,thr).tolist()
  assert row['index']==i and row['identity']==sample['identity'] and row['threshold']==thr and row['ordered_equal'] is True and row['selected_indices']==candidate==author==oldrow[0]['selected_indices'] and row['selected_count']==len(author);checks.append({'index':i,'threshold':thr,'count':len(author)})
 if i%4==3:print('independent images',i+1,flush=True)
assert len(checks)==96 and all(sha(k)==v for k,v in source['source_sha256'].items())
proof={'status':'passed','independent_review':True,'images':32,'comparisons':96,'ordered_matches':96,'recomputed_synthetic_cases':cases,'additional_boundary_cases':boundary,'GT_opened':False,'predictions_opened':False,'gpu_used':False,'cuda_initialized':False,'source_plan_sha256':sha(D/'plan001.json'),'result_sha256':sha(D/'result001.json'),'run_started_sha256':sha(D/'monitor_check001/started.json'),'run_terminal_sha256':sha(D/'monitor_check001/terminal.json'),'prior_review_sha256':sha(I/'pixel_threshold32_review.json'),'reviewer_sha256':sha(__file__),'checks':checks,'limitations':['Only finite CPU BF16 processor inputs and finite synthetic float32/BF16 cases are tested; GPU and model behavior not established.','The antidiagonal recurrence follows left and up from the previous diagonal and upper-left from two diagonals earlier, but parallel arithmetic, input dtype and threshold rounding must still be checked on a real device.','CPU wall times unpaired, no speed or full pipeline claim.','Inputs and threshold grid selected after prior DEV observations.'],'errors':[]};a.out.write_text(json.dumps(proof,indent=2)+'\n');print('passed',flush=True)
