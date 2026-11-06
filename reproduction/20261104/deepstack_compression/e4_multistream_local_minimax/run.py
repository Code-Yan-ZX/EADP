"""Admitted CPU-only feature diagnostic; no GT, predictions or model execution."""
import argparse,datetime,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';END=1794065414
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(mon,script,out=None):
 a=read(mon/'started.json');b=read(mon/'terminal.json');cwd=Path(a['cwd']);cmd=a['command']
 if a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor mismatch')
def admission():
 if datetime.datetime.now(datetime.timezone.utc).timestamp()>=END or read(B/'initial_plan.json')['budget_end_unix']!=END:raise ValueError('deadline')
 p=read(D/'plan001.json');inventory=read(D/'inventory001.json');cpu=read(D/'cpu_proof001.json');pre=I/'local_minimax_prelaunch_review.json';reviewer=I/'review_local_minimax_prelaunch.py';term(I/'monitor_local_minimax_prelaunch',reviewer,pre);q=read(pre)
 term(D/'monitor_inventory001',D/'inventory.py');term(D/'monitor_cpu001',D/'cpu_check.py');term(D/'monitor_freeze001',D/'freeze.py')
 if p.get('status')!='frozen' or p.get('real_tensors_deserialized') is not False or p.get('GT_opened') is not False or inventory.get('status')!='passed' or inventory.get('images')!=32 or cpu.get('status')!='passed' or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('frozen source')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('real_tensors_deserialized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('plan_sha256')!=sha(D/'plan001.json') or q.get('inventory_sha256')!=sha(D/'inventory001.json') or q.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('independent prelaunch proof')
 return p,inventory
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
 if args.out.resolve()!=(D/'result001.json').resolve() or args.out.exists() or (D/'rows001.jsonl').exists():raise ValueError('fixed output')
 plan,inv=admission()
 if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only thread environment')
 expected=plan['source_sha256'][str(D/'core.py')]
 if sha(D/'core.py')!=expected or '_local_minimax_core' in sys.modules:raise ValueError('core source')
 spec=importlib.util.spec_from_file_location('_local_minimax_core',D/'core.py');core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 import numpy as np,torch
 if np.__version__!=plan['numpy_version']:raise ValueError('NumPy version')
 torch.set_num_threads(1)
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 rows=[]
 for item in inv['archives']:
  path=Path(item['path'])
  if sha(path)!=item['sha256']:raise ValueError('archive hash')
  data=torch.load(path,map_location='cpu',weights_only=True);x=data['actual'];v=x['visual'];grid=x['processor']['image_grid_thw'].tolist()
  if grid!=[[1,64,64]] or not isinstance(v,(tuple,list)) or len(v)!=2 or len(v[1])!=3:raise ValueError('grid or four streams')
  tensors=[v[0],*v[1]]
  if any(not torch.is_tensor(t) or t.ndim!=2 or t.shape[0]!=1024 or not t.is_floating_point() for t in tensors):raise ValueError('feature shape')
  streams=[t.detach().cpu().to(dtype=torch.float64).numpy().reshape(32,32,-1) for t in tensors];result=core.select(streams,item['index']);ordered=x['ordered'].reshape(-1).tolist()
  if len(ordered)!=256 or len(set(ordered))!=256 or any(type(i) is not int or i<0 or i>=1024 for i in ordered):raise ValueError('EADP IDs')
  occupancy=[0]*256
  for i in ordered:occupancy[(i//32//2)*16+(i%32//2)]+=1
  output={'index':item['index'],'task':item['task'],'qid':str(item['qid']),'group_id':item['group_id'],'archive_sha256':item['sha256'],'tie_blocks':result['tie_blocks'],'zero_streams':result['zero_streams'],'scales':[float(v) for v in result['scales']],'eadp_occupancy':{str(n):occupancy.count(n) for n in range(5)},'arms':{}}
  for name,ids in result['ids'].items():
   if len(ids)!=256 or len(set(ids))!=256:raise ValueError('ID inventory')
   radii=result['radii'][name];output['arms'][name]={'ids':ids,'radii':radii,'max_radius':max(radii),'median_radius':float(np.median(radii)),'eadp_set_overlap':len(set(ids)&set(ordered))}
  if sha(path)!=item['sha256']:raise ValueError('archive changed during load')
  rows.append(output);del data,x,v,tensors,streams,result
 if len(rows)!=32 or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('incomplete or source drift')
 rp=D/'rows001.jsonl'
 with rp.open('x') as f:
  for x in rows:f.write(json.dumps(x,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 improvements=sum(any(x<y for x,y in zip(r['arms']['minimax']['radii'],r['arms']['main_minimax']['radii'])) for r in rows)
 output={'status':'passed','images':32,'strict_radius_improvement_images':improvements,'stop_no_strict_improvement':improvements==0,'GT_opened':False,'GPU_used':False,'model_executed':False,'existing_archive_deserialized':True,'rows_sha256':sha(rp),'plan_sha256':sha(D/'plan001.json'),'limitations':['Posthoc 32-image feature diagnostic; no quality, logit, timing, causal or novelty conclusion.','One representative per block can discard task-critical tokens.','Full archive deserialized, only four streams/grid/selected IDs parsed.'],'errors':[]}
 with args.out.open('x') as f:json.dump(output,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
