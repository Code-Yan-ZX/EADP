"""Independent vectorized 32-image four-stream minimax recomputation."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np,torch
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e4_multistream_local_minimax_v2';OLD=B/'e4_multistream_local_minimax';E=B/'e4_c_microbench_full';SEEDS=(20261104,20261105,20261106)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervision '+str(mon))
def block_ids():
 return np.array([[r*32+c,r*32+c+1,(r+1)*32+c,(r+1)*32+c+1] for r in range(0,32,2) for c in range(0,32,2)],dtype=np.int64)
def recompute(streams,image_index):
 ids=block_ids();joint=np.zeros((256,4,4));main=np.zeros((256,4,4));scales=[];zero=[]
 for stream,x in enumerate(streams):
  a=x.detach().cpu().to(torch.float64).numpy();scale=float(np.sqrt(np.sum(a*a,axis=1).mean())+1e-6);scales.append(scale)
  if not np.isfinite(a).all() or not math.isfinite(scale):raise ValueError('nonfinite stream')
  if np.count_nonzero(a)==0:zero.append(stream)
  for i in range(4):
   for j in range(i+1,4):
    delta=a[ids[:,i]]-a[ids[:,j]];dist=np.sqrt((delta*delta).sum(axis=1))/scale
    if not np.isfinite(dist).all():raise ValueError('nonfinite pair distance')
    joint[:,i,j]=np.maximum(joint[:,i,j],dist);joint[:,j,i]=joint[:,i,j]
    if stream==0:main[:,i,j]=dist;main[:,j,i]=dist
 joint_radius=joint.max(axis=1);main_radius=main.max(axis=1);choice={'minimax':np.argmin(joint_radius,axis=1),'top_left':np.zeros(256,dtype=int),'main_minimax':np.argmin(main_radius,axis=1)}
 for seed in SEEDS:
  choice[f'random_{seed}']=np.array([int.from_bytes(hashlib.sha256(f'local-minimax-random-v1|{seed}|{image_index}|{b//16}|{b%16}'.encode()).digest()[:8],'big')%4 for b in range(256)])
 ties=[]
 for b in range(256):
  tie=[int(ids[b,i]) for i in range(4) if joint_radius[b,i]==joint_radius[b,choice['minimax'][b]]]
  if len(tie)>1:ties.append({'block_index':b,'candidate_ids':tie})
 result={}
 for name,indices in choice.items():
  selected=ids[np.arange(256),indices].tolist();radius=joint_radius[np.arange(256),indices].tolist()
  result[name]=(selected,sorted(selected),radius)
 return result,ties,zero,scales
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'local_minimax_v2_result_review.json'
 if a.out.resolve()!=dest or dest.exists() or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('output/CPU environment')
 torch.set_num_threads(1)
 if torch.cuda.is_initialized():raise ValueError('unexpected CUDA')
 plan=read(D/'plan001.json');inv=read(D/'inventory001.json');result=read(D/'result001.json');pre=I/'local_minimax_v2_prelaunch_review.json';q=read(pre);clean(D/'monitor_run001',D/'run.py',D/'result001.json');clean(I/'monitor_local_minimax_v2_prelaunch',I/'review_local_minimax_v2_prelaunch.py',pre)
 if q['status']!='passed' or q['plan_sha256']!=sha(D/'plan001.json') or q['inventory_sha256']!=sha(D/'inventory001.json') or q['reviewer_sha256']!=sha(I/'review_local_minimax_v2_prelaunch.py') or result['status']!='passed' or result['images']!=32 or result['GT_opened'] is not False or result['GPU_used'] is not False or result['plan_sha256']!=sha(D/'plan001.json') or result['rows_sha256']!=sha(D/'rows001.jsonl') or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source/result/prelaunch')
 rows=[json.loads(x) for x in (D/'rows001.jsonl').read_text().splitlines()];items=inv['archives']
 if len(rows)!=32 or len(items)!=32:raise ValueError('incomplete panel')
 improvements=0
 for index,(row,item) in enumerate(zip(rows,items)):
  path=Path(item['path']);digest=sha(path)
  if index!=item['index'] or row['index']!=index or row['archive_sha256']!=digest or digest!=item['sha256'] or any(row[k]!=str(item[k]) if k=='qid' else row[k]!=item[k] for k in ('task','qid','group_id')):raise ValueError('archive identity')
  data=torch.load(path,map_location='cpu',weights_only=True);x=data['actual'];visual=x['visual'];grid=x['processor']['image_grid_thw'].tolist()
  if grid!=[[1,64,64]] or len(visual)!=2 or len(visual[1])!=3:raise ValueError('four stream grid')
  streams=[visual[0],*visual[1]]
  if any(not torch.is_tensor(t) or t.ndim!=2 or t.shape[0]!=1024 or not t.is_floating_point() for t in streams):raise ValueError('stream shape')
  arms,ties,zero,scales=recompute(streams,index);ordered=x['ordered'].reshape(-1).tolist();occupancy=[0]*256
  if len(ordered)!=256 or len(set(ordered))!=256:raise ValueError('EADP identity')
  for n in ordered:
   if type(n) is not int or not 0<=n<1024:raise ValueError('EADP position')
   occupancy[(n//32//2)*16+(n%32//2)]+=1
  if row['eadp_occupancy']!={str(n):occupancy.count(n) for n in range(5)} or row['tie_cases']!=ties or row['tie_blocks']!=len(ties) or row['zero_stream_indices']!=zero or row['zero_streams']!=len(zero) or not np.allclose(row['scales'],scales,rtol=1e-13,atol=1e-13):raise ValueError('ties, zero, scales or EADP occupancy')
  for name,(byblock,sorted_ids,radii) in arms.items():
   a0=row['arms'][name]
   if a0['block_ids']!=byblock or a0['ids']!=sorted_ids or a0['eadp_set_overlap']!=len(set(sorted_ids)&set(ordered)) or not np.allclose(a0['radii'],radii,rtol=1e-12,atol=1e-12) or not math.isclose(a0['max_radius'],max(radii),rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(a0['median_radius'],float(np.median(radii)),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('arm recomputation '+name)
  if any(x>y+1e-12 for name,(_,_,radii) in arms.items() for x,y in zip(arms['minimax'][2],radii)):raise ValueError('per-block objective')
  improvements+=any(x<y for x,y in zip(arms['minimax'][2],arms['main_minimax'][2]))
  if sha(path)!=digest:raise ValueError('archive changed during read')
  del data,x,visual,streams,arms
  if index%4==3:print('audited',index+1,flush=True)
 if improvements!=result['strict_radius_improvement_images'] or result['stop_no_strict_improvement']!=(improvements==0) or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('aggregate or source drift')
 proof={'status':'passed','independent_review':True,'images':32,'GT_opened':False,'GPU_used':False,'model_executed':False,'existing_archive_deserialized':True,'errors':[],'plan_sha256':sha(D/'plan001.json'),'result_sha256':sha(D/'result001.json'),'rows_sha256':sha(D/'rows001.jsonl'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'prelaunch_sha256':sha(pre),'reviewer_sha256':sha(__file__),'limitations':['32 posthoc cached features; no causal, quality, timing, novelty or equal-compute claim.','Only per-block geometric proxy; it does not establish model output behavior.','Full input archives deserialized; only four streams/grid/selected IDs used.']}
 with dest.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
