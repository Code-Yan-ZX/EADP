"""CPU verification for a complete PriorTR 32-question diagnostic, no GT."""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation';sys.path.insert(0,str(PREP))
import gate
from panel_core import ARMS,EVENT,validate
from launch import configure,PYTHON

def readlines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial JSONL')
 return [json.loads(x) for x in b.splitlines()]

def actual_exit(run,s,t,l,e,b):
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or l.get('controller')!=e.get('controller') or l.get('controller',{}).get('pid')!=s.get('worker_pid')):raise ValueError('supervisor identity')
 for k,v in (('timeout_seconds',b['outer_timeout_seconds']),('stale_seconds',b['journal_stale_seconds']),('snapshot_seconds',b['snapshot_seconds']),('poll_seconds',b['poll_seconds']),('grace_seconds',b['grace_seconds']),('progress_paths',[str(run/'data/generated.jsonl')])):
  if s.get(k)!=v:raise ValueError('supervisor policy')
 cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or cmd.count('--run-id')!=1 or run.name!='run_'+cmd[cmd.index('--run-id')+1]:raise ValueError('supervisor launch command')
 if l.get('worker_sha256')!=gate.sha(HERE/'panel_worker.py') or l.get('launcher_sha256')!=gate.sha(HERE/'launch.py') or l.get('budget_sha256')!=gate.sha(HERE/'resource_plan.json') or l['started_unix']+b['outer_timeout_seconds']>gate.read(BASE/'initial_plan.json')['budget_end_unix']:raise ValueError('launch source or deadline')
 if (e.get('returncode')!=0 or e.get('container_exit_code')!=0 or e.get('timed_out') is not False or e.get('interrupted') is not None or e.get('failure') is not None or e.get('container_stopped_verified') is not True or e.get('client_exited_while_container_running') is not False or e.get('gpu_pids_after')!=[]):raise ValueError('launcher exit')
 ident=e['container_identity'];obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],timeout=20))[0]
 if obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or obj['State']['Running'] or obj['State']['Pid']!=0 or obj['State']['ExitCode']!=0 or obj['State']['OOMKilled']:raise ValueError('actual container exit')
 cmd=l['command'];image=obj['Config']['Image'];plan=gate.read(BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 prior=gate.read(PREP/'smoke_001/launch.json')['command'][:];name='eadp-priortr32-'+run.name.removeprefix('run_')+'-'+str(e['controller']['pid']);prior[prior.index('--name')+1]=name
 i=prior.index(str(PREP/'smoke_worker.py'));expected=configure(prior[:i]+[str(HERE/'panel_worker.py'),'--execute','--out',str(run)],run/'runtime_cache')
 devices={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'};actual_devices=obj['HostConfig'].get('Devices') or []
 envs=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']
 if (cmd!=expected or e.get('name')!=name or obj['Image']!=plan['container_image'] or image!=plan['container_image'] or obj['Config']['Entrypoint']!=[str(PYTHON)] or obj['Config']['User']!='1000:1000' or obj['Config']['WorkingDir']!=plan['author_root'] or obj['HostConfig']['NetworkMode']!='none' or {x['PathOnHost'] for x in actual_devices}!=devices or any(x['PathOnHost']!=x['PathInContainer'] or x['CgroupPermissions']!='rwm' for x in actual_devices) or any(x not in obj['Config']['Env'] for x in envs)):
  raise ValueError('actual container image/device/environment differs')
 if cmd.count(str(HERE/'panel_worker.py'))!=1 or cmd.count('--out')!=1 or Path(cmd[cmd.index('--out')+1]).resolve()!=run or obj['Config']['Cmd']!=cmd[cmd.index(image)+1:]:raise ValueError('actual worker command')
 spec=importlib.util.spec_from_file_location('_priortr_smoke_verify',PREP/'smoke_verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for proc in (e['controller'],ident['host_init']):
  if not isinstance(proc,dict) or not isinstance(proc.get('pid'),int) or not isinstance(proc.get('start_ticks'),int) or not proc.get('boot_id') or module.alive(proc):raise ValueError('process still live or identity invalid')
 return module

def validate_trace(r):
 trace=r['trace'];calls=trace['calls'];layers=trace['layers'];inj=trace['injections'];prunes=trace['prunes'];ids=r['output_ids'];seq=calls[0]['tokens'];k=1024 if r['arm']=='stock_native' else (113 if r['arm']=='author_default' else 256)
 counts=[x['tokens'] for x in layers if x['step']==0]
 if counts!=[seq]*3+[seq-(1024-k)]*33 or len(inj)!=3:raise ValueError('actual layer sequence counts')
 for i,x in enumerate(inj):
  want=1024 if i<2 else k
  if x['rows']!=want or x['mask_count']!=want:raise ValueError('actual injection alignment')
 length=seq-(1024-k)
 if len(calls)!=len(ids) or calls[0]['cache_after']!=[length]*36:raise ValueError('prefill KV or call count')
 for i,c in enumerate(calls[1:]):
  if c['cache_before']!=[length+i]*36 or c['cache_after']!=[length+i+1]*36:raise ValueError('decode cache')
 small=r['small_diagnostics']
 if r['arm']=='stock_native':
  if prunes or small:raise ValueError('stock prune diagnostics')
  return
 if len(prunes)!=1:raise ValueError('missing actual prune')
 p=prunes[0];start,end=p['range_before'];idx=np.asarray(small['selected_ids_0']);scores=np.asarray(small['selection_scores_0'],dtype=np.float64)
 if p['layer']!=2 or end-start!=1024 or p['range_after']!=[start,start+k] or len(idx)!=k or len(set(idx.tolist()))!=k or (idx<0).any() or (idx>=1024).any() or (np.diff(idx)<0).any() or scores.shape!=(1024,) or not np.isfinite(scores).all():raise ValueError('actual selection')
 mask=np.ones(1024,dtype=bool);mask[idx]=False
 if mask.any() and scores[idx].min()<scores[mask].max():raise ValueError('actual selection not top-k')
 kept=np.concatenate((np.arange(start),start+idx,np.arange(end,seq)));before=np.asarray(small['prune_before_position_0']);after=np.asarray(small['prune_after_position_0']);actual_mask=np.asarray(small['prune_after_mask_0'])
 if not np.array_equal(np.take(before,kept,axis=2),after) or actual_mask.tolist()!=[[False]*start+[True]*k+[False]*(seq-end)]:raise ValueError('actual position/mask gather')

def e4_comparison_gate():
 formal=BASE/'e4_panel_v2/panel_003_verification.json';review=BASE/'e1_independent_review/e4panel003_review.json';records=BASE/'e4_panel_v2/panel_003/records.jsonl';fs=BASE/'e4_panel_v2/monitor_verify_panel003/started.json';ft=BASE/'e4_panel_v2/monitor_verify_panel003/terminal.json';rs=BASE/'e1_independent_review/monitor_e4panel003/started.json';rt=BASE/'e1_independent_review/monitor_e4panel003/terminal.json';verifier=BASE/'e4_panel_v2/panel_verify.py';reviewer=BASE/'e1_independent_review/review_e4panel.py'
 gate.terminal(fs,ft,verifier,formal);gate.terminal(rs,rt,reviewer,review);f=gate.read(formal);r=gate.read(review)
 if (f.get('status')!='passed' or f.get('stage')!='full' or f.get('n_questions')!=384 or f.get('all_arms_validated')!=3072 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('verifier_sha256')!=gate.sha(verifier) or r.get('status')!='passed' or r.get('n_verified')!=384 or r.get('arms_verified')!=3072 or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('formal_proof_sha256')!=gate.sha(formal) or r.get('formal_verifier_started_sha256')!=gate.sha(fs) or r.get('formal_verifier_terminal_sha256')!=gate.sha(ft) or r.get('reviewer_sha256')!=gate.sha(reviewer) or f['files_sha256'].get(str(records.resolve()))!=gate.sha(records)):
  raise ValueError('E4 comparison proof or exit invalid')
 for path,digest in f['files_sha256'].items():
  if gate.sha(path)!=digest:raise ValueError('E4 artifact changed')
 return formal,records

def main():
 p=argparse.ArgumentParser()
 for name in ('run','run-started','run-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();budget=gate.read(HERE/'resource_plan.json');launch=gate.read(run/'launch.json');exit=gate.read(run/'exit.json');smoke=actual_exit(run,gate.read(a.run_started),gate.read(a.run_terminal),launch,exit,budget)
 approval=gate.execution_gate();up=gate.upstream();frozen=gate.read(run/'frozen.json')
 if frozen.get('approval')!=approval or launch.get('approval')!=approval or frozen.get('upstream')!=up or frozen.get('GT_opened') is not False or frozen.get('scoring_performed') is not False:raise ValueError('run frozen admission')
 for pth,h in frozen['source_sha256'].items():
  if gate.sha(pth)!=h:raise ValueError('frozen code changed')
 for path,expected in frozen['weight_stats_before'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=expected:raise ValueError('weights changed')
 plan=gate.read(HERE/'sample_001.json');rows=plan['sample'];e1=gate.read(BASE/'e1/e1full001_completion.json');refs={(r['task'],str(r['qid'])):r for r in readlines(e1['files']['journal']['path']) if r.get('forced') is False and r.get('arm')=='P3-D1'}
 records=readlines(run/'data/records.jsonl');events=readlines(run/'data/generated.jsonl')
 if len(records)!=96 or len(events)!=96 or gate.read(run/'data/worker_complete.json')!={'questions':32,'natural_generations':96,'GT_opened':False,'scoring_performed':False} or gate.read(run/'worker_complete.json')!=gate.read(run/'data/worker_complete.json'):raise ValueError('full denominator')
 from transformers import AutoTokenizer
 recipe=gate.read(BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['recipe'];tok=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 anchors={next((r['task'],str(r['qid'])) for r in rows if r['task']==t) for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')};array_paths={};groups={}
 for arm_index,arm in enumerate(ARMS):
  for i,row in enumerate(rows):
   r=records[arm_index*32+i];event=events[arm_index*32+i];key=(row['task'],str(row['qid']))
   if event!={k:r[k] for k in EVENT}:raise ValueError('exposure log differs')
   validate(r,row,arm,refs[key]);validate_trace(r)
   if tok.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:raise ValueError('token decoding')
   arc=r['arrays'];groups.setdefault(key,{})[arm]=r
   if key in anchors:
    path=Path(arc['path']).resolve()
    if arc['kind']!='full_arrays' or path.parent!=(run/'data/arrays').resolve() or path in array_paths or gate.sha(path)!=arc['sha256']:raise ValueError('array path/hash')
    array_paths[path]=arc['sha256'];t=r['trace'];converted={'arrays':arc,'case':{'stock_native':'stock','author_default':'author_default','priortr_k256':'k256'}[arm],'calls':t['calls'],'layers':t['layers'],'injections':t['injections'],'prunes':t['prunes'],'ids':r['output_ids'],'forced':False,'eos':r['eos'],'eos_ids':r['eos_ids'],'truncated':r['truncated']}
    with np.load(path,allow_pickle=False) as arrays:smoke.validate_record(converted,arrays,refs[key]['processor'],36)
   elif arc['kind']!='summary_only':raise ValueError('non-anchor archive')
 if len(array_paths)!=9 or set(array_paths)!={x.resolve() for x in (run/'data/arrays').glob('*.npz')}:raise ValueError('array inventory')
 for key,byarm in groups.items():
  baseline=byarm['stock_native']['arrays']['tensors']
  for arm in ('author_default','priortr_k256'):
   other=byarm[arm]['arrays']['tensors']
   for name in ('full_visual','original_prefill_position','original_prefill_embeds','original_visual_mask','full_ds_0','full_ds_1','full_ds_2'):
    if baseline[name]!=other[name]:raise ValueError('pre-prune inputs differ between arms')
 # Cross-backend output comparison is a diagnostic, no quality scores.
 e4p,e4rec=e4_comparison_gate()
 full={(r['task'],str(r['qid'])):r for r in readlines(e4rec) if r.get('budget')==1024 and r.get('ds_on') is True}
 if len(full)!=384 or any(k not in full for k in groups):raise ValueError('E4 full reference inventory')
 same=sum(groups[k]['stock_native']['output_ids']==full[k]['output_ids'] for k in groups)
 files=[a.run_started,a.run_terminal,run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'data/worker_complete.json',run/'data/records.jsonl',run/'data/generated.jsonl',e4p,e4rec]
 result={'status':'passed','questions':32,'generations':96,'GT_opened':False,'scoring_performed':False,'stock_same_output_as_e4_full':same,'stock_compared':32,'cross_backend_first_logits':'unavailable for frozen 32 samples','files_sha256':{str(p.resolve()):gate.sha(p) for p in files},'array_sha256':{str(k):v for k,v in array_paths.items()},'verifier_sha256':gate.sha(__file__),'limits':['DEV diagnostic only; no winner selection.','Only nine anchor arrays retained; non-anchor tensors cannot be reconstructed.','PriorTR uses SDPA and a different pinned environment than E4 FA2; not a latency comparison.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
