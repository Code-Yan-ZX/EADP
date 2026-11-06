"""Batch-2 admission; defaults locked until independent CPU and container review."""
import hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;APPROVAL=HERE/'runtime_approval.json'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervised(start,end,script,out):
 s,t=read(start),read(end);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()):raise ValueError('batch-2 supervisor/proof mismatch')
def budget(plan,now,require_future=True):
 initial=read(ROOT/'initial_plan.json')
 if (plan['budget_start_unix']!=initial['budget_start_unix'] or plan['budget_end_unix']!=initial['budget_end_unix'] or initial['budget_end_unix']!=1794065414 or
     plan['source_sha256'].get(str(ROOT/'initial_plan.json'))!=sha(ROOT/'initial_plan.json') or now<plan['budget_start_unix'] or now>=plan['budget_end_unix'] or
     (require_future and now+plan['outer_timeout_seconds']>plan['budget_end_unix']) or
     (plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['snapshot_seconds'],plan['progress_stale_seconds'],plan['question_checkpoint_seconds'])!=(7200,7380,3600,2400,1800) or
     (plan['questions'],plan['new_generations_per_question'],plan['maximum_new_generations'])!=(128,6,768)):
  raise ValueError('batch-2 deadline or bounds invalid')
def validate(*,host=False,require_future=True,preflight_only=False):
 if not APPROVAL.is_file():raise ValueError('batch-2 locked: independent CPU review and actual container preflight required')
 a=read(APPROVAL);keys={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
 if not preflight_only:keys|={'preflight_proof','preflight_started','preflight_terminal'}
 if set(a)!=keys:raise ValueError('batch-2 approval inventory')
 planpath=HERE/'execution_plan_002.json';plan=read(planpath);budget(plan,time.time(),require_future)
 if any(sha(path)!=value for path,value in plan['source_sha256'].items()) or sha(plan['sample_path'])!=plan['sample_sha256']:raise ValueError('batch-2 frozen plan/source changed')
 cpu=Path(a['cpu_proof']);c=read(cpu);supervised(a['cpu_started'],a['cpu_terminal'],HERE/'cpu_validation.py',cpu.parent);src=c.get('source_sha256_after',{})
 if (c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('predictions_opened') is not False or c.get('real_model_loaded') is not False or
     c.get('source_sha256_before')!=src or any(sha(p)!=d for p,d in src.items()) or any(src.get(str(p.resolve()))!=sha(p) for p in list(HERE.glob('*.py'))+[planpath])):
  raise ValueError('batch-2 CPU sources changed')
 review=Path(a['review_proof']);expected=HERE/('independent_preflight_review.json' if preflight_only else 'independent_final_review.json');reviewer=ROOT/'e1_independent_review'/('review_m3batch2_preflight.py' if preflight_only else 'review_m3batch2_final.py')
 if review.resolve()!=expected.resolve():raise ValueError('wrong batch-2 independent review')
 supervised(a['review_started'],a['review_terminal'],reviewer,review);r=read(review)
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or
     r.get('cpu_proof_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(a['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
     r.get('plan_sha256')!=sha(planpath) or r.get('source_sha256')!=src or r.get('GT_opened') is not False or r.get('gpu_used') is not False):raise ValueError('batch-2 independent review mismatch')
 result={'approval_sha256':sha(APPROVAL),'plan_sha256':sha(planpath),'cpu_sha256':sha(cpu),'review_sha256':sha(review),'source_count_verified':len(src),'source_manifest_sha256':hashlib.sha256(json.dumps(src,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
 if preflight_only:return result
 pp=Path(a['preflight_proof']);p=read(pp);s=read(a['preflight_started']);t=read(a['preflight_terminal']);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'));rid=pp.stem.removeprefix('preflight_');d=HERE/('preflight_'+rid)
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']!=[(HERE/'preflight.py').resolve()] or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=rid or
     p.get('status')!='passed' or p.get('container_config_verified') is not True or p.get('preflight_sha256')!=sha(HERE/'preflight.py') or p.get('cpu_proof_sha256')!=sha(cpu) or
     any(p.get(k)!=sha(d/f) for k,f in (('launch_sha256','launch.json'),('exit_sha256','exit.json'),('log_sha256','launcher.log'),('receipt_sha256','host_receipt.json'),('approval_snapshot_sha256','approval_snapshot.json'))) or
     p.get('container_running') is not False or p.get('container_pid')!=0 or p.get('container_exit_code')!=0 or p.get('result',{}).get('GT_opened') is not False or
     p.get('result',{}).get('predictions_opened') is not False or p.get('result',{}).get('cuda_initialized') is not False or p.get('result',{}).get('model_loaded') is not False or
     p.get('result',{}).get('inputs_verified')!=384 or p.get('result',{}).get('sample_verified')!=128 or p.get('result',{}).get('weights_sha256_verified')!=4 or
     p.get('result',{}).get('source_count_verified')!=len(src) or p.get('result',{}).get('source_manifest_sha256')!=result['source_manifest_sha256'] or
     r.get('preflight_sha256')!=sha(pp) or r.get('preflight_started_sha256')!=sha(a['preflight_started']) or r.get('preflight_terminal_sha256')!=sha(a['preflight_terminal'])):raise ValueError('batch-2 actual preflight incomplete')
 ex=read(d/'exit.json');launch=read(d/'launch.json');docker=launch.get('command',[])
 previous=read(d/'approval_snapshot.json');receipt=read(d/'host_receipt.json')
 if set(previous)!={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'} or any(previous[k]!=a[k] for k in ('cpu_proof','cpu_started','cpu_terminal')):raise ValueError('batch-2 preflight CPU binding mismatch')
 provisional=HERE/'independent_preflight_review.json';provisional_reviewer=ROOT/'e1_independent_review/review_m3batch2_preflight.py'
 if Path(previous['review_proof']).resolve()!=provisional.resolve():raise ValueError('batch-2 wrong provisional review')
 supervised(previous['review_started'],previous['review_terminal'],provisional_reviewer,provisional);pr=read(provisional)
 if (pr.get('status')!='passed' or pr.get('independent_review') is not True or pr.get('preflight_only') is not True or pr.get('reviewer_sha256')!=sha(provisional_reviewer) or
     pr.get('cpu_proof_sha256')!=sha(cpu) or pr.get('cpu_started_sha256')!=sha(a['cpu_started']) or pr.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
     pr.get('plan_sha256')!=sha(planpath) or pr.get('source_sha256')!=src or pr.get('GT_opened') is not False or pr.get('gpu_used') is not False or
     receipt.get('approval_sha256')!=sha(d/'approval_snapshot.json') or receipt.get('review_sha256')!=sha(provisional) or
     receipt.get('cpu_sha256')!=sha(cpu) or receipt.get('plan_sha256')!=sha(planpath) or receipt.get('source_count_verified')!=len(src) or receipt.get('source_manifest_sha256')!=result['source_manifest_sha256']):raise ValueError('batch-2 provisional preflight proof mismatch')
 if (ex!=p.get('exit') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('interrupted') is not None or ex.get('timed_out') is not False or ex.get('errors')!=[] or ex.get('client_exited_while_container_running') is not False or
     launch.get('host_receipt')!={'path':str((d/'host_receipt.json').resolve()),'sha256':sha(d/'host_receipt.json')} or launch.get('cpu_proof_sha256')!=sha(cpu) or launch.get('worker_sha256')!=sha(HERE/'runtime_worker.py') or
     docker.count(str(HERE/'runtime_worker.py'))!=1 or docker.count('--preflight-only')!=1 or 'CUDA_VISIBLE_DEVICES=' not in docker or any('/dev/nvidia' in str(x) for x in docker)):raise ValueError('batch-2 preflight launch/exit mismatch')
 if host:
  q=subprocess.run(['docker','inspect',p['container_id']],capture_output=True,text=True,timeout=20,check=True);v=json.loads(q.stdout)[0];state=v['State']
  if v['Id']!=p['container_id'] or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0:raise ValueError('batch-2 preflight container not stopped')
  import importlib.util
  spec=importlib.util.spec_from_file_location('_m3_batch2_preflight_host_verify',HERE/'preflight.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  if not mod.actual_mounts_and_env_match(docker,v):raise ValueError('batch-2 actual preflight mounts or environment changed')
 result['preflight_sha256']=sha(pp);return result
