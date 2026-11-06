"""Fail-closed full DEV admission; independent CPU/review/preflight required."""
import hashlib,json,subprocess,time
from pathlib import Path
from full_imports import HERE,SMOKE,load
_plan=load('eadp_selected_full_plan',HERE/'freeze_plan.py');ROOT=_plan.ROOT;REVIEW=_plan.REVIEW;sha=_plan.sha;build=_plan.build
APPROVAL=HERE/'approval.json'
def read(path):return json.loads(Path(path).read_text())
def supervised(started,terminal,script,out):
 s=read(started);t=read(terminal);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or
     cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()):raise ValueError('supervised proof identity')
def upstream(*,host=False):
 plan=read(HERE/'plan.json')
 if plan!=build():raise ValueError('full plan or upstream changed')
 formal=SMOKE/'smoke_001_verification.json';review=REVIEW/'selected_smoke001_review.json'
 fs=SMOKE/'monitor_verify001/started.json';ft=SMOKE/'monitor_verify001/terminal.json'
 rs=REVIEW/'monitor_selected_smoke001/started.json';rt=REVIEW/'monitor_selected_smoke001/terminal.json'
 reviewer=REVIEW/'review_selected_smoke.py';supervised(fs,ft,SMOKE/'verify.py',formal);supervised(rs,rt,reviewer,review)
 p=read(formal);r=read(review);run=SMOKE/'smoke_001';manifest=SMOKE/'smoke_001_sources/manifest.json'
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=3 or r.get('generations')!=16 or
     r.get('pairs')!=8 or r.get('forced_decode_steps')!=32 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or
     r.get('errors')!=[] or r.get('reviewer_sha256')!=sha(reviewer) or r.get('formal_proof_sha256')!=sha(formal) or
     r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or
     r.get('run_started_sha256')!=sha(SMOKE/'monitor_smoke001/started.json') or r.get('run_terminal_sha256')!=sha(SMOKE/'monitor_smoke001/terminal.json') or
     r.get('source_manifest_sha256')!=sha(manifest) or p.get('verifier_sha256')!=sha(SMOKE/'verify.py')):
  raise ValueError('independent Div/CD identity evidence changed')
 for source,digest in read(manifest)['source_sha256'].items():
  if sha(source)!=digest:raise ValueError('identity source changed: '+source)
 if host:
  ex=read(run/'exit.json');cid=(ex.get('container_identity') or {}).get('container_id')
  q=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=20,check=True);v=json.loads(q.stdout)[0];state=v['State']
  if v['Id']!=cid or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0:raise ValueError('identity smoke container not stopped')
 return {'smoke_proof_sha256':sha(formal),'smoke_review_sha256':sha(review),'smoke_source_manifest_sha256':sha(manifest),
         'plan_sha256':sha(HERE/'plan.json')}
def check(*,preflight_only=False,require_future=False,host=False):
 plan=read(HERE/'plan.json');now=time.time()
 if (now>=plan['budget_end_unix'] or (require_future and now+plan['outer_timeout_seconds']>plan['budget_end_unix']) or
     plan['questions']!=384 or plan['generation_count']!=1536 or plan['inner_timeout_seconds']!=10800 or
     plan['outer_timeout_seconds']!=10980 or plan['snapshot_seconds']!=3600):raise ValueError('full budget/deadline')
 if not APPROVAL.exists():raise ValueError('full panel locked: independent approval absent')
 a=read(APPROVAL);keys={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
 if not preflight_only:keys|={'preflight_proof','preflight_started','preflight_terminal'}
 if set(a)!=keys:raise ValueError('full approval fields')
 cpu=Path(a['cpu_proof']);supervised(a['cpu_started'],a['cpu_terminal'],HERE/'cpu_validation.py',cpu.parent);c=read(cpu);src=c['source_sha256_after']
 if (c.get('status')!='passed' or c.get('source_sha256_before')!=src or c.get('GT_opened') is not False or
     c.get('real_model_loaded') is not False or any(sha(p)!=d for p,d in src.items()) or
     any(src.get(str(p.resolve()))!=sha(p) for p in list(HERE.glob('*.py'))+[HERE/'plan.json'])):raise ValueError('CPU evidence changed')
 review=Path(a['review_proof']);reviewer=REVIEW/('review_selected_full_preflight.py' if preflight_only else 'review_selected_full.py')
 expected=HERE/('independent_preflight_review.json' if preflight_only else 'independent_final_review.json')
 if review.resolve()!=expected.resolve():raise ValueError('wrong full reviewer')
 supervised(a['review_started'],a['review_terminal'],reviewer,review);r=read(review)
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or
     r.get('reviewer_sha256')!=sha(reviewer) or r.get('cpu_proof_sha256')!=sha(cpu) or r.get('source_sha256')!=src or
     r.get('cpu_started_sha256')!=sha(a['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
     r.get('GT_opened') is not False or r.get('gpu_used') is not False):raise ValueError('full review mismatch')
 u=upstream(host=host);manifest=hashlib.sha256(json.dumps(src,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 result={'approval_sha256':sha(APPROVAL),'cpu_sha256':sha(cpu),'review_sha256':sha(review),
         'source_count_verified':len(src),'source_manifest_sha256':manifest,'upstream':u}
 if not preflight_only:
  pp=Path(a['preflight_proof']);pre=read(pp);st=read(a['preflight_started']);te=read(a['preflight_terminal']);cmd=st.get('command',[]);cwd=Path(st.get('cwd','/'))
  run=pp.stem.removeprefix('preflight_');directory=HERE/('preflight_'+run)
  if (te.get('reason')!='completed' or te.get('worker_returncode')!=0 or te.get('remaining_process_group')!=[] or te.get('worker_pid')!=st.get('worker_pid') or
      [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']!=[(HERE/'preflight.py').resolve()] or
      cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=run or pre.get('status')!='passed' or
      pre.get('preflight_sha256')!=sha(HERE/'preflight.py') or pre.get('cpu_proof_sha256')!=sha(cpu) or
      any(pre.get(k)!=sha(directory/f) for k,f in (('launch_sha256','launch.json'),('exit_sha256','exit.json'),('log_sha256','launcher.log'),('receipt_sha256','host_receipt.json'))) or
      pre.get('container_running') is not False or pre.get('container_pid')!=0 or pre.get('container_exit_code')!=0 or
      type(pre.get('actual_mount_count_verified')) is not int or pre['actual_mount_count_verified']<=0 or
      any(pre.get('result',{}).get(k) is not False for k in ('GT_opened','predictions_opened','cuda_initialized','model_loaded')) or
      pre.get('result',{}).get('inputs_verified')!=384 or pre.get('result',{}).get('weights_sha256_verified')!=4 or
      pre.get('result',{}).get('source_count_verified')!=len(src) or pre.get('result',{}).get('source_manifest_sha256')!=manifest or
      r.get('preflight_sha256')!=sha(pp) or r.get('preflight_started_sha256')!=sha(a['preflight_started']) or
      r.get('preflight_terminal_sha256')!=sha(a['preflight_terminal'])):raise ValueError('full preflight incomplete')
  ex=read(directory/'exit.json');launch=read(directory/'launch.json');docker=launch.get('command',[])
  if (ex!=pre.get('exit') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or
      ex.get('interrupted') is not None or ex.get('timed_out') is not False or ex.get('errors')!=[] or
      ex.get('client_exited_while_container_running') is not False or launch.get('worker_sha256')!=sha(HERE/'worker.py') or
      launch.get('cpu_proof_sha256')!=sha(cpu) or launch.get('host_receipt')!={'path':str((directory/'host_receipt.json').resolve()),'sha256':sha(directory/'host_receipt.json')} or
      docker.count(str(HERE/'worker.py'))!=1 or docker.count('--preflight-only')!=1 or 'CUDA_VISIBLE_DEVICES=' not in docker or
      any('/dev/nvidia' in str(x) for x in docker)):raise ValueError('full preflight execution mismatch')
  if host:
   cid=pre['container_id'];q=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=20,check=True);v=json.loads(q.stdout)[0]
   if v['Id']!=cid or v['State'].get('Running') is not False or v['State'].get('Pid')!=0 or v['State'].get('ExitCode')!=0:raise ValueError('full preflight Docker state')
  result['preflight_sha256']=sha(pp)
 return result
