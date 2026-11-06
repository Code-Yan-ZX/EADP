"""Fail-closed review and actual preflight boundary; never opens labels."""
import hashlib,json,time
from pathlib import Path
from freeze_plan import sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
APPROVAL=HERE/'approval.json'
def read(path):return json.loads(Path(path).read_text())
def supervised(started,terminal,script,out):
 s=read(started);t=read(terminal);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or
     cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):
  raise ValueError('supervisor exit/command mismatch')
def budget(require_future):
 p=read(HERE/'plan.json');initial=ROOT/'initial_plan.json';orig=read(initial)
 if (sha(initial)!=p['budget_source_sha256'] or orig['budget_end_unix']!=p['budget_end_unix'] or
     p['budget_end_unix']!=1794065414 or time.time()>=p['budget_end_unix'] or
     (require_future and time.time()+p['outer_timeout_seconds']>p['budget_end_unix']) or
     p['inner_timeout_seconds']!=7200 or p['outer_timeout_seconds']!=7380 or p['snapshot_seconds']!=3600):
  raise ValueError('deadline or bound invalid')
 return p
def check(*,preflight_only=False,require_future=False):
 budget(require_future)
 if not APPROVAL.exists():raise ValueError('selected smoke locked: independent approval absent')
 a=read(APPROVAL);needed={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
 if not preflight_only:needed|={'preflight_proof','preflight_started','preflight_terminal'}
 if set(a)!=needed:raise ValueError('approval inventory mismatch')
 cpu=Path(a['cpu_proof']);supervised(a['cpu_started'],a['cpu_terminal'],HERE/'cpu_validation.py',cpu.parent);c=read(cpu)
 src=c.get('source_sha256_after',{})
 if (c.get('status')!='passed' or c.get('source_sha256_before')!=src or c.get('GT_opened') is not False or
     c.get('real_model_loaded') is not False or any(sha(path)!=value for path,value in src.items()) or
     any(src.get(str(path.resolve()))!=sha(path) for path in list(HERE.glob('*.py'))+[HERE/'plan.json'])):
  raise ValueError('CPU proof or sources changed')
 reviewer=ROOT/'e1_independent_review'/('review_selected_preflight.py' if preflight_only else 'review_selected_final.py')
 review=Path(a['review_proof']);expected=HERE/('independent_preflight_review.json' if preflight_only else 'independent_final_review.json')
 if review.resolve()!=expected.resolve():raise ValueError('wrong independent reviewer')
 supervised(a['review_started'],a['review_terminal'],reviewer,review);r=read(review)
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or
     r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=src or r.get('cpu_proof_sha256')!=sha(cpu) or
     r.get('cpu_started_sha256')!=sha(a['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
     r.get('GT_opened') is not False or r.get('gpu_used') is not False):raise ValueError('independent review mismatch')
 manifest=hashlib.sha256(json.dumps(src,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 result={'approval_sha256':sha(APPROVAL),'plan_sha256':sha(HERE/'plan.json'),'cpu_sha256':sha(cpu),'review_sha256':sha(review),
         'source_count_verified':len(src),'source_manifest_sha256':manifest}
 if not preflight_only:
  pp=Path(a['preflight_proof']);pre=read(pp);started=read(a['preflight_started']);terminal=read(a['preflight_terminal'])
  run=pp.stem.removeprefix('preflight_');directory=HERE/('preflight_'+run);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
      terminal.get('worker_pid')!=started.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']!=[(HERE/'preflight.py').resolve()] or
      cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=run or pre.get('status')!='passed' or
      pre.get('preflight_sha256')!=sha(HERE/'preflight.py') or pre.get('cpu_proof_sha256')!=sha(cpu) or
      pre.get('launch_sha256')!=sha(directory/'launch.json') or pre.get('exit_sha256')!=sha(directory/'exit.json') or
      pre.get('log_sha256')!=sha(directory/'launcher.log') or pre.get('receipt_sha256')!=sha(directory/'host_receipt.json') or
      pre.get('container_running') is not False or pre.get('container_pid')!=0 or pre.get('container_exit_code')!=0 or
      type(pre.get('actual_mount_count_verified')) is not int or pre['actual_mount_count_verified']<=0 or
      pre.get('result',{}).get('GT_opened') is not False or pre.get('result',{}).get('predictions_opened') is not False or
      pre.get('result',{}).get('cuda_initialized') is not False or pre.get('result',{}).get('model_loaded') is not False or
      pre.get('result',{}).get('anchors_verified')!=3 or pre.get('result',{}).get('weights_sha256_verified')!=4 or
      pre.get('result',{}).get('source_count_verified')!=len(src) or pre.get('result',{}).get('source_manifest_sha256')!=manifest or
      r.get('preflight_sha256')!=sha(pp) or r.get('preflight_started_sha256')!=sha(a['preflight_started']) or
      r.get('preflight_terminal_sha256')!=sha(a['preflight_terminal'])):raise ValueError('container preflight or review incomplete')
  ex=read(directory/'exit.json');launch=read(directory/'launch.json');docker=launch.get('command',[])
  if (ex!=pre.get('exit') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or
      ex.get('container_stopped_verified') is not True or ex.get('interrupted') is not None or ex.get('timed_out') is not False or
      ex.get('errors')!=[] or ex.get('client_exited_while_container_running') is not False or
      launch.get('host_receipt')!={'path':str((directory/'host_receipt.json').resolve()),'sha256':sha(directory/'host_receipt.json')} or
      launch.get('cpu_proof_sha256')!=sha(cpu) or launch.get('worker_sha256')!=sha(HERE/'worker.py') or
      docker.count(str(HERE/'worker.py'))!=1 or docker.count('--preflight-only')!=1 or 'CUDA_VISIBLE_DEVICES=' not in docker or
      any('/dev/nvidia' in str(x) for x in docker)):
   raise ValueError('preflight actual worker/exit evidence mismatch')
  result['preflight_sha256']=sha(pp)
 return result
