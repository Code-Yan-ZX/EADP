"""Fail-closed admission for no-GPU runtime container preflight only."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parent;R=B/'e1_independent_review';OLD=B/'e4_c_memory'
OLDPROOF=R/'e4c_memory_preflight001_review.json';OLDMON=R/'monitor_e4c_memory_preflight001';OLDREVIEWER=R/'review_e4c_memory_preflight001.py'
CPU=HERE/'cpu_check001/verification.json';CPUMON=HERE/'monitor_runtime_cpu001';REVIEW=R/'e4c_memory_runtime_preflight_review.json';REVIEWER=R/'review_e4c_memory_runtime_preflight.py';REVIEWMON=R/'monitor_e4c_memory_runtime_preflight'
NAMES=('bridge.py','controller_adapter.py','controller_io.py','docker_guard.py','interruptible.py','lifecycle.py','sampler_entry.py','sampler_supervisor.py','worker_barrier.py','worker_candidate.py','runtime_admission.py','runtime_preflight_entry.py','runtime_preflight_launch.py','runtime_cpu_check.py','test_bridge.py','test_controller_adapter.py','test_controller_io.py','test_docker_guard.py','test_interruptible.py','test_lifecycle.py','test_protocol.py','test_sampler_entry.py','test_sampler_supervisor.py','test_supervisor_process.py','test_worker_barrier.py','test_worker_candidate.py','test_runtime_admission.py','test_runtime_entry.py','preflight_plan.json')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervisor(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('review supervision identity')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('review supervision output')
 return s,t
def sources():
 previous=read(OLDPROOF).get('approval',{}).get('source_sha256',{})
 if not isinstance(previous,dict) or len(previous)!=59:raise ValueError('previous source inventory')
 return [HERE/x for x in NAMES]+[Path(x) for x in previous]
def source_hashes():return {str(x.resolve()):sha(x) for x in sources()}
def prior():
 supervisor(OLDMON,OLDREVIEWER,OLDPROOF);r=read(OLDPROOF)
 if (r.get('status'),r.get('independent_review'),r.get('GT_opened'),r.get('scoring_performed'),r.get('model_loaded'),r.get('gpu_used'))!=('passed',True,False,False,False,False) or r.get('reviewer_sha256')!=sha(OLDREVIEWER):raise ValueError('prior memory preflight review')
 for field,path in [('started_sha256',OLD/'monitor_container_preflight001/started.json'),('terminal_sha256',OLD/'monitor_container_preflight001/terminal.json'),('launch_sha256',OLD/'container_preflight001/launch.json'),('exit_sha256',OLD/'container_preflight001/exit.json'),('worker_preflight_sha256',OLD/'container_preflight001/worker_preflight.json')]:
  if r.get(field)!=sha(path):raise ValueError('prior preflight evidence changed')
 source=r.get('approval',{}).get('source_sha256',{})
 if len(source)!=59 or any(sha(p)!=h for p,h in source.items()):raise ValueError('prior memory source changed')
 return r
def admit():
 for p in (CPU,CPUMON/'started.json',CPUMON/'terminal.json',REVIEW,REVIEWER,REVIEWMON/'started.json',REVIEWMON/'terminal.json'):
  if not p.is_file():raise ValueError('independent runtime preflight review required')
 prior();supervisor(CPUMON,HERE/'runtime_cpu_check.py');supervisor(REVIEWMON,REVIEWER,REVIEW)
 c=read(CPU);r=read(REVIEW);actual=source_hashes()
 if c.get('status')!='passed' or c.get('gpu_used') is not False or c.get('model_loaded') is not False or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual:raise ValueError('runtime CPU proof mismatch')
 fields={'status':'passed','independent_review':True,'scope':'memory_runtime_cpu_preflight_only','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'errors':[],'source_sha256':actual,'cpu_proof_sha256':sha(CPU),'cpu_started_sha256':sha(CPUMON/'started.json'),'cpu_terminal_sha256':sha(CPUMON/'terminal.json'),'prior_review_sha256':sha(OLDPROOF),'reviewer_sha256':sha(REVIEWER)}
 if any(r.get(k)!=v for k,v in fields.items()):raise ValueError('independent runtime preflight proof mismatch')
 return {'scope':'memory_runtime_cpu_preflight_only','gpu_authorized':False,'source_sha256':actual,'cpu_sha256':sha(CPU),'review_sha256':sha(REVIEW),'prior_review_sha256':sha(OLDPROOF)}
