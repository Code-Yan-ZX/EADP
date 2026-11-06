"""Source and explicit independent review gate; no model or data access."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if not (t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]):raise ValueError('supervisor/source mismatch')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output mismatch')
 return s
def verify_plan():
 p=read(D/'plan001.json');src=p.get('source_sha256')
 if p.get('scope')!='minimal real CUDA allocation and host NVML handshake' or p.get('allocation_bytes')!=16*1024*1024 or p.get('hold_seconds')!=8 or not isinstance(src,dict) or not src or any(sha(x)!=h for x,h in src.items()):raise ValueError('plan/source mismatch')
 return p
def verify_review():
 p=verify_plan();cpu=D/'cpu001.json';terminal(D/'monitor_cpu001',D/'cpu_check.py',cpu);c=read(cpu)
 if c.get('status')!='passed' or c.get('plan_sha256')!=sha(D/'plan001.json'):raise ValueError('CPU proof mismatch')
 pre=D/'preflight001.json';terminal(D/'monitor_preflight001',D/'preflight_launch.py');f=read(pre);ex=read(D/'run_preflight001/exit.json')
 if f.get('status')!='passed' or f.get('cuda_initialized') is not False or f.get('gpu_used') is not False or f.get('model_loaded') is not False or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0 or ex.get('returncode')!=0 or ex.get('last_inspect',{}).get('host_devices')!=[]:raise ValueError('preflight mismatch')
 extra=D/'container_cpu001';terminal(D/'monitor_container_cpu001',D/'container_cpu_regression.py');e=read(extra/'exit.json');l=read(extra/'launch.json')
 if e.get('container_stopped_verified') is not True or e.get('container_exit_code')!=0 or e.get('returncode')!=0 or e.get('last_inspect',{}).get('host_devices')!=[] or l.get('probe_sha256')!=sha(D/'container_cpu_regression.py') or l.get('plan_sha256')!=sha(D/'plan001.json'):raise ValueError('container CPU import regression mismatch')
 review=R/'e4c_memory_real_handshake_v2_prelaunch_review.json';reviewer=R/'review_e4c_memory_real_handshake_v2_prelaunch.py';mon=R/'monitor_e4c_memory_real_handshake_v2_prelaunch';terminal(mon,reviewer,review);r=read(review)
 expected={'status':'passed','independent_review':True,'scope':'minimal real GPU memory handshake prelaunch','GT_opened':False,'model_loaded':False,'errors':[],'plan_sha256':sha(D/'plan001.json'),'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'preflight_sha256':sha(pre),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_sha256':sha(reviewer),'container_cpu_started_sha256':sha(D/'monitor_container_cpu001/started.json'),'container_cpu_terminal_sha256':sha(D/'monitor_container_cpu001/terminal.json'),'container_cpu_launch_sha256':sha(extra/'launch.json'),'container_cpu_exit_sha256':sha(extra/'exit.json'),'container_cpu_log_sha256':sha(extra/'docker.log'),'container_cpu_probe_sha256':sha(D/'container_cpu_regression.py')}
 if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('independent review mismatch')
 return {'plan_sha256':sha(D/'plan001.json'),'review_sha256':sha(review),'review_terminal_sha256':sha(mon/'terminal.json')}
