"""Two-phase source-bound admission for multistream GPU v2."""
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or len(c)<2 or (cwd/c[1]).resolve()!=Path(script).resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=Path(out).resolve()))):raise ValueError('supervisor mismatch')
def admit(preflight_only,host=False):
 cpu=D/'cpu_check001.json';terminal(D/'monitor_cpu_check001',D/'cpu_check.py');c=read(cpu)
 if c.get('status')!='passed' or c.get('GPU_used') is not False or c.get('GT_opened') is not False or c.get('source_sha256_before')!=c.get('source_sha256_after') or any(sha(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('CPU source freeze')
 name='e4_multistream_expansion352_v2_'+('preflight_review' if preflight_only else 'gpu_review');proof=I/(name+'.json');reviewer=I/('review_'+name+'.py');monitor=I/('monitor_'+name)
 if not proof.is_file():raise ValueError('independent review missing')
 terminal(monitor,reviewer,proof);r=read(proof)
 fields={'reviewer_sha256':reviewer,'cpu_sha256':cpu,'cpu_started_sha256':D/'monitor_cpu_check001/started.json','cpu_terminal_sha256':D/'monitor_cpu_check001/terminal.json','selection_sha256':P/'selection001.json'}
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('source_sha256')!=c['source_sha256_after'] or any(r.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('independent review mismatch')
 if not preflight_only:
  prev=admit(True,host=host);run=D/'preflight001';terminal(D/'monitor_preflight001',D/'launch.py');la=read(run/'launch.json');ex=read(run/'exit.json');w=read(run/'worker_preflight.json')
  fields={'preflight_sha256':run/'worker_preflight.json','preflight_launch_sha256':run/'launch.json','preflight_exit_sha256':run/'exit.json','preflight_started_sha256':D/'monitor_preflight001/started.json','preflight_terminal_sha256':D/'monitor_preflight001/terminal.json','preflight_review_sha256':I/'e4_multistream_expansion352_v2_preflight_review.json','formal_plan_sha256':D/'formal_plan001.json'}
  if (any(r.get(k)!=sha(v) for k,v in fields.items()) or la.get('approval_bundle')!=prev or la.get('gpu_requested') is not False or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0 or ex.get('failure_type') is not None or w.get('status')!='passed' or w.get('pilot_images')!=9 or w.get('processor_inputs_checked')!=9 or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('GT_opened') is not False or w.get('source_sha256')!=c['source_sha256_after']):raise ValueError('v2 actual preflight mismatch')
  if host:
   obj=json.loads(subprocess.run(['docker','inspect',la['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=obj['State'];devices=obj['HostConfig']
   if (obj['Id']!=ex['container_identity']['container_id'] or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False or (devices.get('Devices') or [])!=[] or devices.get('DeviceRequests') not in (None,[])):raise ValueError('real preflight container')
 return {'preflight_only':preflight_only,'source_sha256':c['source_sha256_after'],'review_sha256':sha(proof),'selection_sha256':sha(P/'selection001.json')}
