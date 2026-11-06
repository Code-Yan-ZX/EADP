"""Reproduce failed preflight image/mount visibility with a bounded read-only probe."""
import json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];run=D/'run001';run.mkdir(exist_ok=False)
old=json.loads((B/'e6/rres_k256_pilot/run_preflight001/launch.json').read_text());cmd=old['command'][:];name='eadp-rres-path-probe-'+str(os.getpid());cmd[cmd.index('--name')+1]=name;idx=cmd.index(str(B/'e6/rres_k256_pilot/preflight_worker.py'));cmd=cmd[:idx]+[str(D/'worker.py')]
if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('unexpected device')
with (run/'launch.json').open('x') as f:json.dump({'command':cmd,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
try:
 with (run/'docker.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90);rc=r.returncode
except subprocess.TimeoutExpired:
 subprocess.run(['docker','stop','--time','2',name],capture_output=True,timeout=10);subprocess.run(['docker','kill',name],capture_output=True,timeout=10);raise
finally:
 i=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
 if i.returncode==0:
  obj=json.loads(i.stdout)[0];state=obj['State'];payload={'container_id':obj['Id'],'image':obj['Image'],'running':state['Running'],'pid':state['Pid'],'exit_code':state['ExitCode'],'oom':state['OOMKilled'],'devices':obj['HostConfig']['Devices'],'cmd':obj['Config']['Cmd']}
  with (run/'inspect.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if rc or payload['running'] or payload['pid']!=0 or payload['exit_code']!=0 or payload['oom'] or payload['devices']:raise ValueError('path probe failed')
