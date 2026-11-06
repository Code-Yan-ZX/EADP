"""Fail-closed reviewed admission for a single-image component diagnostic."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REVIEW=BASE/'e1_independent_review'
sys.path.insert(0,str(BASE/'e4_c_latency_full'));sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal,cpu_container
from pilot_gate import pilot_gate

def sources():
 from latency_worker import sources as upstream
 files=[*upstream(),*HERE.glob('*.py'),BASE/'e4_c_latency_full/pilot_gate.py',BASE/'e4_c_latency_full/budget_plan.json',BASE/'e4_c_microbench/schedule001.json',BASE/'e4_c_microbench/budget_plan.json',BASE/'e4_c_microbench/capture.py']
 return sorted(set(p.resolve() for p in files),key=str)

def admit(*,preflight_only,host):
 suffix='preflight' if preflight_only else 'final';proof=HERE/('independent_'+suffix+('_review_002.json' if preflight_only else '_review.json'));reviewer=REVIEW/('review_component_pilot_'+suffix+'.py');monitor=REVIEW/('monitor_component_pilot_'+suffix+('_002' if preflight_only else ''))
 if not proof.exists():raise ValueError('component pilot locked: independent '+suffix+' review missing')
 supervised(monitor/'started.json',monitor/'terminal.json',reviewer,proof);r=json.loads(proof.read_text());expected={str(p):sha(p) for p in sources()};cpu=HERE/'cpu_check_010/verification.json';cs=HERE/'monitor_cpu010/started.json';ct=HERE/'monitor_cpu010/terminal.json';supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);c=json.loads(cpu.read_text())
 if (c.get('status')!='passed' or c.get('source_sha256_before')!=expected or c.get('source_sha256_after')!=expected or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=expected or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('GT_opened') is not False):raise ValueError('component source or review mismatch')
 upstream=pilot_gate(host=host)
 if not preflight_only:
  out=HERE/'worker_preflight002';s=HERE/'monitor_worker_preflight002/started.json';t=HERE/'monitor_worker_preflight002/terminal.json';started=terminal(s,t);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'));expected_launcher=HERE/'pilot_launch.py'
  if [(cwd/x).resolve() for x in cmd if Path(x).name==expected_launcher.name]!=[expected_launcher.resolve()] or cmd.count('--preflight-only')!=1 or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('preflight supervisor command mismatch')
  launch=json.loads((out/'launch.json').read_text());exit=json.loads((out/'exit.json').read_text());result=json.loads((out/'worker_preflight.json').read_text());cpu_container(launch,exit,started,HERE/'pilot_worker.py')
  for key,path in [('worker_launch',out/'launch.json'),('worker_exit',out/'exit.json'),('worker_started',s),('worker_terminal',t),('worker_result',out/'worker_preflight.json')]:
   if r.get(key+'_sha256')!=sha(path):raise ValueError('preflight proof mismatch '+key)
  prior=admit(preflight_only=True,host=host)
  if launch.get('approval_bundle')!=prior or result.get('source_sha256')!=expected or result.get('status')!='passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('predictions_opened') is not True or result.get('GT_opened') is not False or result.get('scoring_performed') is not False or result.get('jobs')!=60:raise ValueError('worker preflight invalid')
  if host:
   import subprocess
   obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];validate_container(launch,exit,obj,gpu=False)
 return {'review_sha256':sha(proof),'review_started_sha256':sha(monitor/'started.json'),'review_terminal_sha256':sha(monitor/'terminal.json'),'source_sha256':expected,'upstream':upstream,'preflight_only':preflight_only}

def validate_container(launch,exited,obj,*,gpu):
 from real_worker import PLAN
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command
 cmd=launch['command'];controller=launch['controller'];out=Path(launch['out']).resolve();name=launch['name'];args=[str(HERE/'pilot_worker.py'),'--out',str(out)]+([] if gpu else ['--preflight-only']);expected=container_command(json.loads(PLAN.read_text()),args,gpu,name);expected.remove('--rm')
 if cmd!=expected or exited.get('controller')!=controller:raise ValueError('container command/controller mismatch')
 cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=json.loads(PLAN.read_text())['container_image'];ident=exited.get('container_identity') or {}
 if obj.get('Name')!='/'+name or obj['Id']!=ident.get('container_id') or obj['Image']!=image or cfg.get('Image')!=image or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False:raise ValueError('container state/image mismatch')
 if cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]] or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1] or cfg.get('Cmd')!=cmd[cmd.index(image)+1:] or host.get('NetworkMode')!=cmd[cmd.index('--network')+1] or host.get('PidMode') not in ('',None):raise ValueError('container process mismatch')
 for env in [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']:
  if [v for v in cfg.get('Env',[]) if v.startswith(env.split('=',1)[0]+'=')]!=[env]:raise ValueError('container env mismatch')
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or {(d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in actual}!={(d,d,'rwm') for d in devices} or len(actual)!=len(devices) or (not gpu and devices):raise ValueError('container devices mismatch')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   parts=cmd[i+1].split(',');attrs=dict(p.split('=',1) for p in parts if '=' in p);mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
 actual=obj.get('Mounts') or [];parsed=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual if m.get('Type')=='bind']
 if len(actual)!=len(mounts) or sorted(parsed)!=sorted(mounts):raise ValueError('container mounts mismatch')
 return {'container_id':obj['Id'],'image':image,'devices':devices,'bind_mounts':len(mounts)}
