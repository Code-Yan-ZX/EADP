"""Two-stage fail-closed gate for the 96-image expansion."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;REVIEW=BASE/'e1_independent_review';sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal
from expansion_contract import load_manifest
from panel_reference import accepted as e4_accepted

def sources():
 prev=json.loads((BASE/'e5/selector_real32/independent_final_review.json').read_text())['source_sha256']
 return sorted(set([*[Path(x) for x in prev],*HERE.glob('*.py'),HERE/'budget_plan.json',BASE/'e5/selector_real32/sample_manifest_001.json']),key=str)

def prior(host):
 old=BASE/'e5/selector_real32';f=old/'run_001_verification.json';fs=old/'monitor_verify_run001/started.json';ft=old/'monitor_verify_run001/terminal.json';r=REVIEW/'selector_real32_run001_review.json';rs=REVIEW/'monitor_selector_real32_run001/started.json';rt=REVIEW/'monitor_selector_real32_run001/terminal.json';reviewer=REVIEW/'review_selector_real32_run001.py'
 supervised(fs,ft,old/'verify.py',f);supervised(rs,rt,reviewer,r);terminal(old/'monitor_run001/started.json',old/'monitor_run001/terminal.json');a=json.loads(f.read_text());b=json.loads(r.read_text())
 if a.get('status')!='passed' or b.get('status')!='passed' or b.get('independent_review') is not True or a.get('images')!=32 or b.get('images')!=32 or a.get('cases')!=96 or b.get('cases')!=96 or b.get('formal_sha256')!=sha(f) or b.get('formal_started_sha256')!=sha(fs) or b.get('formal_terminal_sha256')!=sha(ft) or b.get('reviewer_sha256')!=sha(reviewer) or b.get('array_sha256')!=a.get('array_sha256') or a.get('GT_opened') is not False or b.get('GT_opened') is not False:raise ValueError('original 32-image review mismatch')
 if len(a.get('array_sha256',{}))!=128 or any(sha(p)!=h for p,h in a['array_sha256'].items()) or any(sha(p)!=h for p,h in a.get('files_sha256',{}).items()):raise ValueError('32-image artifacts changed')
 # Keep old evidence and source snapshots frozen; do not rerun old model.
 old_final=json.loads((old/'independent_final_review.json').read_text())
 if any(sha(p)!=h for p,h in old_final['source_sha256'].items()):raise ValueError('previous source changed')
 manifest,remaining=load_manifest(old/'sample_manifest_001.json')
 if len(remaining)!=96 or sha(old/'sample_manifest_001.json')!=json.loads((HERE/'budget_plan.json').read_text())['manifest_sha256']:raise ValueError('96-image manifest mismatch')
 e4=e4_accepted()
 return {'pilot_formal_sha256':sha(f),'pilot_review_sha256':sha(r),'pilot_review_terminal_sha256':sha(rt),'e4':e4,'manifest_sha256':sha(old/'sample_manifest_001.json')}

def admit(*,preflight_only,host):
 suffix='preflight' if preflight_only else 'final';proof=HERE/('independent_'+suffix+'_review.json');reviewer=REVIEW/('review_selector_real128_'+suffix+'.py');mon=REVIEW/('monitor_selector_real128_'+suffix)
 if not proof.is_file():raise ValueError('96-image GPU/preflight locked pending independent '+suffix+' review')
 supervised(mon/'started.json',mon/'terminal.json',reviewer,proof);r=json.loads(proof.read_text());expected={str(x.resolve()):sha(x) for x in sources()};cpu=HERE/'cpu_check001/verification.json';cs=HERE/'monitor_cpu001/started.json';ct=HERE/'monitor_cpu001/terminal.json';supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);c=json.loads(cpu.read_text());identity=prior(host)
 if c.get('status')!='passed' or c.get('source_sha256_before')!=expected or c.get('source_sha256_after')!=expected or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=expected or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('GT_opened') is not False or r.get('prior_identity')!=identity:raise ValueError('expansion review mismatch')
 if not preflight_only:
  out=HERE/'worker_preflight002';s=HERE/'monitor_worker_preflight002/started.json';t=HERE/'monitor_worker_preflight002/terminal.json';started=terminal(s,t);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or cmd.count('--preflight-only')!=1 or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('expansion preflight supervisor mismatch')
  launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());result=json.loads((out/'worker_preflight.json').read_text())
  for k,p in [('worker_launch',out/'launch.json'),('worker_exit',out/'exit.json'),('worker_started',s),('worker_terminal',t),('worker_result',out/'worker_preflight.json')]:
   if r.get(k+'_sha256')!=sha(p):raise ValueError('preflight evidence '+k)
  if launch.get('approval_bundle')!=admit(preflight_only=True,host=host) or launch.get('controller',{}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or result.get('source_sha256')!=expected or result.get('status')!='passed' or result.get('images')!=96 or result.get('cases')!=288 or result.get('e4_identity_records')!=288 or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('predictions_opened') is not True or result.get('GT_opened') is not False or result.get('triton_cache',{}).get('path')!=str(HERE/'triton_cache') or result.get('triton_cache',{}).get('uid')!=1000 or result.get('triton_cache',{}).get('writable') is not True:raise ValueError('preflight incomplete')
  if host:
   import subprocess
   obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];validate_container(launch,ex,obj,gpu=False)
 return {'review_sha256':sha(proof),'review_started_sha256':sha(mon/'started.json'),'review_terminal_sha256':sha(mon/'terminal.json'),'source_sha256':expected,'prior_identity':identity,'preflight_only':preflight_only}

def validate_container(launch,exited,obj,*,gpu):
 from real_worker import PLAN
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command
 cmd=launch['command'];controller=launch['controller'];out=Path(launch['out']).resolve();name=launch['name'];args=[str(HERE/'worker.py'),'--out',str(out)]+([] if gpu else ['--preflight-only']);expected=container_command(json.loads(PLAN.read_text()),args,gpu,name);expected.remove('--rm')
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
