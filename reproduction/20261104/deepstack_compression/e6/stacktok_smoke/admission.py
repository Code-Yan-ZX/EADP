"""Independent review and complete no-GPU preflight gates; absent proof locks GPU."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];R=BASE/'e1_independent_review';P=BASE/'e6/stacktok_audit/prototype';A=BASE/'e6/external_sources/stacktok_571bdc6';sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal

def sources():
 old=json.loads((BASE/'e5/selector_real32/independent_final_review.json').read_text())['source_sha256']
 extra=[P/x for x in ('plan001.json','local_adapter.py','runtime_capture.py','model_adapter.py','test_local_adapter.py','test_runtime_capture.py','test_model_adapter.py')]
 extra += [A/'stacktok/core/selector.py',A/'tests/test_selector.py',BASE/'e4_native/native_capture.py',BASE/'e4/decode_observer.py',BASE/'e4/identity_records.py',BASE/'e4_native/native_001/frozen.json']
 return sorted(set([*[Path(x) for x in old],*HERE.glob('*.py'),HERE/'plan001.json',HERE/'budget_plan.json',*extra]),key=str)

def prior():
 d=BASE/'e4_native';f=d/'native_001_verification.json';fs=d/'monitor_verify_native001/started.json';ft=d/'monitor_verify_native001/terminal.json';r=R/'e4native001_review.json';rs=R/'monitor_e4native001/started.json';rt=R/'monitor_e4native001/terminal.json';reviewer=R/'review_e4native.py';a=json.loads(f.read_text());b=json.loads(r.read_text());terminal(fs,ft);st=terminal(rs,rt);cwd=Path(st['cwd']);cmd=st['command']
 if [(cwd/x).resolve() for x in cmd if Path(x).name==reviewer.name]!=[reviewer.resolve()] or a.get('status')!='passed' or a.get('pairs')!=6 or a.get('GT_opened') is not False or b.get('status')!='passed' or b.get('pairs_verified')!=6 or b.get('reviewer_sha256')!=sha(reviewer) or b.get('formal_proof_sha256')!=sha(f) or b.get('formal_verifier_terminal_sha256')!=sha(ft):raise ValueError('native identity proof mismatch')
 plan=json.loads((HERE/'plan001.json').read_text());frozen=d/'native_001/frozen.json'
 if sha(frozen)!=plan['sample_source_sha256'] or plan['anchors']!=json.loads(frozen.read_text())['anchors']:raise ValueError('native anchor mismatch')
 if plan['author_commit']!='571bdc6c4c6b3c92a94fe2a92e3aaf42b989cc96' or sha(A/'stacktok/core/selector.py')!='329e3a80cef25ec09f159954a4dd30e7ec7a7dbfe03f394e15727a6e8f3e42cc':raise ValueError('author selector source mismatch')
 return {'native_formal_sha256':sha(f),'native_review_sha256':sha(r),'native_review_terminal_sha256':sha(rt),'frozen_anchor_sha256':sha(frozen)}

def admit(*,preflight_only,host):
 suffix='preflight' if preflight_only else 'final';proof=HERE/('independent_'+suffix+'_review.json');reviewer=R/('review_stacktok_smoke_'+suffix+'.py');mon=R/('monitor_stacktok_smoke_'+suffix)
 if not proof.is_file():raise ValueError('StackTok smoke locked pending independent '+suffix+' review')
 supervised(mon/'started.json',mon/'terminal.json',reviewer,proof);r=json.loads(proof.read_text());expected={str(x.resolve()):sha(x) for x in sources()};cpu=HERE/'cpu_check008/verification.json';cs=HERE/'monitor_cpu008/started.json';ct=HERE/'monitor_cpu008/terminal.json';supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);c=json.loads(cpu.read_text());identity=prior()
 if c.get('status')!='passed' or c.get('source_sha256_before')!=expected or c.get('source_sha256_after')!=expected or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=expected or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('GT_opened') is not False or r.get('prior_identity')!=identity:raise ValueError('StackTok review mismatch')
 if not preflight_only:
  out=HERE/'worker_preflight002';s=HERE/'monitor_worker_preflight002/started.json';t=HERE/'monitor_worker_preflight002/terminal.json';started=terminal(s,t);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or cmd.count('--preflight-only')!=1 or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('preflight supervisor mismatch')
  launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());result=json.loads((out/'worker_preflight.json').read_text())
  for k,p in [('worker_launch',out/'launch.json'),('worker_exit',out/'exit.json'),('worker_started',s),('worker_terminal',t),('worker_result',out/'worker_preflight.json')]:
   if r.get(k+'_sha256')!=sha(p):raise ValueError('preflight evidence '+k)
  if launch.get('approval_bundle')!=admit(preflight_only=True,host=host) or launch.get('controller',{}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or result.get('source_sha256')!=expected or result.get('status')!='passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('images')!=3 or result.get('GT_opened') is not False:raise ValueError('preflight incomplete')
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
