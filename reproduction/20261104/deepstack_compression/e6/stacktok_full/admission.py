"""Independent review and complete no-GPU preflight gates; absent proof locks GPU."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];R=BASE/'e1_independent_review';P=BASE/'e6/stacktok_audit/prototype';A=BASE/'e6/external_sources/stacktok_571bdc6';sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal

def sources():
 old=json.loads((BASE/'e6/stacktok_smoke/independent_final_review.json').read_text())['source_sha256']
 return sorted(set([*[Path(x) for x in old],*HERE.glob('*.py'),HERE/'PLAN.md',HERE/'plan.json',HERE/'budget_plan.json']),key=str)

def prior():
 smoke=BASE/'e6/stacktok_smoke';formal=smoke/'run_001_verification.json';fs=smoke/'monitor_verify001/started.json';ft=smoke/'monitor_verify001/terminal.json';review=R/'stacktok_smoke001_review.json';rs=R/'monitor_stacktok_smoke001/started.json';rt=R/'monitor_stacktok_smoke001/terminal.json';reviewer=R/'review_stacktok_smoke_result.py'
 supervised(fs,ft,smoke/'verify.py',formal);supervised(rs,rt,reviewer,review);p=json.loads(formal.read_text());q=json.loads(review.read_text());plan=json.loads((HERE/'plan.json').read_text());inputfile=BASE/'e1/input_preparation_002/inputs.jsonl';budget=BASE/'initial_plan.json'
 if (p.get('status')!='passed' or p.get('images')!=3 or p.get('generations')!=18 or p.get('keepall_pairs')!=6 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('verifier_sha256')!=sha(smoke/'verify.py') or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=3 or q.get('generations')!=18 or q.get('keepall_pairs')!=6 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('reviewer_sha256')!=sha(reviewer) or q.get('formal_sha256')!=sha(formal) or q.get('formal_started_sha256')!=sha(fs) or q.get('formal_terminal_sha256')!=sha(ft) or q.get('run_started_sha256')!=sha(smoke/'monitor_run001/started.json') or q.get('run_terminal_sha256')!=sha(smoke/'monitor_run001/terminal.json')):raise ValueError('StackTok diagnostic identity changed')
 if plan.get('smoke_formal_sha256')!=sha(formal) or plan.get('smoke_independent_sha256')!=sha(review) or plan.get('input_sha256')!=sha(inputfile) or plan.get('user_budget_sha256')!=sha(budget) or plan.get('user_budget_end_unix')!=json.loads(budget.read_text())['budget_end_unix'] or plan.get('questions')!=384 or plan.get('budget')!=256:raise ValueError('full plan changed')
 return {'smoke_formal_sha256':sha(formal),'smoke_review_sha256':sha(review),'smoke_review_terminal_sha256':sha(rt),'input_sha256':sha(inputfile),'plan_sha256':sha(HERE/'plan.json')}

def admit(*,preflight_only,host):
 suffix='preflight' if preflight_only else 'final';proof=HERE/('independent_'+suffix+'_review.json');reviewer=R/('review_stacktok_full_'+suffix+'.py');mon=R/('monitor_stacktok_full_'+suffix)
 if not proof.is_file():raise ValueError('StackTok full locked pending independent '+suffix+' review')
 supervised(mon/'started.json',mon/'terminal.json',reviewer,proof);r=json.loads(proof.read_text());expected={str(x.resolve()):sha(x) for x in sources()};cpu=HERE/'cpu_check003/verification.json';cs=HERE/'monitor_cpu003/started.json';ct=HERE/'monitor_cpu003/terminal.json';supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);c=json.loads(cpu.read_text());identity=prior()
 if c.get('status')!='passed' or c.get('source_sha256_before')!=expected or c.get('source_sha256_after')!=expected or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=expected or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('GT_opened') is not False or r.get('prior_identity')!=identity:raise ValueError('StackTok review mismatch')
 if not preflight_only:
  out=HERE/'worker_preflight002';s=HERE/'monitor_worker_preflight002/started.json';t=HERE/'monitor_worker_preflight002/terminal.json';started=terminal(s,t);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or cmd.count('--preflight-only')!=1 or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('preflight supervisor mismatch')
  launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());result=json.loads((out/'worker_preflight.json').read_text())
  for k,p in [('worker_launch',out/'launch.json'),('worker_exit',out/'exit.json'),('worker_started',s),('worker_terminal',t),('worker_result',out/'worker_preflight.json')]:
   if r.get(k+'_sha256')!=sha(p):raise ValueError('preflight evidence '+k)
  if launch.get('approval_bundle')!=admit(preflight_only=True,host=host) or launch.get('controller',{}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or result.get('source_sha256')!=expected or result.get('status')!='passed' or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('questions')!=384 or result.get('GT_opened') is not False or result.get('weights_verified')!=4 or result.get('predictions_opened') is not True:raise ValueError('preflight incomplete')
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
