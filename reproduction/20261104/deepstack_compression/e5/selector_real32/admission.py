"""Fail-closed review gates for 32-image real selector capture and comparison."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;FULL=BASE/'e4_c_microbench_full';REVIEW=BASE/'e1_independent_review';sys.path[:0]=[str(FULL),str(BASE/'e4_c_microbench_pilot'),str(BASE/'e4_c_latency')]
from full_admission import admit as full_admit,sources as full_sources
from identity_gate import sha,supervised
from latency_admission import terminal,cpu_container
from panel_reference import accepted as e4_accepted

def sources():
 return sorted(set([*full_sources(),*HERE.glob('*.py'),HERE/'budget_plan.json',HERE/'sample_manifest_001.json',BASE/'e5/fused_selector/kernel.py',BASE/'e5/graph_selector/graph.py',BASE/'e4_panel_v2/panel_verify.py',REVIEW/'review_e4panel.py']),key=str)

def prior(host):
 from freeze_manifest import build
 fm=HERE/'sample_manifest_001.json';ms=HERE/'monitor_freeze001/started.json';mt=HERE/'monitor_freeze001/terminal.json';started=terminal(ms,mt);cwd=Path(started.get('cwd','/'));cmd=started.get('command',[])
 if [(cwd/x).resolve() for x in cmd if Path(x).name=='freeze_manifest.py']!=[(HERE/'freeze_manifest.py').resolve()]:raise ValueError('sample freeze supervisor command mismatch')
 sample=json.loads(fm.read_text());inputs=BASE/'e1/input_preparation_002/inputs.jsonl';schedule=BASE/'e4_c_microbench/schedule001.json';rows=[json.loads(x) for x in inputs.read_text().splitlines() if x.strip()];pilot=json.loads(schedule.read_text())['input_identities']
 if sample.get('status')!='frozen' or sample.get('GT_opened') is not False or sample.get('predictions_opened') is not False or sample.get('gpu_used') is not False or sample.get('inputs_sha256')!=sha(inputs) or sample.get('schedule_sha256')!=sha(schedule) or sample.get('freeze_script_sha256')!=sha(HERE/'freeze_manifest.py') or sample.get('full_128')!=build(rows,pilot):raise ValueError('frozen sample proof mismatch')
 upstream=full_admit(preflight_only=False,host=host);f=FULL/'full_001_verification.json';fs=FULL/'monitor_verify_full001/started.json';ft=FULL/'monitor_verify_full001/terminal.json';r=REVIEW/'component_full001_review.json';rs=REVIEW/'monitor_component_full001/started.json';rt=REVIEW/'monitor_component_full001/terminal.json';reviewer=REVIEW/'review_component_full001.py';runs=FULL/'monitor_full001/started.json';runt=FULL/'monitor_full001/terminal.json';plan=json.loads((HERE/'budget_plan.json').read_text());manifest=HERE/'sample_manifest_001.json'
 supervised(fs,ft,FULL/'full_verify.py',f);supervised(rs,rt,reviewer,r);terminal(runs,runt);a=json.loads(f.read_text());b=json.loads(r.read_text())
 if a.get('status')!='passed' or b.get('status')!='passed' or b.get('independent_review') is not True or any(a.get(k)!=v or b.get(k)!=v for k,v in [('images',32),('jobs',1920),('warmups',288),('diagnostic_generations',32),('GT_opened',False),('scoring_performed',False)]) or b.get('reviewer_sha256')!=sha(reviewer) or b.get('formal_sha256')!=sha(f) or b.get('formal_started_sha256')!=sha(fs) or b.get('formal_terminal_sha256')!=sha(ft) or b.get('run_started_sha256')!=sha(runs) or b.get('run_terminal_sha256')!=sha(runt) or a.get('array_sha256')!=b.get('array_sha256'):raise ValueError('full 32-image component proof mismatch')
 if len(a.get('array_sha256',{}))!=32 or any(sha(p)!=h for p,h in a.get('array_sha256',{}).items()) or any(sha(p)!=h for p,h in a.get('files_sha256',{}).items()) or plan['manifest_sha256']!=sha(manifest) or plan['kernel_sha256']!=sha(BASE/'e5/fused_selector/kernel.py') or plan['graph_sha256']!=sha(BASE/'e5/graph_selector/graph.py') or plan['full_component_formal_sha256']!=sha(f) or plan['full_component_review_sha256']!=sha(r):raise ValueError('full component evidence or plan mismatch')
 graph=REVIEW/'graph_selector_smoke001_review.json';gs=REVIEW/'monitor_graph_selector_smoke001/started.json';gt=REVIEW/'monitor_graph_selector_smoke001/terminal.json';gr=REVIEW/'review_graph_selector_smoke001.py';supervised(gs,gt,gr,graph);g=json.loads(graph.read_text())
 if g.get('status')!='passed' or g.get('independent_review') is not True or g.get('selection_count')!=256 or g.get('measurements')!=15 or g.get('GT_opened') is not False or g.get('reviewer_sha256')!=sha(gr) or plan['graph_single_review_sha256']!=sha(graph) or any(sha(p)!=h for p,h in g.get('files_sha256',{}).items()):raise ValueError('graph single-input identity proof mismatch')
 e4=e4_accepted()
 if plan['e4_panel_formal_sha256']!=e4['formal_sha256'] or plan['e4_panel_review_sha256']!=e4['review_sha256']:raise ValueError('E4 prior mismatch')
 return {'component_upstream':upstream,'component_formal_sha256':sha(f),'component_review_sha256':sha(r),'graph_review_sha256':sha(graph),'e4':e4}

def admit(*,preflight_only,host):
 suffix='preflight' if preflight_only else 'final';proof=HERE/('independent_'+suffix+('_review_005.json' if preflight_only else '_review.json'));reviewer=REVIEW/('review_selector_real32_'+suffix+'.py');monitor=REVIEW/('monitor_selector_real32_'+suffix+('_005' if preflight_only else ''))
 if not proof.exists():raise ValueError('real selector locked: independent '+suffix+' review missing')
 supervised(monitor/'started.json',monitor/'terminal.json',reviewer,proof);r=json.loads(proof.read_text());expected={str(p):sha(p) for p in sources()};cpu=HERE/'cpu_check_005/verification.json';cs=HERE/'monitor_cpu005/started.json';ct=HERE/'monitor_cpu005/terminal.json';supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);c=json.loads(cpu.read_text());identity=prior(host)
 if c.get('status')!='passed' or c.get('source_sha256_before')!=expected or c.get('source_sha256_after')!=expected or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not preflight_only or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=expected or r.get('cpu_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('GT_opened') is not False or r.get('prior_identity')!=identity:raise ValueError('real selector source/review mismatch')
 if not preflight_only:
  out=HERE/'worker_preflight002';s=HERE/'monitor_worker_preflight002/started.json';t=HERE/'monitor_worker_preflight002/terminal.json';started=terminal(s,t);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if [(cwd/x).resolve() for x in cmd if Path(x).name=='launch.py']!=[(HERE/'launch.py').resolve()] or cmd.count('--preflight-only')!=1 or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('real selector preflight supervisor mismatch')
  launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());result=json.loads((out/'worker_preflight.json').read_text());cpu_container(launch,ex,started,HERE/'worker.py')
  for k,p in [('worker_launch',out/'launch.json'),('worker_exit',out/'exit.json'),('worker_started',s),('worker_terminal',t),('worker_result',out/'worker_preflight.json')]:
   if r.get(k+'_sha256')!=sha(p):raise ValueError('worker preflight evidence mismatch '+k)
  if launch.get('approval_bundle')!=admit(preflight_only=True,host=host) or result.get('source_sha256')!=expected or result.get('status')!='passed' or result.get('images')!=32 or result.get('cases')!=96 or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('predictions_opened') is not True or result.get('GT_opened') is not False or result.get('triton_cache',{}).get('path')!=str(HERE/'triton_cache') or result.get('triton_cache',{}).get('writable') is not True or result.get('triton_cache',{}).get('uid')!=1000:raise ValueError('real selector preflight result mismatch')
  if host:
   import subprocess
   obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];validate_container(launch,ex,obj,gpu=False)
 return {'review_sha256':sha(proof),'review_started_sha256':sha(monitor/'started.json'),'review_terminal_sha256':sha(monitor/'terminal.json'),'source_sha256':expected,'prior_identity':identity,'preflight_only':preflight_only}

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
