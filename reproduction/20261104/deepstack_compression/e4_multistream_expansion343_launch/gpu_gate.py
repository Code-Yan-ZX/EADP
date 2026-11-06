"""GPU host guard for the exact v2 worker exercised in CPU preflight."""
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;V=B/'e4_multistream_expansion343_gpu';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path,digest):
 path=Path(path).resolve()
 if sha(path)!=digest or name in sys.modules:raise ValueError('changed/preloaded '+name)
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def terminal(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or len(cmd)<2 or (cwd/cmd[1]).resolve()!=Path(script).resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=Path(out).resolve()))):raise ValueError('supervision')
def inspect_preflight(sources):
 run=V/'preflight001';la=read(run/'launch.json');ex=read(run/'exit.json');identity=read(run/'container_identity.json');p=B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';diag=p/'diagnose.py';authorpath=p/'run_panel.py'
 if 'diagnose' in sys.modules:raise ValueError('diagnose preloaded')
 load('diagnose',diag,sources[str(diag.resolve())]);author=load('_multistream_v3_preflight_author',authorpath,sources[str(authorpath.resolve())]);plan=read(p/'run_plan.json');cmd=author.container_command(plan,[str(V/'worker.py'),'--out',str(run),'--preflight-only'],False,la['name'])
 if cmd.count('--rm')!=1:raise ValueError('unexpected docker flags')
 cmd.remove('--rm')
 if cmd!=la['command'] or la.get('gpu_requested') is not False or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0 or ex.get('failure_type') is not None or identity!=ex.get('container_identity'):raise ValueError('preflight launch/exit')
 obj=json.loads(subprocess.run(['docker','inspect',la['name']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];image=plan['container_image']
 if obj['Id']!=identity['container_id'] or obj['Image']!=image or cfg['Image']!=image or st['Running'] is not False or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] is not False or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or cfg['Cmd']!=cmd[cmd.index(image)+1:] or host['NetworkMode']!=cmd[cmd.index('--network')+1] or (host.get('Devices') or [])!=[] or host.get('DeviceRequests') not in (None,[]):raise ValueError('preflight Docker process/image/device')
 for value in (cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'):
  if [x for x in cfg.get('Env',[]) if x.startswith(value.split('=',1)[0]+'=')]!=[value]:raise ValueError('preflight environment')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   a=cmd[i+1].split(',');d=dict(z.split('=',1) for z in a if '=' in z);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in a))
 actual=obj.get('Mounts') or [];parsed=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual if m.get('Type')=='bind']
 if len(actual)!=len(mounts) or sorted(parsed)!=sorted(mounts):raise ValueError('preflight mounts')
 return {'container_id':obj['Id'],'image':image,'mounts':len(mounts)}
def analysis_prelaunch():
 d=B/'e4_multistream_expansion343_analysis';code=read(d/'code_plan002.json');cpu=read(d/'cpu_proof002.json');plan=read(d/'plan_freeze002.json')
 terminal(d/'monitor_cpu002',d/'freeze_code002.py');terminal(d/'monitor_freeze_plan002',d/'freeze_plan002.py')
 if (code.get('status')!='frozen' or cpu.get('status')!='passed' or plan.get('status')!='frozen' or cpu.get('GT_opened') is not False or cpu.get('new_343_values_opened') is not False or cpu.get('source_before')!=cpu.get('source_after') or code.get('source_sha256')!=cpu.get('source_after') or code.get('cpu_sha256')!=sha(d/'cpu_proof002.json') or code.get('plan_sha256')!=sha(d/'plan_freeze002.json') or plan.get('plan_sha256')!=sha(d/'PLAN.md') or any(sha(k)!=v for k,v in code['source_sha256'].items()) or any(sha(k)!=v for k,v in plan['identity_and_prior_sha256'].items())):raise ValueError('frozen analysis sources')
 proof=I/'e4_multistream_expansion343_analysis_prelaunch_review.json';reviewer=I/'review_e4_multistream_expansion343_analysis_prelaunch.py';mon=I/'monitor_e4_multistream_expansion343_analysis_prelaunch';terminal(mon,reviewer,proof);q=read(proof)
 fields={'reviewer_sha256':reviewer,'plan_sha256':d/'plan_freeze002.json','code_sha256':d/'code_plan002.json','cpu_sha256':d/'cpu_proof002.json','cpu_started_sha256':d/'monitor_cpu002/started.json','cpu_terminal_sha256':d/'monitor_cpu002/terminal.json'}
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GT_opened') is not False or q.get('new_343_values_opened') is not False or q.get('source_sha256')!=code['source_sha256'] or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('analysis independent review')
 return sha(proof)
def admit():
 analysis_review=analysis_prelaunch();cpu=D/'cpu_check001.json';terminal(D/'monitor_cpu_check001',D/'cpu_check.py');c=read(cpu)
 if c.get('status')!='passed' or c['source_sha256_before']!=c['source_sha256_after'] or any(sha(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('v3 source freeze')
 proof=I/'e4_multistream_expansion343_guard_gpu_review.json';reviewer=I/'review_e4_multistream_expansion343_guard_gpu_review.py';mon=I/'monitor_e4_multistream_expansion343_guard_gpu_review';terminal(mon,reviewer,proof);q=read(proof)
 fields={'reviewer_sha256':reviewer,'cpu_sha256':cpu,'cpu_started_sha256':D/'monitor_cpu_check001/started.json','cpu_terminal_sha256':D/'monitor_cpu_check001/terminal.json','worker_gpu_review_sha256':I/'e4_multistream_expansion343_gpu_review.json','preflight_result_review_sha256':I/'e4_multistream_expansion343_preflight001_result_review.json','formal_plan_sha256':D/'formal_plan001.json','analysis_review_sha256':I/'e4_multistream_expansion343_analysis_prelaunch_review.json'}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('errors')!=[] or q.get('GT_opened') is not False or q.get('source_sha256')!=c['source_sha256_after'] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('v3 independent GPU review')
 pre=I/'e4_multistream_expansion343_preflight001_result_review.json';rev=I/'review_e4_multistream_expansion343_preflight001_result.py';mon=I/'monitor_e4_multistream_expansion343_preflight001_result';terminal(mon,rev,pre);r=read(pre);run=V/'preflight001'
 fields={'reviewer_sha256':rev,'cpu_sha256':V/'cpu_check002.json','worker_preflight_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':V/'monitor_preflight001/started.json','terminal_sha256':V/'monitor_preflight001/terminal.json'}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('images')!=343 or r.get('processor_inputs_checked')!=343 or r.get('e1_p3d1_complete_references')!=343 or r.get('model_loaded') is not False or r.get('cuda_initialized') is not False or r.get('GT_opened') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('v2 actual preflight independent proof')
 sources=read(V/'cpu_check002.json')['source_sha256_after'];m=load('_multistream_v3_v2admission',V/'admission.py',sources[str((V/'admission.py').resolve())]);approval=m.admit(False,host=True);inspection=inspect_preflight(sources)
 return {'analysis_review_sha256':analysis_review,'full_approval':approval,'guard_review_sha256':sha(proof),'guard_source_sha256':c['source_sha256_after'],'preflight_inspection':inspection}
def inspect_gpu(launch,exited):
 author=sys.modules.get('_multistream_v3_preflight_author')
 if author is None:raise ValueError('author command builder missing')
 root=B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';plan=read(root/'run_plan.json');cmd=author.container_command(plan,[str(V/'worker.py'),'--out',str(V/'run001')],True,launch['name'])
 if cmd.count('--rm')!=1:raise ValueError('GPU docker flags')
 cmd.remove('--rm')
 if cmd!=launch['command'] or launch.get('gpu_requested') is not True or exited.get('container_stopped_verified') is not True or exited.get('container_exit_code')!=0 or exited.get('failure_type') is not None or exited.get('gpu_pids_after')!=[]:raise ValueError('GPU launch/exit')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];image=plan['container_image']
 if obj['Id']!=exited['container_identity']['container_id'] or obj['Image']!=image or cfg['Image']!=image or st['Running'] is not False or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] is not False or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or cfg['Cmd']!=cmd[cmd.index(image)+1:] or host['NetworkMode']!=cmd[cmd.index('--network')+1]:raise ValueError('GPU Docker image/process')
 for value in (cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'):
  if [x for x in cfg.get('Env',[]) if x.startswith(value.split('=',1)[0]+'=')]!=[value]:raise ValueError('GPU environment')
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in actual}!={(d,d,'rwm') for d in devices} or len(actual)!=len(devices) or not devices:raise ValueError('GPU device')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   parts=cmd[i+1].split(',');d=dict(z.split('=',1) for z in parts if '=' in z);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in parts))
 actual=obj.get('Mounts') or [];parsed=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual if m.get('Type')=='bind']
 if len(actual)!=len(mounts) or sorted(parsed)!=sorted(mounts):raise ValueError('GPU mounts')
 return {'container_id':obj['Id'],'image':image,'mounts':len(mounts),'devices':devices}
