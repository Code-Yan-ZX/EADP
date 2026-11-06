"""Independently reconstruct and validate an Div/CD smoke Docker identity.

No Docker commands, model, predictions, or reference answers are accessed.
The caller must obtain ``inspected`` with a fresh host-side Docker inspect.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def expected_command(run,controller_pid):
 run=Path(run).resolve()
 if not run.name.startswith('panel_') or not run.name.removeprefix('panel_').isalnum() or type(controller_pid)!=int or controller_pid<=0:
  raise ValueError('invalid runtime/controller identity')
 plan=json.loads(PLAN.read_text());script=PLAN.parent/'run_panel.py';runtime=Path(plan['container_runtime_source'])
 available={**plan['source_sha256'],**plan['implementation_sha256']}
 for path in (script,runtime):
  if available.get(str(path))!=sha(path):raise ValueError('container builder source changed')
 from full_imports import load
 diagnose=PLAN.parent/'diagnose.py'
 if available.get(str(diagnose))!=sha(diagnose):raise ValueError('container builder dependency changed')
 load('diagnose',diagnose)
 sys.path.insert(0,str(PLAN.parent))
 try:
  spec=importlib.util.spec_from_file_location('_m3_oracle_reviewed_container_builder',script)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 finally:sys.path.pop(0)
 name='eadp-selected-full-'+run.name.removeprefix('panel_')+'-'+str(controller_pid)
 args=[str(HERE/'worker.py'),'--execute','--out',str(run)]
 command=module.container_command(plan,args,True,name)
 from full_imports import load,SMOKE
 add_audit_mount=load('eadp_selected_full_container_mount',SMOKE/'container_mount.py').add
 command=add_audit_mount(command)
 if command.count('--rm')!=1:raise ValueError('unexpected container removal flag')
 command.remove('--rm')
 return plan,command,name

def validate(run,launch,exited,inspected):
 """Raise on any mismatch; called only after genuine completed host inspect."""
 controller=launch.get('controller') or {};ident=exited.get('container_identity') or {}
 if exited.get('controller')!=controller:raise ValueError('controller identity mismatch')
 plan,expected,name=expected_command(run,controller.get('pid'));cmd=launch.get('command')
 if cmd!=expected or exited.get('name')!=name:raise ValueError('recorded Docker command differs from fixed plan')
 obj=inspected;cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=plan['container_image']
 if (obj['Id']!=ident.get('container_id') or obj['Image']!=image or ident.get('image')!=image or cfg['Image']!=image or
     state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False):
  raise ValueError('container identity or stopped state differs')
 if (cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]] or cfg.get('User')!=cmd[cmd.index('--user')+1] or
     cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1] or host.get('NetworkMode')!=cmd[cmd.index('--network')+1] or
     cfg.get('Cmd')!=cmd[cmd.index(image)+1:]):raise ValueError('actual process/container configuration differs')
 expected_env=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'];env=cfg.get('Env') or []
 for item in expected_env:
  key=item.split('=',1)[0]+'='
  if [value for value in env if value.startswith(key)]!=[item]:raise ValueError('container environment differs')
 expected_devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];devices=host.get('Devices') or []
 if (host.get('DeviceRequests') not in (None,[]) or len(devices)!=len(expected_devices) or
     {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}!={(d,d,'rwm') for d in expected_devices}):
  raise ValueError('container devices differ')
 # Validate bind mounts as well: expected read-only inputs must remain read-only.
 mounts=obj.get('Mounts') or [];expected_mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x!='--mount':continue
  parts=cmd[i+1].split(',');attrs=dict(part.split('=',1) for part in parts if '=' in part)
  if attrs.get('type')!='bind' or 'src' not in attrs or 'dst' not in attrs:raise ValueError('unsupported expected mount')
  expected_mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
 actual_mounts=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in mounts if m.get('Type')=='bind']
 if len(mounts)!=len(expected_mounts) or sorted(actual_mounts)!=sorted(expected_mounts):raise ValueError('container bind mounts differ')
 return {'container_id':obj['Id'],'image':image,'name':name,'devices':expected_devices,'bind_mounts':len(expected_mounts)}
