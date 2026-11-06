"""Rebuild full timing Docker command and compare actual stopped container."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def validate(run,launch,exited,obj,plan,container_command,job,pilot):
 run=Path(run).resolve();pilot=Path(pilot).resolve();controller=launch.get('controller') or {};ident=exited.get('container_identity') or {};pid=controller.get('pid');index=job['order']
 if run.parent!=pilot/'run_pilot001' or run.name!=('job_%03d'%index) or type(pid) is not int or pid<=0 or exited.get('controller')!=controller or launch.get('job')!=job:raise ValueError('controller or job identity mismatch')
 name='eadp-pixel-ttft-%03d-%d'%(index,pid);cmd=container_command(plan,[str(pilot/'gpu_worker_candidate.py'),'--job-index',str(index)],True,name)
 if cmd.count('--rm')!=1:raise ValueError('unexpected removal flag')
 cmd.remove('--rm')
 if launch.get('command')!=cmd or launch.get('container_name')!=name:raise ValueError('Docker command mismatch')
 cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=plan['container_image']
 if (obj.get('Name')!='/'+name or obj['Id']!=ident.get('container_id') or obj['Image']!=image or ident.get('image')!=image or cfg.get('Image')!=image or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False):raise ValueError('container image or state mismatch')
 if (cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]] or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1] or cfg.get('Cmd')!=cmd[cmd.index(image)+1:] or host.get('NetworkMode')!=cmd[cmd.index('--network')+1] or host.get('PidMode') not in ('',None)):raise ValueError('container process or isolation mismatch')
 for env in [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']:
  key=env.split('=',1)[0]+'='
  if [v for v in cfg.get('Env',[]) if v.startswith(key)]!=[env]:raise ValueError('container environment mismatch')
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or len(actual)!=len(devices) or {(d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in actual}!={(d,d,'rwm') for d in devices}:raise ValueError('container devices mismatch')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x!='--mount':continue
  parts=cmd[i+1].split(',');attrs=dict(part.split('=',1) for part in parts if '=' in part)
  if attrs.get('type')!='bind' or 'src' not in attrs or 'dst' not in attrs:raise ValueError('unexpected mount specification')
  mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
 actual_mounts=obj.get('Mounts') or [];parsed=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual_mounts if m.get('Type')=='bind']
 if len(actual_mounts)!=len(mounts) or sorted(parsed)!=sorted(mounts):raise ValueError('container bind mounts mismatch')
 return {'container_id':obj['Id'],'image':image,'name':name,'devices':devices,'bind_mounts':len(mounts)}
