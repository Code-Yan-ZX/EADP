"""Read-only check of retained, stopped model timing Docker container."""
from pathlib import Path

def validate(job,launch,exited,obj,author,container_command,worker_path):
 ctrl=launch.get('controller') or {};pid=ctrl.get('pid');identity=exited.get('container_identity') or {}
 if type(pid) is not int or pid<=0:raise ValueError('controller identity')
 name='eadp-pixel-t020-ttft-%03d-%d'%(job['order'],pid)
 cmd=container_command(author,[str(Path(worker_path).resolve()),'--job-index',str(job['order'])],True,name)
 if cmd.count('--rm')!=1:raise ValueError('unsafe expected command')
 cmd.remove('--rm')
 if launch.get('job')!=job or launch.get('container_name')!=name or launch.get('command')!=cmd or exited.get('controller')!=ctrl:raise ValueError('launcher mismatch')
 cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=author['container_image']
 if (obj.get('Name')!='/'+name or obj.get('Id')!=identity.get('container_id') or obj.get('Image')!=image or identity.get('image')!=image or cfg.get('Image')!=image or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or state.get('OOMKilled') is not False):raise ValueError('container state or image')
 if (cfg.get('Entrypoint')!=[cmd[cmd.index('--entrypoint')+1]] or cfg.get('User')!=cmd[cmd.index('--user')+1] or cfg.get('WorkingDir')!=cmd[cmd.index('--workdir')+1] or cfg.get('Cmd')!=cmd[cmd.index(image)+1:] or host.get('NetworkMode')!=cmd[cmd.index('--network')+1] or host.get('PidMode') not in ('',None)):raise ValueError('container process')
 for env in [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']:
  key=env.split('=',1)[0]+'='
  if [v for v in cfg.get('Env',[]) if v.startswith(key)]!=[env]:raise ValueError('container environment')
 devices=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--device'];actual=host.get('Devices') or []
 if host.get('DeviceRequests') not in (None,[]) or len(actual)!=len(devices) or {(d['PathOnHost'],d['PathInContainer'],d['CgroupPermissions']) for d in actual}!={(d,d,'rwm') for d in devices}:raise ValueError('container device mapping')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x!='--mount':continue
  parts=cmd[i+1].split(',');attrs=dict(part.split('=',1) for part in parts if '=' in part)
  if attrs.get('type')!='bind' or 'src' not in attrs or 'dst' not in attrs:raise ValueError('mount spec')
  mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
 actual=obj.get('Mounts') or [];parsed=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual if m.get('Type')=='bind']
 if len(actual)!=len(mounts) or sorted(parsed)!=sorted(mounts):raise ValueError('bind mounts')
 return {'container_id':obj['Id'],'name':name,'image':image,'device_count':len(devices),'bind_count':len(mounts)}
