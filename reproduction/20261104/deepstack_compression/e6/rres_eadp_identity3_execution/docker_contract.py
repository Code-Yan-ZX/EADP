"""Compare actual GPU Docker image, full environment/mounts and controller."""
from pathlib import Path
DEVICES={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}
def env(values):
 pairs=[x.split('=',1) for x in values]
 if any(len(x)!=2 for x in pairs) or len({x[0] for x in pairs})!=len(pairs):raise ValueError('invalid duplicate Docker environment')
 return dict(pairs)
def mounts(detail):
 rows=[(x['Source'],x['Destination'],x['RW']) for x in detail['Mounts']]
 if len(set(rows))!=len(rows):raise ValueError('duplicate mount')
 return set(rows)
def check(*,actual,baseline,old_command,launch,identity,exit,arm,worker,out,supervisor_pid):
 cid=identity['container_id'];state=actual['State'];config=actual['Config'];host=actual['HostConfig'];base=baseline['Config'];basehost=baseline['HostConfig']
 if actual['Id']!=cid or state['Running'] or state['Pid']!=0 or state['ExitCode']!=0 or state.get('OOMKilled') or actual['Image']!=baseline['Image'] or config['Image']!=base['Image']:raise ValueError('Docker image or exit')
 if any(config[k]!=base[k] for k in ('User','Entrypoint','WorkingDir')) or any(host[k]!=basehost[k] for k in ('NetworkMode','CapDrop','SecurityOpt')) or host['NetworkMode']!='none':raise ValueError('Docker security or execution identity')
 if config['Cmd']!=[str(worker),'--arm',arm,'--out',str(out)]:raise ValueError('Docker actual command')
 actual_env=env(config['Env']);base_env=env(base['Env'])
 if base_env.get('CUDA_VISIBLE_DEVICES')!='' or actual_env.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('CUDA environment')
 base_env['CUDA_VISIBLE_DEVICES']='0'
 if actual_env!=base_env or mounts(actual)!=mounts(baseline):raise ValueError('Docker environment or mounts')
 devices=host.get('Devices') or []
 if len(devices)!=4 or {d['PathOnHost'] for d in devices}!=DEVICES or any(d['PathOnHost']!=d['PathInContainer'] or d['CgroupPermissions']!='rwm' for d in devices) or host.get('DeviceRequests') not in (None,[]):raise ValueError('Docker device exposure')
 cmd=list(old_command);name_index=cmd.index('--name');cmd[name_index+1]=launch['name'];image_index=cmd.index(baseline['Image']);cmd=cmd[:image_index+1]+[str(worker),'--arm',arm,'--out',str(out)]
 if launch['command']!=cmd or launch['arm']!=arm:raise ValueError('GPU launch command differs from accepted author template')
 c=launch['controller'];hostinit=identity['host_init']
 if c!=exit['controller'] or c.get('pid')!=supervisor_pid or hostinit.get('pid',0)<=0 or hostinit.get('start_ticks',0)<=0 or cid not in identity.get('host_cgroup','') or exit.get('container_identity')!=identity:raise ValueError('controller or container PID identity')
 return True
