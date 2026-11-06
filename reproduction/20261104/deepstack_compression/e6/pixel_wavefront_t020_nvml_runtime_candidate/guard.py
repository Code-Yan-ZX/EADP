"""CPU-only container/device validation helpers, no GPU launch authority."""
import re
from pathlib import Path

DEVICES=('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools')

def parse_inventory(inventory, compute):
    """A unique index-zero A40 and completely empty compute list are required."""
    lines=[x.strip() for x in inventory.splitlines() if x.strip()]
    if len(lines)!=1 or any(x.strip() for x in compute.splitlines()):
        raise ValueError('GPU inventory ambiguous or compute processes present')
    parts=[x.strip() for x in lines[0].split(',')]
    if len(parts)!=3 or parts[0]!='0' or parts[1]!='NVIDIA A40' or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',parts[2]):
        raise ValueError('expected exactly one index-zero A40 with UUID')
    return parts[2]

def create_command(original, name, nonce):
    """Turn reviewed author command into retained, nonce-owned docker create."""
    if (not re.fullmatch(r'eadp-memory-[a-z0-9-]+',name) or not re.fullmatch(r'[0-9a-f]{32}',nonce)
            or original[:2]!=['docker','run'] or original.count('--rm')!=1
            or original.count('--name')!=1 or original[original.index('--name')+1]!=name
            or '--label' in original or '--gpus' in original):
        raise ValueError('unrecognized author docker command')
    devices=[original[i+1] for i,x in enumerate(original[:-1]) if x=='--device']
    if tuple(devices)!=DEVICES: raise ValueError('unexpected CUDA device list')
    cmd=list(original);cmd[1]='create';cmd.remove('--rm');idx=cmd.index('--name')+2
    cmd[idx:idx]=['--label','eadp.memory.nonce='+nonce]
    return cmd

def validate_running(obj, cmd, image, name, worker_argv):
    """Compare actual container profile with reviewed docker-create argv."""
    if not isinstance(obj,dict) or not re.fullmatch(r'[0-9a-f]{64}',str(obj.get('Id',''))): raise ValueError('bad Docker inspect')
    state=obj.get('State') or {};cfg=obj.get('Config') or {};host=obj.get('HostConfig') or {}
    def arg(flag):
        if cmd.count(flag)!=1: raise ValueError('missing or repeated '+flag)
        return cmd[cmd.index(flag)+1]
    if (obj.get('Name')!='/'+name or obj.get('Image')!=image or cfg.get('Image')!=image
        or state.get('Running') is not True or type(state.get('Pid')) is not int or state['Pid']<=0
        or cfg.get('Entrypoint')!=[arg('--entrypoint')] or cfg.get('Cmd')!=worker_argv
        or cfg.get('User')!=arg('--user') or cfg.get('WorkingDir')!=arg('--workdir')
        or host.get('NetworkMode')!='none' or host.get('PidMode') not in ('',None)
        or host.get('DeviceRequests') not in (None,[]) or 'ALL' not in (host.get('CapDrop') or [])
        or not any(x.startswith('no-new-privileges') for x in (host.get('SecurityOpt') or []))):
        raise ValueError('container process or isolation mismatch')
    for env in [cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env']:
        key=env.split('=',1)[0]+'='
        if [v for v in cfg.get('Env',[]) if v.startswith(key)]!=[env]:raise ValueError('container environment mismatch')
    if [v for v in cfg.get('Env',[]) if v.startswith('CUDA_VISIBLE_DEVICES=')]!=['CUDA_VISIBLE_DEVICES=0']:
        raise ValueError('CUDA device visibility mismatch')
    devices=host.get('Devices') or []
    if len(devices)!=len(DEVICES) or {(x.get('PathOnHost'),x.get('PathInContainer'),x.get('CgroupPermissions')) for x in devices}!={(x,x,'rwm') for x in DEVICES}:
        raise ValueError('actual device list mismatch')
    wanted=[]
    for i,x in enumerate(cmd[:-1]):
        if x!='--mount':continue
        parts=cmd[i+1].split(','); attrs=dict(p.split('=',1) for p in parts if '=' in p)
        if attrs.get('type')!='bind' or 'src' not in attrs or 'dst' not in attrs:raise ValueError('bad bind specification')
        wanted.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
    mounts=obj.get('Mounts') or []
    found=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in mounts if m.get('Type')=='bind']
    if len(found)!=len(mounts) or sorted(found)!=sorted(wanted):raise ValueError('actual bind mounts mismatch')
    return {'container_id':obj['Id'],'host_pid':state['Pid'],'image':image,'mount_count':len(found)}

def main(): raise SystemExit('locked: CPU validation helpers do not authorize GPU')
if __name__=='__main__':main()
