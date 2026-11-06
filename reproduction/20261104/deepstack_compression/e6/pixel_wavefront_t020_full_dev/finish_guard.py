"""Actual run supervisor and retained container checks before any result read."""
import json
import subprocess
from pathlib import Path
import baseline

D=Path(__file__).resolve().parent
def validate():
    s,_=baseline.supervised(D/'monitor_run001',D/'launch.py')
    launch=baseline.read(D/'run_run001/launch.json');ex=baseline.read(D/'run_run001/exit.json')
    if s.get('command',[])[-2:]!=['--stage','run']:raise ValueError('wrong run supervisor command')
    root=D.parents[4];author_plan=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
    import launch
    runtime=launch.runtime_module()
    expected=runtime.container_command(baseline.read(author_plan),[str(D/'worker.py'),'--out',str(D/'run_run001/observations')],True,launch['container_name'])
    if expected.count('--rm')!=1:raise ValueError('unexpected fixed command')
    expected.remove('--rm')
    if launch.get('command')!=expected:raise ValueError('launch differs from fixed command')
    if (launch.get('stage')!='run' or launch.get('plan_sha256')!=baseline.sha(D/'plan002.json')
        or s['worker_pid']!=launch.get('controller',{}).get('pid') or ex.get('controller')!=launch.get('controller')
        or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False
        or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]):
        raise ValueError('incomplete run exit')
    result=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15)
    obj=json.loads(result.stdout)[0];state=obj['State']
    if (obj['Id']!=ex['last_inspect']['id'] or obj['Image']!=ex['container_identity']['image']
        or state['Running'] or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled']):
        raise ValueError('actual Docker state mismatch')
    if obj['Config']['Cmd']!=[str(D/'worker.py'),'--out',str(D/'run_run001/observations')] or obj['HostConfig']['NetworkMode']!='none':
        raise ValueError('actual container process mismatch')
    devices=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
    if {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in obj['HostConfig']['Devices']}!={(x,x,'rwm') for x in devices}:
        raise ValueError('actual GPU devices mismatch')
    mounts=[];cmd=launch['command']
    for i,value in enumerate(cmd[:-1]):
        if value=='--mount':
            v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
        if value=='--env' and cmd[i+1] not in obj['Config']['Env']:raise ValueError('environment mismatch')
    actual=sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
    if sorted(mounts)!=actual:raise ValueError('actual mounts mismatch')
    return {'container_id':obj['Id'],'image':obj['Image'],'run_started_sha256':baseline.sha(D/'monitor_run001/started.json'),
            'run_terminal_sha256':baseline.sha(D/'monitor_run001/terminal.json'),'run_exit_sha256':baseline.sha(D/'run_run001/exit.json')}
