"""Review the retained actual no-device timing import container."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_identity3_execution';R=D/'cpu_preflight001';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervision')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_eadp_identity3_execution_preflight_result_review.json'
 if a.out.resolve()!=dest or dest.exists():raise ValueError('output')
 planfile=D/'cpu_plan001.json';plan=read(planfile);review=I/'rres_eadp_identity3_execution_cpu_review.json';reviewer=I/'review_rres_eadp_identity3_execution_cpu.py';terminal(I/'monitor_rres_eadp_identity3_execution_cpu',reviewer,review);q=read(review);s=terminal(D/'monitor_cpu_preflight001',D/'cpu_launch.py',R)
 if q['status']!='passed' or q['independent_review'] is not True or q['preflight_only'] is not True or q['GPU_authorized'] is not False or q['GT_opened'] is not False or q['errors']!=[] or q['source_sha256']!=plan['source_sha256'] or q['plan_sha256']!=sha(planfile) or q['reviewer_sha256']!=sha(reviewer) or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('preflight evidence')
 launch=read(R/'launch.json');identity=read(R/'container_identity.json');ex=read(R/'exit.json');worker=read(R/'worker_preflight.json');cmd=launch['command'];cid=identity['container_id']
 if launch['approval']!={'scope':plan['scope'],'plan_sha256':sha(planfile),'review_sha256':sha(review)} or launch['controller']['pid']!=s['worker_pid'] or ex['controller']['pid']!=s['worker_pid'] or ex['container_identity']!=identity or ex['returncode']!=0 or ex['container_exit_code']!=0 or ex['timed_out'] is not False or ex['container_stopped_verified'] is not True or ex['client_exited_while_container_running'] is not False:raise ValueError('launcher exit')
 if worker['status']!='passed' or worker['container_id']!=cid or worker['inputs']!=62 or worker['GPU_authorized'] is not False or worker['GT_opened'] is not False or worker['existing_predictions_opened'] is not True or worker['model_loaded'] is not False or worker['cuda_initialized'] is not False or worker['errors']!=[]:raise ValueError('worker result')
 expected={name:D/(name+'.py') for name in ('gpu_worker','controller','snapshot','verify','cpu_gate','gpu_gate')}
 if set(expected)!=set(worker['module_paths']):raise ValueError('module inventory')
 for name,path in expected.items():
  if Path(worker['module_paths'][name])!=path.resolve() or plan['source_sha256'].get(str(path.resolve()))!=sha(path):raise ValueError('module source '+name)

 obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];host=obj['HostConfig'];config=obj['Config']
 if obj['Id']!=cid or obj['Image']!=identity['image'] or st['Running'] is not False or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] is not False or host['NetworkMode']!='none' or host['Devices'] or host['DeviceRequests'] or '--device' in cmd or '--gpus' in cmd:raise ValueError('actual docker state/device')
 if config['Cmd']!=[str(D/'cpu_entry.py'),'--out',str(R)] or config['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or config['User']!=cmd[cmd.index('--user')+1] or config['WorkingDir']!=cmd[cmd.index('--workdir')+1]:raise ValueError('docker command/user')
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--env' and cmd[i+1] not in config['Env']:raise ValueError('docker environment')
  if x=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if len(mounts)!=12 or sorted(mounts)!=sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind') or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('mounts/source drift')
 value={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'errors':[],'plan_sha256':sha(planfile),'worker_sha256':sha(R/'worker_preflight.json'),'launch_sha256':sha(R/'launch.json'),'identity_sha256':sha(R/'container_identity.json'),'exit_sha256':sha(R/'exit.json'),'started_sha256':sha(D/'monitor_cpu_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_cpu_preflight001/terminal.json'),'prelaunch_review_sha256':sha(review),'reviewer_sha256':sha(__file__),'limitations':['No-device imports and accepted references only, not real model execution or GPU admission.','Prior runner-name collision retained as prior evidence.','Timing controller, observer, actual model identity, formal result and analysis remain separate work.']}
 with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
