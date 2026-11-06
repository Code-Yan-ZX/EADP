"""Independent audit of completed no-GPU memory worker preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;M=ROOT/'e4_c_memory';sys.path.insert(0,str(M));import admission
from identity_gate import sha

def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e4c_memory_preflight001_review.json'
 run=M/'container_preflight001';s=M/'monitor_container_preflight001/started.json';t=s.parent/'terminal.json';start=admission.terminal(s,t);cmd=start['command'];cwd=Path(start['cwd']);assert [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight_launch.py']==[M/'preflight_launch.py'] and cmd.count('--run-id')==1 and cmd[cmd.index('--run-id')+1]=='001'
 launch=read(run/'launch.json');ex=read(run/'exit.json');result=read(run/'worker_preflight.json');approval=admission.admit();admission.cpu_container(launch,ex,start,M/'preflight_entry.py')
 assert launch['approval_bundle']==result['approval_bundle']==approval and launch['source_sha256']==approval['source_sha256'] and result['status']=='passed' and result['cuda_initialized'] is False and result['model_loaded'] is False and result['memory_job_counts']=={'C-ON':24,'C-DROP':24,'C-SKIP':24} and len(result['weight_stats_before'])==4 and all(approval['source_sha256'].get(k)==v for k,v in result['source_sha256'].items())
 assert result['latency_contract_path']==str((ROOT/'e4_c_latency/contract.py').resolve())
 from latency_gate import require_light_identity
 assert result['light_identity']==require_light_identity()
 for name,expected in result['weight_stats_before'].items():
  st=Path(name).stat();assert expected==[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]

 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];state=obj['State'];cmd=launch['command'];cfg=obj['Config'];host=obj['HostConfig'];image=ex['container_identity']['image'];assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==image and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
 expected=[]
 for i,x in enumerate(cmd):
  if x=='--env':
   e=cmd[i+1];assert [v for v in cfg['Env'] if v.startswith(e.split('=',1)[0]+'=')]==[e]
  if x=='--mount':
   v=cmd[i+1].split(',');m=dict(x.split('=',1) for x in v if '=' in x);assert m['type']=='bind';expected.append((str(Path(m['src']).resolve()),m['dst'],'readonly' not in v))
 actual=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in obj['Mounts'] if m['Type']=='bind'];assert len(actual)==len(expected)==len(obj['Mounts']) and sorted(actual)==sorted(expected)
 out={'status':'passed','independent_review':True,'scope':'completed CPU-only worker preflight, no GPU authorization','GT_opened':False,'scoring_performed':False,'model_loaded':False,'gpu_used':False,'reviewer_sha256':sha(__file__),'started_sha256':sha(s),'terminal_sha256':sha(t),'launch_sha256':sha(run/'launch.json'),'exit_sha256':sha(run/'exit.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),'approval':approval,'limitations':['GPU worker/sampler handshake, isolation and process memory still require separate implementation and independent admission.']}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed'}))
if __name__=='__main__':main()
