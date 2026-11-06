"""Independent final review of the 32-image CPU container preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_microbench_full';R=B/'e1_independent_review';O=D/'worker_preflight001';sys.path.insert(0,str(D));from full_admission import accepted_pilot,sources
from identity_gate import sha
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==D/'independent_final_review.json'
def get(x):return json.loads(x.read_text())
def term(mon):
 s=get(mon/'started.json');t=get(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
term(R/'monitor_component_full_preflight_002');term(D/'monitor_cpu002');s=term(D/'monitor_worker_preflight001');prior=accepted_pilot(host=True);source={str(x):sha(x) for x in sources()};cpu=get(D/'cpu_check_002/verification.json');assert cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after']==source
l=get(O/'launch.json');ex=get(O/'exit.json');w=get(O/'worker_preflight.json');pre=get(D/'independent_preflight_review_002.json');assert pre['status']=='passed' and l['approval_bundle']['review_sha256']==sha(D/'independent_preflight_review_002.json') and l['approval_bundle']['source_sha256']==source==w['source_sha256'];assert w['status']=='passed' and w['images']==32 and w['jobs']==1920 and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GT_opened'] is False and w['scoring_performed'] is False and w['predictions_opened'] is True and len(w['weight_stats_before'])==4
assert s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none' and ins['Config']['Cmd']==[str(D/'full_worker.py'),'--out',str(O),'--preflight-only']
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if c=='--env':assert cmd[i+1] in ins['Config']['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind') and 'CUDA_VISIBLE_DEVICES=' in ins['Config']['Env']
paths={'worker_launch':O/'launch.json','worker_exit':O/'exit.json','worker_started':D/'monitor_worker_preflight001/started.json','worker_terminal':D/'monitor_worker_preflight001/terminal.json','worker_result':O/'worker_preflight.json'}
r={'status':'passed','independent_review':True,'preflight_only':False,'reviewer_sha256':sha(__file__),'source_sha256':source,'cpu_sha256':sha(D/'cpu_check_002/verification.json'),'cpu_started_sha256':sha(D/'monitor_cpu002/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu002/terminal.json'),'GT_opened':False,'pilot_identity':prior,'scope':'GPU prelaunch prerequisites, scheduling by root','predictions_opened':True,'scoring_performed':False,'limitations':['32-image expansion postdates pilot result','individual measurement outputs validated online but not independently archived','DS staging excludes decoder addition and is not TTFT','4 weight shards hashed by preflight; this review binds proof without payload rehash'],'errors':[]};r.update({k+'_sha256':sha(v) for k,v in paths.items()});a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
