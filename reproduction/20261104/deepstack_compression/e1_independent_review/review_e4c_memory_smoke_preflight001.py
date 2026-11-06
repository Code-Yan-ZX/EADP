"""Independent retained CPU-container preflight audit; no GPU measurement."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_memory_smoke';R=B/'e1_independent_review';O=D/'container_preflight001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();term(R/'monitor_e4c_memory_smoke_worker_preflight');s=term(D/'monitor_preflight001');review=get(R/'e4c_memory_smoke_worker_preflight_review.json');l=get(O/'launch.json');e=get(O/'exit.json');w=get(O/'worker_preflight.json');assert review['status']=='passed' and l['approval']['review_sha256']==sha(R/'e4c_memory_smoke_worker_preflight_review.json')==w['approval_review_sha256'] and w['source_sha256']==l['approval']['source_sha256']==review['source_sha256'];assert len(w['source_sha256'])==119
for f,h in w['source_sha256'].items():assert sha(f)==h,f
assert w['status']=='passed' and w['gpu_authorized'] is False and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GT_opened'] is False and w['scoring_performed'] is False and w['existing_DEV_predictions_opened'] is True and w['new_predictions_generated'] is False and w['schedule_jobs']==72 and len(w['weight_stats_before'])==4
assert s['worker_pid']==l['controller']['pid']==e['controller']['pid'] and e['returncode']==e['container_exit_code']==0 and e['container_stopped_verified'] and e['client_exited_while_container_running'] is False and e['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',e['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==e['last_inspect']['id'] and ins['Image']==e['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none' and ins['Config']['Cmd']==[str(D/'preflight_entry.py'),'--out',str(O)]
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if c=='--env':assert cmd[i+1] in ins['Config']['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind') and 'CUDA_VISIBLE_DEVICES=' in ins['Config']['Env']
files=[O/'launch.json',O/'exit.json',O/'worker_preflight.json',D/'monitor_preflight001/started.json',D/'monitor_preflight001/terminal.json',R/'e4c_memory_smoke_worker_preflight_review.json',R/'e4c_memory_runtime_container001_scope_correction.json'];r={'status':'passed','independent_review':True,'scope':'C-ON worker CPU preflight, no actual sampling','gpu_authorized':False,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'new_predictions_generated':False,'schedule_jobs':72,'source_count':len(w['source_sha256']),'container_id':ins['Id'],'files_sha256':{str(f):sha(f) for f in files},'reviewer_sha256':sha(__file__),'limitations':['no actual GPU worker, NVML sampler or GPU measurement authorized','actual weight rehash was performed by CPU preflight; independent review binds evidence, does not rehash weight payloads'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
