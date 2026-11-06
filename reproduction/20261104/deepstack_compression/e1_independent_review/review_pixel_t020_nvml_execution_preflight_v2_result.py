"""Independent actual no-device Pixel NVML dependency preflight review."""
import argparse,hashlib,importlib.util,json,os,subprocess
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_nvml_execution_preflight_v2';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';P=B/'e6/pixel_wavefront_t020_timing_final_v9';OUT=I/'pixel_t020_nvml_execution_preflight_v2_result_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor identity/exit')
 return s
def inspect(cid):return json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
def mounts(obj):return sorted((m['Type'],m['Source'],m['Destination'],m['RW']) for m in obj['Mounts'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);out=a.parse_args().out.resolve()
 if out!=OUT.resolve() or out.exists():raise ValueError('output')
 mon=D/'monitor_preflight001';run=D/'run_preflight001';plan=read(D/'plan001.json');w=read(run/'worker_preflight.json');launch=read(run/'launch.json');ident=read(run/'container_identity.json');ex=read(run/'exit.json');s=supervised(mon,D/'launch.py',run)
 rev=I/'pixel_t020_nvml_execution_preflight_v2_review.json';reviewer=I/'review_pixel_t020_nvml_execution_preflight_v2.py';supervised(I/'monitor_pixel_t020_nvml_execution_preflight_v2',reviewer,rev);q=read(rev)
 src=plan['source_sha256'];assert len(src)==600 and all(sha(k)==v for k,v in src.items()) and q['status']=='passed' and q['preflight_only'] is True and q['GPU_authorized'] is False and q['source_sha256']==src and q['reviewer_sha256']==sha(reviewer) and q['plan_sha256']==sha(D/'plan001.json')
 assert s['worker_pid']==launch['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['interrupted'] is None and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None and ex['container_identity']==ident and ex['controller']==launch['controller']
 cid=ident['container_id'];obj=inspect(cid);prior=inspect(read(P/'run_preflight001/exit.json')['container_identity']['container_id']);state=obj['State'];host=obj['HostConfig'];config=obj['Config'];pconfig=prior['Config'];phost=prior['HostConfig']
 assert obj['Id']==cid and obj['Image']==ident['image']==prior['Image'] and state['Running'] is False and state['Pid']==state['ExitCode']==0 and state['OOMKilled'] is False and host['NetworkMode']=='none' and host['Devices'] in (None,[]) and host['DeviceRequests'] in (None,[]) and 'CUDA_VISIBLE_DEVICES=' in config['Env'] and config['Cmd']==[str(D/'worker.py'),'--out',str(run/'worker_preflight.json')]
 assert all(config.get(k)==pconfig.get(k) for k in ('Entrypoint','User','WorkingDir')) and len({v.split('=',1)[0] for v in config['Env']})==len(config['Env']) and sorted(config['Env'])==sorted(pconfig['Env']) and all(host.get(k)==phost.get(k) for k in ('NetworkMode','CapDrop','SecurityOpt','CgroupnsMode')) and len(obj['Mounts'])==12 and mounts(obj)==mounts(prior) and cid in ident['host_cgroup']
 assert w['status']=='passed' and w['model_loaded'] is False and w['cuda_initialized'] is False and w['GPU_authorized'] is False and w['GT_opened'] is False and w['existing_DEV_predictions_opened'] is True and w['container_id']==cid and w['approval']==launch['approval'] and w['approval']['plan_sha256']==sha(D/'plan001.json') and w['approval']['review_sha256']==sha(rev)
 entries=w['actual_entry_smokes'];assert set(entries)=={'worker','controller','sampler'}
 for name in entries:
  hostsmoke=read(E/('monitor_'+name+'_smoke001')/'stdout.log');assert entries[name]==hostsmoke and entries[name]['status']=='passed' and entries[name]['model_loaded'] is False
 assert entries['worker']['references']==9 and entries['worker']['cuda_initialized'] is False and entries['controller']['gpu_queried'] is False and entries['sampler']['gpu_queried'] is False
 weights=w['weight_stat_preflight'];assert len(weights['expected'])==4
 for x in weights['expected']:
  st=Path(x['path']).stat();assert [st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_dev]==[x['size'],x['mtime_ns'],x['ctime_ns'],x['inode'],x['device']]
 weightgate=P/'gpu_gate.py';assert sha(weightgate)==src[str(weightgate)];spec=importlib.util.spec_from_file_location('_independent_pixel_nvml_stat_gate',weightgate);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);assert g.weight_preflight()==weights
 proof={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'plan_sha256':sha(D/'plan001.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),'launch_sha256':sha(run/'launch.json'),'identity_sha256':sha(run/'container_identity.json'),'exit_sha256':sha(run/'exit.json'),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'prelaunch_review_sha256':sha(rev),'reviewer_sha256':sha(__file__),'source_sha256':src,'errors':[],'limitations':['Nine accepted DEV reference identities parsed; no GT.','No model, GPU, CUDA initialization or NVML query was exercised.','Four weight files checked by endpoint stat; transient modifications cannot be excluded.','GPU plan, real sampler handshake and independent GPU prelaunch remain separate requirements.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
