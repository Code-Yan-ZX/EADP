import argparse,hashlib,json,os,subprocess
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression'); D=B/'e6/pixel_wavefront_t020_weight_stat_probe';R=B/'e1_independent_review'; P=B/'e6/pixel_wavefront_model_t020'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();plan=read(D/'plan001.json');w=read(D/'run001/worker.json');ex=read(D/'run001/exit.json');launch=read(D/'run001/launch.json');s=read(D/'monitor001/started.json');t=read(D/'monitor001/terminal.json')
 assert all(sha(p)==h for p,h in plan['source_sha256'].items()) and plan['GPU_authorized'] is False and plan['model_loaded'] is False and plan['GT_opened'] is False
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']==ex['controller']['pid'] and s['command'][-1]=='e6/pixel_wavefront_t020_weight_stat_probe/launch.py'
 assert ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
 id=ex['container_identity']['container_id']; d=json.loads(subprocess.check_output(['docker','inspect',id]))[0]; priorid='047a9c15ab78e6c01acf0f99eb2b823391a0bfc7951313acdccc8a330846ff3d'; p=json.loads(subprocess.check_output(['docker','inspect',priorid]))[0]
 assert d['Id']==id and d['State']['Running'] is False and d['State']['Pid']==0 and d['State']['ExitCode']==0 and d['Image']==p['Image']
 for key in ('Entrypoint','User','WorkingDir'):assert d['Config'][key]==p['Config'][key]
 assert d['Config']['Cmd']==[str(D/'worker.py')] and sorted(d['Config']['Env'])==sorted(p['Config']['Env']) and len({x.split('=',1)[0] for x in d['Config']['Env']})==len(d['Config']['Env'])
 for key in ('NetworkMode','PidMode','Devices','DeviceRequests','Binds','CgroupnsMode'):assert d['HostConfig'][key]==p['HostConfig'][key]
 assert d['HostConfig']['NetworkMode']=='none' and not d['HostConfig']['Devices'] and not d['HostConfig']['DeviceRequests']
 assert sorted(json.dumps(x,sort_keys=True) for x in d['Mounts'])==sorted(json.dumps(x,sort_keys=True) for x in p['Mounts']) and len(d['Mounts'])==12
 assert w['status']=='passed' and w['gate_sha256']==sha(B/'e6/pixel_wavefront_t020_timing_final_v4/gpu_gate.py') and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GT_opened'] is False and w['GPU_used'] is False and w['pid']==1 and w['hostname']==id[:12] and id in w['self_cgroup']
 prior=read(P/'run_preflight001/worker_preflight.json'); review=R/'pixel_wavefront_model_t020_preflight_result_review.json'; assert w['weights']['worker_preflight_sha256']==sha(P/'run_preflight001/worker_preflight.json') and w['weights']['review_sha256']==sha(review) and w['weights']['expected']==prior['weight_stats'] and len(prior['weight_stats'])==4
 for x in prior['weight_stats']:
  st=Path(x['path']).stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==[x['device'],x['inode'],x['size'],x['mtime_ns'],x['ctime_ns']]
 assert launch['name'] and ex['container_identity']['host_argv'][-1]==str(D/'worker.py')
 out={'status':'passed','independent_review':True,'scope':'supplemental no-GPU bind-mount stat mapping only','GT_opened':False,'GPU_used':False,'model_loaded':False,'weights_verified':4,'probe_plan_sha256':sha(D/'plan001.json'),'worker_sha256':sha(D/'run001/worker.json'),'launch_sha256':sha(D/'run001/launch.json'),'exit_sha256':sha(D/'run001/exit.json'),'started_sha256':sha(D/'monitor001/started.json'),'terminal_sha256':sha(D/'monitor001/terminal.json'),'source_sha256':plan['source_sha256'],'reviewer_sha256':sha(__file__),'container_id':id,'reference_container_id':priorid,'limitations':['Compares current host and container observed stat identity; does not hash payload again or exclude a transient change.','Container inspector environment and mounts compared as unordered full records; no model, CUDA or GPU initialized.'],'errors':[]}
 with open(a.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
