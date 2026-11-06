"""Independent prelaunch audit of retained CPU preflight; no GPU execution."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_microbench_pilot';I=R/'e1_independent_review';sys.path.insert(0,str(H));import pilot_admission as g

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',required=True);o=Path(p.parse_args().out).resolve();assert o==(H/'independent_final_review.json').resolve()
pre=g.admit(preflight_only=True,host=True);s=mon(H/'monitor_worker_preflight002');out=H/'worker_preflight002';launch=read(out/'launch.json');ex=read(out/'exit.json');result=read(out/'worker_preflight.json');source={str(p):sha(p) for p in g.sources()}
assert len(source)==58 and pre['source_sha256']==source==result['source_sha256'] and launch['approval_bundle']==pre and launch['worker_sha256']==sha(H/'pilot_worker.py')
assert s['cwd']==str(R) and s['command'][1:]==['e4_c_microbench_pilot/pilot_launch.py','--run-id','002','--preflight-only'] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and s['timeout_seconds']==1500 and s['grace_seconds']==90
assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['failure_type'] is None and ex['timed_out'] is False and ex['client_exited_while_container_running'] is False
assert result['status']=='passed' and result['jobs']==60 and result['cuda_initialized'] is False and result['model_loaded'] is False and result['predictions_opened'] is True and result['GT_opened'] is False and result['scoring_performed'] is False and len(result['weight_stats_before'])==4
cid=ex['container_identity']['container_id'];data=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,check=True,timeout=30).stdout);assert len(data)==1;inspection=g.validate_container(launch,ex,data[0],gpu=False);assert inspection['container_id']==cid and inspection['devices']==[]
cpu=H/'cpu_check_010/verification.json';cs=H/'monitor_cpu010/started.json';ct=H/'monitor_cpu010/terminal.json';mon(H/'monitor_cpu010')
proof={'status':'passed','independent_review':True,'preflight_only':False,'reviewer_sha256':sha(__file__),'source_sha256':source,'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'GT_opened':False,'worker_launch_sha256':sha(out/'launch.json'),'worker_exit_sha256':sha(out/'exit.json'),'worker_started_sha256':sha(H/'monitor_worker_preflight002/started.json'),'worker_terminal_sha256':sha(H/'monitor_worker_preflight002/terminal.json'),'worker_result_sha256':sha(out/'worker_preflight.json'),'container_id':cid,'limitations':['Preflight parsed accepted E1 DEV mixed prediction journal only for identity fields.','Component-only diagnostic: DS staging excludes decoder addition, not TTFT.','This review validates prelaunch only; no GPU has run under this proof.'],'errors':[]}
with o.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
