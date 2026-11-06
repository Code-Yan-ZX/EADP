"""Independent GPU-prelaunch review of real selector no-GPU worker preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e5/selector_real128';R=B/'e1_independent_review';O=D/'worker_preflight002';sys.path.insert(0,str(D))
from admission import prior,sources,admit,validate_container
from identity_gate import sha
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==D/'independent_final_review.json'
def get(x):return json.loads(x.read_text())
def term(mon):
 s=get(mon/'started.json');t=get(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
term(R/'monitor_selector_real128_preflight');term(R/'monitor_selector_real128_cpu_tests');term(D/'monitor_cpu001');s=term(D/'monitor_worker_preflight002');assert s['timeout_seconds']==1500 and s['worker_pid']==s['process_group']
source={str(x):sha(x) for x in sources()};cpu=get(D/'cpu_check001/verification.json');assert len(source)==114 and cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after']==source
identity=prior(host=True);pre=admit(preflight_only=True,host=True);l=get(O/'launch.json');ex=get(O/'exit.json');w=get(O/'worker_preflight.json');assert l['approval_bundle']==pre and w['source_sha256']==source
assert w['status']=='passed' and w['images']==96 and w['cases']==288 and w['e4_identity_records']==288 and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GT_opened'] is False and w['scoring_performed'] is False and w['predictions_opened'] is True and len(w['weight_stats_before'])==4
assert w['triton_cache']=={'path':str(D/'triton_cache'),'writable':True,'uid':1000,'gid':1000}
assert s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];snapshot=validate_container(l,ex,ins,gpu=False);assert not snapshot['devices'] and ins['Config']['User'].split(':')==['1000','1000'] and 'CUDA_VISIBLE_DEVICES=' in ins['Config']['Env']
cmd=s['command'];assert cmd.count('--run-id')==1 and cmd[cmd.index('--run-id')+1]=='002' and cmd.count('--preflight-only')==1 and (Path(s['cwd'])/cmd[1]).resolve()==D/'launch.py'
paths={'worker_launch':O/'launch.json','worker_exit':O/'exit.json','worker_started':D/'monitor_worker_preflight002/started.json','worker_terminal':D/'monitor_worker_preflight002/terminal.json','worker_result':O/'worker_preflight.json'}
r={'status':'passed','independent_review':True,'preflight_only':False,'reviewer_sha256':sha(__file__),'source_sha256':source,'cpu_sha256':sha(D/'cpu_check001/verification.json'),'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),'GT_opened':False,'prior_identity':identity,'container':snapshot,'scope':'GPU prelaunch prerequisites; GPU scheduling by root','predictions_opened':True,'prediction_scope':'accepted E1 and E4 DEV mixed records parsed for input and selector identity','scoring_performed':False,'limitations':['96-image sample is exploratory after prior selector timing','E4 full panel retains summaries and hashes, not high-dimensional arrays','preflight hashed four weights; this review binds its evidence without rehashing payload','cases failing identity are retained and corresponding candidate is not timed'],'errors':[]};r.update({k+'_sha256':sha(v) for k,v in paths.items()});a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
