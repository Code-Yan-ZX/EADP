"""Single input fused-greedy prototype prelaunch audit, no broad equivalence claim."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e5/fused_selector';R=B/'e1_independent_review';M=R/'monitor_fused_selector_prelaunch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json'
plan=get(D/'plan001.json');assert sha(D/'plan001.json').startswith('aae205e9')
for f,h in plan['sha256'].items():assert sha(f)==h,f
term(R/'monitor_selector_fused_cpu001');term(R/'monitor_component_pilot001');term(D/'monitor_test001');term(R/'monitor_fused_selector_cpu_review');assert 'Ran 3 tests' in (R/'monitor_fused_selector_cpu_review/stderr.log').read_text();review=get(R/'selector_fused_cpu001_review.json');assert review['status']=='passed' and review['cpu_order_matches_recorded_gpu'] and review['artifact_sha256']==sha(B/'e5/selector_fused_cpu/numeric_inputs001.pt')
pre=get(D/'preflight001.json');assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['input_sha256_verified']==review['artifact_sha256']
s=term(D/'monitor_preflight001');launch=get(D/'run_preflight001/launch.json');ex=get(D/'run_preflight001/exit.json');assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none' and ins['Config']['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
cmd=launch['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(q.split('=',1) if '=' in q else (q,True) for q in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if c=='--env':assert cmd[i+1] in ins['Config']['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
r={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'scope':'single numerical input and synthetic GPU identity gate prior to component timing','limitations':['Triton and torch reduction order and rounding can differ; tie and near-tie tests do not prove universal selection identity','timing includes repeated kernel launches and allocations; only single fixed numeric input, not full model or deployed method','each measured selection revalidated; any observed mismatch invalidates run'],'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
