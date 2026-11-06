"""Independent prelaunch audit: one-image PixelPrune GPU diagnostic only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/ziprerank_qtext_smoke_v2';M=B/'e1_independent_review/monitor_ziprerank_qtext_v2_prelaunch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json'
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='199fddd1bae619e24711c808922b6d72685fb1af18f1883ff5439bcb763e6b0b'
for k,v in plan['sha256'].items():assert sha(k)==v,k
assert plan['forced_generated_tokens']==33 and plan['forced_decode_calls_after_prefill']==32
pre=get(D/'preflight001.json');assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['weight_sha256_verified']==4
s=term(D/'monitor_preflight001');term(B/'e1_independent_review/monitor_ziprerank_qtext_v2_cpu_review');term(D/'monitor_test001')
assert 'Ran 11 tests' in (B/'e1_independent_review/monitor_ziprerank_qtext_v2_cpu_review/stderr.log').read_text()
launch=get(D/'run_preflight001/launch.json');ex=get(D/'run_preflight001/exit.json');assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none';assert ins['Config']['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
cmd=launch['command'];mount_args=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--mount'];mounts=[]
for arg in mount_args:
 v=dict(q.split('=',1) if '=' in q else (q,True) for q in arg.split(','));mounts.append((v['src'],v['dst'],'readonly' not in v))
assert set(mounts)=={(x['Source'],x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind'}
started=get(M/'started.json');result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'gpu_authorized':'subject to root scheduling','cpu_tests':11,'scope':'single-image forced 33-token diagnostics, author QI plus local RoPE and user-text adaptation; not quality or throughput','limitations':['local RoPE correction and user-text selection are not original author behavior','user text contains instructions and options; not strictly question-only','only summaries/hashes retained for high dimensional tensors','no GPU result yet'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
