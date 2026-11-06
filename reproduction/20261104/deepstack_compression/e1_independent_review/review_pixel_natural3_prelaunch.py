"""Independent prelaunch audit: three-anchor PixelPrune natural generation diagnostic only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_natural3_v3';M=B/'e1_independent_review/monitor_pixel_natural3_prelaunch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json'
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='972484e2480124136e2e64c671de03dd02602fc9a42408a17962e475d6f834e0'
for k,v in plan['sha256'].items():assert sha(k)==v,k
assert len(plan['anchors'])==3 and len({x['task'] for x in plan['anchors']})==3 and plan['threshold']==0 and plan['method']=='pred_2d' and plan['metric']=='max' and plan['generation']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.8,'top_k':20,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
pre=get(D/'preflight001.json');assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['weight_sha256_verified']==4
s=term(D/'monitor_preflight001');term(B/'e1_independent_review/monitor_pixel_natural3_cpu_review');term(D/'monitor_test004')
assert 'Ran 8 tests' in (B/'e1_independent_review/monitor_pixel_natural3_cpu_review/stderr.log').read_text()
launch=get(D/'run_preflight001/launch.json');ex=get(D/'run_preflight001/exit.json');assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none';assert ins['Config']['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
cmd=launch['command'];mount_args=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--mount'];mounts=[]
for arg in mount_args:
 v=dict(q.split('=',1) if '=' in q else (q,True) for q in arg.split(','));mounts.append((v['src'],v['dst'],'readonly' not in v))
assert set(mounts)=={(x['Source'],x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind'}
started=get(M/'started.json');result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'gpu_authorized':'subject to root scheduling','cpu_tests':8,'scope':'three DEV anchor natural generations with author patch before model loading; no quality score or timing','limitations':['threshold zero produces data-dependent visual budgets, not equal K','generation top_p=0.8 and top_k=20 differ from E1 0.001 and 1 even though do_sample=false; no complete recipe identity claim','full processor tensor hashes not recorded; no processor tensor identity claim','high-dimensional logits, DS and KV not archived; online comparison only','CPU preflight imports/patches actual model class but does not process actual images; no GPU result yet'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
