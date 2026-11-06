"""Independent prelaunch audit: 384-question PixelPrune DEV natural generation only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_dev384';M=B/'e1_independent_review/monitor_pixel_dev384_prelaunch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(p):return json.loads(Path(p).read_text())
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==D/'independent_prelaunch_review.json'
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='8a52b138edc2d446a03fff80caadcb0c6c0927702ea74623ef49d359f308e0ec'
for k,v in plan['sha256'].items():assert sha(k)==v,k
assert len(plan['anchors'])==384 and len({x['group_id'] for x in plan['anchors']})==384 and len({(x['task'],x['qid']) for x in plan['anchors']})==384 and len({x['task'] for x in plan['anchors']})==3 and plan['threshold']==0 and plan['method']=='pred_2d' and plan['metric']=='max' and plan['generation']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
pre=get(D/'preflight001.json');assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False and pre['weight_sha256_verified']==4
assert pre['patch_before_model'] and pre['predictions_opened'] is True and len(pre['processor_verified'])==3 and len({x['task'] for x in pre['processor_verified']})==3
s=term(D/'monitor_preflight001');term(B/'e1_independent_review/monitor_pixel_dev384_cpu_review');term(D/'monitor_test001')
assert 'Ran 9 tests' in (B/'e1_independent_review/monitor_pixel_dev384_cpu_review/stderr.log').read_text()
launch=get(D/'run_preflight001/launch.json');ex=get(D/'run_preflight001/exit.json');assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['failure_type'] is None
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,check=True,timeout=15).stdout)[0];st=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'];assert not ins['HostConfig']['Devices'] and not ins['HostConfig']['DeviceRequests'] and ins['HostConfig']['NetworkMode']=='none';assert ins['Config']['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')]
cmd=launch['command'];mount_args=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--mount'];mounts=[]
for arg in mount_args:
 v=dict(q.split('=',1) if '=' in q else (q,True) for q in arg.split(','));mounts.append((v['src'],v['dst'],'readonly' not in v))
assert set(mounts)=={(x['Source'],x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind'}
started=get(M/'started.json');result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'gpu_authorized':'subject to root scheduling','cpu_tests':9,'scope':'384 frozen DEV natural generations; patch before loading; no GT scoring or timing','limitations':['threshold zero gives dynamic visual budgets, not equal K','high-dimensional logits, DS and KV are not archived; online comparison only','CPU preflight checks actual processor on three images, all 384 to be checked at runtime','accepted E1 mixed prediction journal parsed for identity only','no GPU result yet'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
