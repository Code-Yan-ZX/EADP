"""Independent M3 execution-chain prelaunch review; no labels or GPU used."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent;HERE=BASE/'m3_oracle_exec'
sys.path.insert(0,str(HERE))
import runtime_gate,container_identity
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.out.resolve()==HERE/'independent_runtime_review.json'
 cpu=HERE/'cpu_check_009/verification.json';cs=HERE/'monitor_cpu009/started.json';ct=HERE/'monitor_cpu009/terminal.json'
 provisional=HERE/'independent_runtime_preflight.json';rs=BASE/'e1_independent_review/monitor_m3oracle_runtime_preflight/started.json';rt=BASE/'e1_independent_review/monitor_m3oracle_runtime_preflight/terminal.json'
 reviewer=BASE/'e1_independent_review/review_m3oracle_runtime_preflight.py'
 runtime_gate.supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);runtime_gate.supervised(rs,rt,reviewer,provisional)
 c=read(cpu);r=read(provisional);source=c['source_sha256_after'];assert c['status']=='passed' and source==c['source_sha256_before'] and len(source)==127 and all(sha(p)==h for p,h in source.items())
 assert r['status']=='passed' and r['preflight_only'] is True and r['independent_review'] is True and r['reviewer_sha256']==sha(reviewer) and r['cpu_proof_sha256']==sha(cpu) and r['cpu_started_sha256']==sha(cs) and r['cpu_terminal_sha256']==sha(ct) and r['source_sha256']==source
 prior=runtime_gate.validate(preflight_only=True);assert prior['cpu_sha256']==sha(cpu) and prior['review_sha256']==sha(provisional)
 pp=HERE/'preflight_007.json';ps=HERE/'monitor_preflight007/started.json';pt=HERE/'monitor_preflight007/terminal.json';out=HERE/'preflight_007'
 s,t=read(ps),read(pt);command=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']==s['process_group']
 assert [(cwd/x).resolve() for x in command if Path(x).name=='preflight.py']==[(HERE/'preflight.py').resolve()] and command.count('--run-id')==1 and command[command.index('--run-id')+1]=='007' and command.count('--cpu-id')==1 and command[command.index('--cpu-id')+1]=='009'
 proof=read(pp);launch=read(out/'launch.json');ex=read(out/'exit.json');receipt=out/'host_receipt.json';result=proof['result'];cmd=launch['command'];plan=read(HERE/'runtime_plan.json');runtime_gate.check_budget(plan,time.time())
 assert proof['status']=='passed' and proof['exit']==ex and proof['container_running'] is False and proof['container_pid']==0 and proof['container_exit_code']==0
 assert proof['launch_sha256']==sha(out/'launch.json') and proof['exit_sha256']==sha(out/'exit.json') and proof['log_sha256']==sha(out/'launcher.log') and proof['receipt_sha256']==sha(receipt) and proof['cpu_proof_sha256']==sha(cpu) and proof['preflight_sha256']==sha(HERE/'preflight.py')
 assert launch['cpu_proof_sha256']==sha(cpu) and launch['worker_sha256']==sha(HERE/'runtime_worker.py') and launch['host_receipt']=={'path':str(receipt),'sha256':sha(receipt)} and read(receipt)==prior
 assert ex['controller']['pid']==s['worker_pid'] and ex['returncode']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['errors']==[] and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
 assert result['status']=='passed' and result['cuda_initialized'] is False and result['model_loaded'] is False and result['GT_opened'] is False and result['predictions_opened'] is False and result['inputs_verified']==384 and result['sample_verified']==128 and result['weights_sha256_verified']==4 and result['prediction_journal_byte_hash_deferred'] is True
 assert cmd.count(str(HERE/'runtime_worker.py'))==1 and cmd.count('--preflight-only')==1 and cmd.count('--execute')==1 and cmd.count('--out')==1 and Path(cmd[cmd.index('--out')+1]).resolve()==out
 assert 'CUDA_VISIBLE_DEVICES=' in cmd and all('/dev/nvidia' not in str(x) for x in cmd)
 ident=ex['container_identity'];obj=json.loads(subprocess.check_output(['docker','inspect',ident['container_id']],text=True))[0];state=obj['State'];source_plan=read(container_identity.PLAN)
 assert obj['Id']==ident['container_id']==proof['container_id'] and obj['Image']==ident['image']==source_plan['container_image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
 assert not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests'] and obj['HostConfig']['NetworkMode']=='none'
 assert obj['Config']['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and obj['Config']['Cmd']==cmd[cmd.index(source_plan['container_image'])+1:] and obj['Config']['User']==cmd[cmd.index('--user')+1] and obj['Config']['WorkingDir']==cmd[cmd.index('--workdir')+1]
 envs=[cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'];assert all(x in obj['Config']['Env'] for x in envs)
 last=None
 for line in (out/'launcher.log').read_text().splitlines():
  try:last=json.loads(line)
  except ValueError:pass
 assert last==result
 q={'status':'passed','independent_review':True,'GT_opened':False,'gpu_used':False,'scoring_performed':False,'errors':[],
    'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),
    'preflight_sha256':sha(pp),'preflight_started_sha256':sha(ps),'preflight_terminal_sha256':sha(pt),'plan_sha256':sha(HERE/'runtime_plan.json'),'source_sha256':source,
    'provisional_review_sha256':sha(provisional),'preflight_receipt_sha256':sha(receipt),
    'limits':['No GPU, labels, predictions, model loading or scoring were used by this review/preflight.',
              'Prediction journal byte hashing is deferred to actual runtime before any reference access.',
              'Natural full-panel tensor arrays are not retained; audit relies on recorded summaries and previously accepted diagnostics.',
              'Runtime requires root-coordinated exclusive GPU and a fresh budget check; no performance or accuracy result is implied.']}
 with a.out.open('x') as f:json.dump(q,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(source),'preflight':'007'}))
if __name__=='__main__':main()
