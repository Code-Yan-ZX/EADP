"""Independent code review for M3's no-GPU, no-predictions worker preflight."""
import argparse,hashlib,json,sys,time
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
 assert a.out.resolve()==HERE/'independent_runtime_preflight.json'
 cpu=HERE/'cpu_check_009/verification.json';start=HERE/'monitor_cpu009/started.json';end=HERE/'monitor_cpu009/terminal.json';s,t=read(start),read(end)
 cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']==s['process_group']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[(HERE/'cpu_validation.py').resolve()]
 assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==cpu.parent
 c=read(cpu);assert c['status']=='passed' and c['GT_opened'] is False and c['real_model_loaded'] is False and c['generation_performed'] is False and c['cuda_visible_devices']==''
 source=c['source_sha256_after'];assert len(source)==127 and source==c['source_sha256_before'] and all(sha(p)==h for p,h in source.items())
 assert all(source.get(str(p.resolve()))==sha(p) for p in HERE.glob('*.py'))
 tests=c['tests'];assert len(tests)==17 and len({x['test'] for x in tests})==17
 for test in tests:
  assert test['returncode']==0 and test['timeout'] is False
  for stream in ('stdout','stderr'):assert sha(cpu.parent/(test['test']+'.'+stream))==test[stream+'_sha256']
  assert (HERE/test['test']).resolve().as_posix() in test['command']
 planpath=HERE/'runtime_plan.json';plan=read(planpath);runtime_gate.check_budget(plan,time.time())
 assert sha(plan['sample_path'])==plan['sample_sha256'] and sha(plan['amendment_path'])==plan['amendment_sha256']
 sample=read(plan['sample_path']);assert len(sample['examples'])==128 and sample['GT_opened'] is False and sample['predictions_opened'] is False
 assert len({(x['task'],x['group_id']) for x in sample['examples']})==128
 # Actual future Docker command is reconstructed from the fixed, hashed plan.
 planned,command,name=container_identity.expected_command(HERE/'runtime_REVIEW',999)
 assert command.count(str(HERE/'runtime_worker.py'))==1 and '--rm' not in command and name=='eadp-m3-oracle-REVIEW-999'
 result={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'gpu_used':False,'scoring_performed':False,
         'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(start),'cpu_terminal_sha256':sha(end),
         'plan_sha256':sha(planpath),'source_sha256':source,
         'limits':['Only no-GPU worker preflight is approved; no real labels, predictions, or model generation.',
                   'Prediction journal byte hashing is deferred until actual execution and must precede any reference access.',
                   'Cooperative per-question checkpoints are backed by independent outer container and supervisor hard timeouts.']}
 with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','scope':'preflight_only','sources':len(source)}))
if __name__=='__main__':main()
