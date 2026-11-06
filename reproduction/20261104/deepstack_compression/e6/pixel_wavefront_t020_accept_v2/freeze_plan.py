import hashlib,json,os
from pathlib import Path
A=Path(__file__).resolve().parent;B=A.parents[1];D=B/'e6/pixel_wavefront_model_t020_run_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
s=read(A/'monitor_cpu001/started.json');t=read(A/'monitor_cpu001/terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==A/'test_verify.py'
plan={'stage':'pre-run formal identity acceptance','verifier_sha256':sha(A/'verify.py'),'test_sha256':sha(A/'test_verify.py'),'cpu_started_sha256':sha(A/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(A/'monitor_cpu001/terminal.json'),'execution_plan_sha256':sha(D/'execution_plan001.json'),'GPU_results_opened':False,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'limitations':['Formal acceptance is pre-run but verifier requires an additional independent post-run review.','KV fingerprints are runtime records, raw KV tensors are not archived.','Candidate implementation author reviewed earlier no-GPU preflights.']}
with (A/'verification_plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
