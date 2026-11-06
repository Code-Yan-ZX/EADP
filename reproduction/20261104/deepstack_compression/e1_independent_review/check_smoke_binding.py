"""Independently bind completed E1 smoke verifier/reviewer to frozen evidence."""
import hashlib,json,os
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression')
e=B/'e1'; run=e/'e1smoke001'; r=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
p=read(e/'e1smoke001_verification.json');v=read(r/'e1smoke001_review.json');launch=read(e/'e1smoke001_launch.json'); checks={}
for k,path in [('frozen_sha256',run/'frozen.json'),('raw_predictions_sha256',run/'raw_predictions.jsonl'),('generated_arms_sha256',run/'generated_arms.jsonl'),('validated_quartets_sha256',run/'validated_quartets.jsonl'),('worker_complete_sha256',run/'worker_complete.json'),('launch_sha256',e/'e1smoke001_launch.json'),('container_exit_sha256',e/'e1smoke001_exit.json'),('supervisor_terminal_sha256',e/'monitore1smoke001/terminal.json')]:
 checks[k]=p.get(k)==v.get(k)==sha(path)
checks['review_proof']=v['run_proof_sha256']==sha(e/'e1smoke001_verification.json')
checks['review_verifier_terminal']=v['verifier_terminal_sha256']==sha(e/'monitorverifye1smoke001/terminal.json')
checks['verifier_source']=p['verifier_sha256']==sha(e/'e1smoke001_sources/verify_smoke.py')
checks['reviewer_source']=v['reviewer_sha256']==sha(r/'review_smoke.py')
for path,h in launch['source_sha256'].items():checks['source:'+path]=sha(e/'e1smoke001_sources'/Path(path).name)==h
for name,path in [('experiment',e/'monitore1smoke001/terminal.json'),('verifier',e/'monitorverifye1smoke001/terminal.json'),('reviewer',r/'monitor_e1smoke001/terminal.json')]:
 t=read(path);checks['terminal:'+name]=t.get('reason')=='completed' and t.get('worker_returncode')==0 and not t.get('remaining_process_group')
checks['review_status']=v['status']=='passed' and v['n_verified']==32 and not v['errors']
checks['verifier_status']=p['status']=='passed' and p['n_verified']==32 and not p['errors']
result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'proof_sha256':sha(e/'e1smoke001_verification.json'),'review_sha256':sha(r/'e1smoke001_review.json'),'review_terminal_sha256':sha(r/'monitor_e1smoke001/terminal.json'),'verifier_terminal_sha256':sha(e/'monitorverifye1smoke001/terminal.json'),'binding_script_sha256':sha(Path(__file__))}
with (r/'e1smoke001_binding.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(result));raise SystemExit(0 if result['status']=='passed' else 1)
