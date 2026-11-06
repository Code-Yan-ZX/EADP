"""Pre-run source and verification freeze; never loads a model or starts GPU."""
import json,os
from pathlib import Path
import gate
D=Path(__file__).resolve().parent;B=D.parents[1];RS=B/'e6/random_smoke';A=B/'e6/random_smoke_accept';I=B/'e1_independent_review'

def save(path,obj):
 with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 gate.supervised(D/'monitor_cpu001',D/'cpu_check.py')
 old=gate.read(RS/'execution_plan009.json');sources=dict(old['source_sha256'])
 for path,digest in sources.items():
  if gate.sha(path)!=digest:raise ValueError('previously reviewed source changed '+path)
 additions=[D/x for x in ('core.py','references.py','worker.py','gate.py','launch.py','snapshot.py','verify.py','verify_contract.py','test_core.py','test_verify_contract.py','test_gate.py','cpu_check.py','freeze.py','monitor_cpu001/started.json','monitor_cpu001/terminal.json','monitor_cpu001/stdout.log','monitor_cpu001/stderr.log')]
 additions += [B/'e6/random_full/inventory.py',RS/'execution_plan009.json',RS/'manifest_validation001.json',RS/'monitor_validate001/started.json',RS/'monitor_validate001/terminal.json',A/'verification001.json',A/'monitor_verify001/started.json',A/'monitor_verify001/terminal.json',I/'random_smoke_run001_review.json',I/'review_random_smoke_run001.py',I/'monitor_random_smoke_run001/started.json',I/'monitor_random_smoke_run001/terminal.json']
 for p in additions:sources[str(p)]=gate.sha(p)
 p={'stage':'random_full_dev','questions':384,'generations':1152,'seeds':[20261104,20261105,20261106],
    'source_sha256':sources,'budget_source_sha256':old['budget_source_sha256'],
    'preflight_inner_seconds':1800,'preflight_outer_seconds':2100,'run_inner_seconds':21600,'run_outer_seconds':21900,
    'stale_seconds':1800,'grace_seconds':90,'acceptance_reserve_seconds':3600,
    'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'GPU_authorized':False,
    'scope':'prelaunch plan only; accepted E1 DEV mixed journal parsed solely for identity; random selection includes author greedy and is not a performance baseline'}
 save(D/'runtime_plan001.json',p)
 save(D/'verification_plan001.json',{'stage':'no-GT complete panel acceptance, frozen before generation',
      'runtime_plan_sha256':gate.sha(D/'runtime_plan001.json'),'verifier_sha256':gate.sha(D/'verify.py'),
      'contract_sha256':gate.sha(D/'verify_contract.py'),'snapshot_sha256':gate.sha(D/'snapshot.py'),
      'report':str(D/'verification001.json'),'questions':384,'generations':1152,'GT_opened':False,'scoring_performed':False,
      'scope':'all outputs must complete; summaries do not prove reconstruction of tensor bytes; postrun source snapshot cannot exclude transient replacement'})

if __name__=='__main__':main()
