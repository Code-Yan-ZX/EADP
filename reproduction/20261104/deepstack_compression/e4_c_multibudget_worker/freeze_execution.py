"""CPU-only preliminary freeze of 32-image timing worker; GPU remains locked."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 plan_path=HERE/'runtime_plan001.json';proof_path=HERE/'execution_cpu001.json'
 if plan_path.exists() or proof_path.exists():raise ValueError('frozen output exists')
 prior=json.loads((BASE/'e4_c_multibudget_references_v2/execution_plan001.json').read_text());timing=json.loads((BASE/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json').read_text());sources=dict(prior['source_sha256'])
 local=['runner.py','worker_core.py','preflight.py','references.py','fixed_reference_gate.py','admission.py','worker.py','launch.py','test_runner.py','test_worker_core.py','test_execution.py','freeze_execution.py']
 extra=[BASE/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json',BASE/'e4_c_multibudget_expansion/pilot_plan001.json',BASE/'e4_c_multibudget_expansion/pilot_contract.py']
 for p in [*(HERE/x for x in local),*extra]:
  path=str(p.resolve());digest=sha(p)
  if path in sources and sources[path]!=digest:raise ValueError('previously frozen source changed '+path)
  sources[path]=digest
 for path,digest in sources.items():
  if sha(path)!=digest:raise ValueError('source already changed '+path)
 before=dict(sources);env=dict(os.environ,CUDA_VISIBLE_DEVICES='');cmd=[sys.executable,'-m','unittest','-v','test_runner','test_worker_core','test_execution']
 t=subprocess.run(cmd,cwd=HERE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 if {name:sha(name) for name in before}!=before:raise ValueError('source changed during CPU tests')
 if t.returncode:
  with proof_path.open('x') as f:json.dump({'status':'failed','test_returncode':t.returncode,'test_output':t.stdout},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  raise SystemExit(t.returncode)
 plan={'stage':'post-hoc 32-image 4-budget natural/fixed32 timing pilot','jobs':1520,'source_sha256':before,'pilot_plan_sha256':sha(BASE/'e4_c_multibudget_expansion/pilot_plan001.json'),'generation_source_sha256':timing['generation_source_sha256'],'budget_source_sha256':timing['budget_source_sha256'],
       'inner_timeout_seconds':7200,'outer_timeout_seconds':7320,'stale_seconds':1800,'grace_seconds':90,'acceptance_reserve_seconds':3600,'minimum_free_bytes':timing['minimum_free_bytes'],
       'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'GPU_authorized_by_plan':False,
       'scope':'preliminary CPU timing worker freeze only; GPU hard-locked until 12-pair and 248-reference formal/independent proofs and final review'}
 with plan_path.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','test_returncode':t.returncode,'test_output':t.stdout,'source_sha256':before,'plan_sha256':sha(plan_path),'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'model_loaded':False,'GPU_authorized':False}
 with proof_path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('timing worker CPU preliminary freeze; sources',len(before),flush=True)
if __name__=='__main__':main()
