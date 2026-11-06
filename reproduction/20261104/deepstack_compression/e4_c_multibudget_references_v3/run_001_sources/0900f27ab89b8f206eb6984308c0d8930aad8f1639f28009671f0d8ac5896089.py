"""CPU-only proof and source freeze for planned 248 fixed32 references."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 plan_path=HERE/'execution_plan001.json';proof_path=HERE/'execution_cpu001.json'
 if plan_path.exists() or proof_path.exists():raise ValueError('frozen output exists')
 prior=json.loads((BASE/'e4_c_multibudget_fixed32/execution_plan005.json').read_text());timing=json.loads((BASE/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json').read_text());sources=dict(prior['source_sha256'])
 for name,value in timing['source_sha256'].items():
  if name in sources and sources[name]!=value:raise ValueError('conflicting previously frozen source '+name)
  sources[name]=value
 local=['PLAN.md','reference_arm.py','reference_cache.py','core.py','gate.py','worker.py','launch.py','test_core.py','test_execution.py','freeze_execution.py']
 extra=[BASE/'e4_c_multibudget_fixed32/execution_plan005.json',BASE/'e4_c_multibudget_fixed32/execution_cpu005.json',BASE/'e4_c_multibudget_fixed32/monitor_execution_cpu005/started.json',BASE/'e4_c_multibudget_fixed32/monitor_execution_cpu005/terminal.json',BASE/'e4_c_multibudget_worker/preflight.py',BASE/'e4_c_multibudget_worker/references.py',BASE/'e4_c_multibudget_worker/worker_core.py',BASE/'e4_c_multibudget_worker/runner.py',BASE/'e4_c_multibudget_expansion/pilot_plan001.json',BASE/'e4_c_multibudget_expansion/pilot_contract.py']
 extra.extend(BASE/x for x in ('e4_c_multibudget_fixed32_accept/verification001.json','e4_c_multibudget_fixed32_accept/monitor_verify001/started.json','e4_c_multibudget_fixed32_accept/monitor_verify001/terminal.json','e4_c_multibudget_fixed32/run_001_sources/manifest.json','e4_c_multibudget_fixed32/monitor_run001/started.json','e4_c_multibudget_fixed32/monitor_run001/terminal.json','e1_independent_review/e4c_multibudget_fixed32_result_review.json','e1_independent_review/review_e4c_multibudget_fixed32_result.py','e1_independent_review/monitor_e4c_multibudget_fixed32_result_002/started.json','e1_independent_review/monitor_e4c_multibudget_fixed32_result_002/terminal.json'))
 for p in [*(HERE/x for x in local),*extra]:sources[str(p.resolve())]=sha(p)
 for path,digest in sources.items():
  if sha(path)!=digest:raise ValueError('frozen source already changed '+path)
 before=dict(sources);env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 cmd=[sys.executable,'-m','unittest','-v','test_core','test_execution'];test=subprocess.run(cmd,cwd=HERE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 if {name:sha(name) for name in before}!=before:raise ValueError('source changed during CPU tests')
 if test.returncode:
  with proof_path.open('x') as f:json.dump({'status':'failed','test_returncode':test.returncode,'test_output':test.stdout},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  raise SystemExit(test.returncode)
 plan={'stage':'62-input four-budget fixed32 reference generation','jobs':248,'source_sha256':before,'pilot_plan_sha256':sha(BASE/'e4_c_multibudget_expansion/pilot_plan001.json'),
       'budget_source_sha256':prior['budget_source_sha256'],'inner_timeout_seconds':5400,'outer_timeout_seconds':5520,'stale_seconds':1200,'grace_seconds':90,'acceptance_reserve_seconds':3600,'minimum_free_bytes':prior['minimum_free_bytes'],
       'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'GPU_authorized_by_plan':False,'scope':'248 fresh fixed32 DEV references; no score or performance; GPU hard-locked pending actual 12-pair identity and independent prelaunch'}
 with plan_path.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','test_returncode':test.returncode,'test_output':test.stdout,'source_sha256':before,'plan_sha256':sha(plan_path),'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'model_loaded':False,'GPU_authorized':False}
 with proof_path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('reference execution CPU candidate frozen; sources',len(before),flush=True)
if __name__=='__main__':main()
