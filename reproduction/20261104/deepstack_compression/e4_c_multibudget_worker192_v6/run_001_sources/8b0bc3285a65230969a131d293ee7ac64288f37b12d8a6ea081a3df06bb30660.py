"""CPU test and source freeze for the fixed32 diagnostic launcher; no GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 plan_path=HERE/'execution_plan005.json';proof_path=HERE/'execution_cpu005.json'
 if plan_path.exists() or proof_path.exists():raise ValueError('frozen output exists')
 prior=json.loads((BASE/'e4_c_budget_execution/execution_plan001.json').read_text());sources=dict(prior['source_sha256'])
 local=('fresh_fixed.py','light_fixed.py','fixed_common.py','compare_fixed.py','core.py','test_fixed.py','test_real_arm_schema.py','test_core.py','cpu_check.py','cpu002.json','monitor_cpu002/started.json','monitor_cpu002/terminal.json','gate.py','launch.py','worker.py','test_execution.py','freeze_execution.py')
 for name in local:
  p=HERE/name;sources[str(p.resolve())]=sha(p)
 for p in (BASE/'e4_c_budget_execution/execution_plan001.json',BASE/'e4_c_budget_ttft_pilot_v3/identity_gate.py',BASE/'e1_independent_review/e4c_budget_identity001_review.json',BASE/'e1_independent_review/monitor_e4c_budget_identity001/started.json',BASE/'e1_independent_review/monitor_e4c_budget_identity001/terminal.json'):
  sources[str(p.resolve())]=sha(p)
 before=dict(sources)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='');cmd=[sys.executable,'-m','unittest','-v','test_fixed','test_real_arm_schema','test_core','test_execution']
 p=subprocess.run(cmd,cwd=HERE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 after={name:sha(name) for name in before}
 if before!=after:raise ValueError('source changed during tests')
 if p.returncode:
  with proof_path.open('x') as f:json.dump({'status':'failed','test_returncode':p.returncode,'test_output':p.stdout},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  raise SystemExit(p.returncode)
 plan={'stage':'4-budget fixed32 24-generation diagnostic','budgets':[128,256,512,1024],'source_sha256':before,'budget_source_sha256':prior['budget_source_sha256'],
       'inner_timeout_seconds':3600,'outer_timeout_seconds':3720,'stale_seconds':1200,'grace_seconds':90,'acceptance_reserve_seconds':3600,'minimum_free_bytes':prior['minimum_free_bytes'],
       'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'GPU_authorized_by_plan':False,
       'scope':'fresh versus lightweight fixed32 on three existing DEV anchors; 24 arms, 12 pairs, no performance claims; all-layer KV fingerprints, not archived KV arrays'}
 with plan_path.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','test_returncode':p.returncode,'test_output':p.stdout,'source_sha256':before,'plan_sha256':sha(plan_path),'GT_opened':False,'predictions_opened':True,'prediction_scope':'source hash only for previously accepted DEV prediction artifacts','scoring_performed':False,'model_loaded':False,'GPU_authorized':False}
 with proof_path.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('fixed32 execution CPU freeze passed; sources',len(before),flush=True)
if __name__=='__main__':main()
