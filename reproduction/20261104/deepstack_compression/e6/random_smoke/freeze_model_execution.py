"""CPU-only source freeze and tests; requires later independent reviews."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 proof=D/'model_cpu_proof009.json';plan=D/'execution_plan009.json'
 if proof.exists() or plan.exists():raise ValueError('previous freeze exists')
 files=['smoke_core.py','test_smoke_core.py','identity_contract.py','test_identity_contract.py','inventory.py','test_inventory.py','diagnostic_run.py','test_diagnostic_run.py','model_core.py','test_model_core.py','e1_refs.py','test_e1_refs.py','manifest_load.py','test_manifest_load.py','smoke_gate.py','test_smoke_gate.py','model_worker.py','model_launch.py','test_model_worker_integration.py','test_actual_import_chain.py','freeze_model_execution.py','plan001.json','selection_manifest001.jsonl','runtime_plan003.json']
 paths=[D/x for x in files]+[B/'e4_c_multibudget_fixed32/fixed_common.py',B/'e4/budget_runner.py',B/'e4/decode_observer.py',B/'e6/random_baselines/selector.py',B/'e6/random_baselines/adapter.py',B/'m3_oracle_exec/selection_adapter.py',B/'m3_oracle_exec/e1_admission.py',B/'e1_independent_review/random_smoke_preflight002_result_review.json',B/'e1_independent_review/monitor_random_smoke_preflight002_result/started.json',B/'e1_independent_review/monitor_random_smoke_preflight002_result/terminal.json']
 sources={str(p.resolve()):sha(p) for p in paths};prior=json.loads((D/'runtime_plan003.json').read_text());sources.update(prior['source_sha256']);before=dict(sources)
 names=['test_smoke_core','test_identity_contract','test_inventory','test_diagnostic_run','test_model_core','test_e1_refs','test_manifest_load','test_smoke_gate','test_model_worker_integration','test_actual_import_chain'];r=subprocess.run([sys.executable,'-m','unittest','-v',*names],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120)
 after={p:sha(p) for p in before};status='passed' if r.returncode==0 and before==after else 'failed'
 evidence={'status':status,'test_returncode':r.returncode,'test_output':r.stdout+r.stderr,'source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'predictions_opened':True,'prediction_scope':'existing DEV mixed prediction files opened only for SHA, not parsed','model_loaded':False,'GPU_authorized':False}
 with proof.open('x') as f:json.dump(evidence,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if status!='passed':raise SystemExit(1)
 sources[str(proof.resolve())]=sha(proof)
 result={'scope':'random K256 three-task 18-arm diagnostic','source_sha256':sources,'budget_source_sha256':prior['budget_source_sha256'],'GT_opened':False,'predictions_opened':True,'GPU_authorized':False,'preflight_inner_seconds':1800,'preflight_outer_seconds':2100,'run_inner_seconds':7200,'run_outer_seconds':7500,'stale_seconds':1200,'grace_seconds':90,'limitations':['posthoc after prior DEV results','EADP greedy computed and discarded: not timing comparable','preflight parses existing accepted E1 DEV mixed journal only for identity','forced 33 tokens yields 32 decode steps; KV contents represented by fingerprints']}
 with plan.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('CPU freeze passed, model execution remains locked')
if __name__=='__main__':main()
