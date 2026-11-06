"""Freeze no-GPU preflight candidate using only source hashes and fake tests."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 planfile=D/'cpu_plan001.json';proofpath=D/'cpu_proof001.json'
 if planfile.exists() or proofpath.exists():raise ValueError('frozen outputs exist')
 refpath=B/'e4_c_multibudget_references192/execution_plan001.json';ref=json.loads(refpath.read_text());sources=dict(ref['source_sha256'])
 local=('PLAN.md','core.py','gate.py','gpu_gate.py','cpu_gate.py','worker.py','launch.py','fixed_reference_gate.py','test_core.py','test_lock.py','test_cpu_preflight.py','test_gpu_gate.py','freeze_cpu.py')
 extra=[refpath,B/'initial_plan.json',B/'e4_c_multibudget_expansion192/proposal002.json',B/'e4_c_multibudget_worker_v3/runner.py',B/'e1_independent_review/e4c_multibudget_references192_worker_preflight_review.json',B/'e1_independent_review/review_e4c_multibudget_references192_worker_preflight.py',B/'e1_independent_review/monitor_e4c_multibudget_references192_worker_preflight/started.json',B/'e1_independent_review/monitor_e4c_multibudget_references192_worker_preflight/terminal.json']
 for p in [*(D/x for x in local),*extra]:sources[str(p.resolve())]=sha(p)
 if any('/random_full_exec' in p or '/references192/run_001/' in p or '/score_panel' in p or p.endswith(('.tsv','.xlsx')) for p in sources):raise ValueError('unexpected output/truth source')
 if any(sha(p)!=h for p,h in sources.items()):raise ValueError('frozen source changed before test')
 cmd=[sys.executable,'-m','unittest','-v','test_core','test_lock','test_cpu_preflight','test_gpu_gate'];r=subprocess.run(cmd,cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
 if any(sha(p)!=h for p,h in sources.items()):raise ValueError('source changed during test')
 if r.returncode:raise RuntimeError('CPU tests failed '+r.stderr[-2000:])
 plan={'status':'prepared_cpu_preflight','inputs':222,'jobs':7920,'source_sha256':sources,'budget_source_sha256':ref['budget_source_sha256'],'minimum_free_bytes':ref['minimum_free_bytes'],'GT_opened':False,'target_predictions_opened':False,'predictions_opened':True,'prediction_scope':'existing DEV mixed prediction sources only opened as bytes for source SHA during preparation','GPU_authorized':False,'scope':'only independently reviewed no-GPU worker preflight may be authorized'}
 with planfile.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','test_returncode':r.returncode,'test_output':r.stdout+r.stderr,'plan_sha256':sha(planfile),'source_sha256':sources,'GT_opened':False,'target_predictions_opened':False,'predictions_opened':True,'prediction_scope':'existing DEV mixed predictions only hashed for source identity','GPU_authorized':False,'model_loaded':False}
 with proofpath.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
