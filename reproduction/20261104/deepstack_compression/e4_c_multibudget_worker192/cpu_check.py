"""Bounded synthetic-only tests and source snapshot for the locked candidate."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'cpu_proof001.json'
 if out.exists():raise ValueError('proof exists')
 paths=[D/x for x in ('PLAN.md','core.py','gate.py','worker.py','fixed_reference_gate.py','test_core.py','test_lock.py','cpu_check.py')]+[B/x for x in ('e4_c_multibudget_worker_v3/runner.py','e4_c_multibudget_expansion192/SELECTION_PLAN.md','e4_c_multibudget_expansion192/SCHEDULE_CONTRACT.md','e4_c_multibudget_expansion192/proposal002.json','e4_c_multibudget_references192/references.py','e4_c_budget_ttft_pilot_v3/generation_identity.py','e4_c_budget_pilot_v2/adapter.py','e6/pixel_natural_timing_pilot_v4/bootstrap.py')]
 before={str(p.resolve()):sha(p) for p in paths};r=subprocess.run([sys.executable,'-m','unittest','-v','test_core','test_lock'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120);after={str(p.resolve()):sha(p) for p in paths}
 proof={'status':'passed' if r.returncode==0 and before==after else 'failed','test_returncode':r.returncode,'test_output':r.stdout+r.stderr,'source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'target_predictions_opened':False,'model_loaded':False,'GPU_authorized':False,'scope':'synthetic locked worker candidate only; not a CPU preflight or GPU authorization'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if proof['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
