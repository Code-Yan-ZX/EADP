"""Freeze and test random-smoke CPU components, never authorize execution."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
NAMES=('smoke_core.py','test_smoke_core.py','identity_contract.py','test_identity_contract.py','inventory.py','test_inventory.py','diagnostic_run.py','test_diagnostic_run.py','model_core.py','test_model_core.py','e1_refs.py','test_e1_refs.py','core_cpu_validation.py','plan001.json','selection_manifest001.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'core_cpu_proof001.json'
 if out.exists():raise ValueError('existing proof')
 paths=[D/n for n in NAMES]+[B/'e4/budget_runner.py',B/'e4/decode_observer.py',B/'e4_c_multibudget_fixed32/fixed_common.py',B/'e6/random_baselines/adapter.py',B/'e6/random_baselines/selector.py',B/'m3_oracle_exec/selection_adapter.py']
 before={str(p.resolve()):sha(p) for p in paths};env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 names=['test_smoke_core','test_identity_contract','test_inventory','test_diagnostic_run','test_model_core','test_e1_refs'];r=subprocess.run([sys.executable,'-m','unittest','-v',*names],cwd=D,env=env,text=True,capture_output=True,timeout=120)
 after={str(p.resolve()):sha(p) for p in paths}
 proof={'status':'passed' if r.returncode==0 and before==after else 'failed','returncode':r.returncode,'test_output':r.stdout+r.stderr,'source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'predictions_opened':False,'model_loaded':False,'GPU_authorized':False,'scope':'CPU synthetic components only; no worker, container or real model authorization'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if proof['status']!='passed':raise SystemExit(1)
 print('random smoke CPU component tests passed')
if __name__=='__main__':main()
