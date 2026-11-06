"""Freeze CPU-only acceptance code before random model GPU generation."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;B=A.parents[1];D=B/'e6/random_smoke'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 out=A/'verification_plan001.json'
 if out.exists():raise ValueError('acceptance plan already frozen')
 paths=[A/x for x in ('verify.py','contract.py','inventory_contract.py','test_verify.py','test_contract.py','test_inventory_contract.py','freeze.py')];before={str(p.resolve()):sha(p) for p in paths}
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_verify','test_contract','test_inventory_contract'],cwd=A,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=60)
 after={str(p.resolve()):sha(p) for p in paths}
 if result.returncode!=0 or before!=after:raise ValueError('acceptance tests failed or source changed\n'+result.stdout+result.stderr)
 plan={'status':'prepared','execution_plan_sha256':sha(D/'execution_plan009.json'),'source_sha256':before,'test_output':result.stdout+result.stderr,'scope':'future completed random 3-anchor/18-arm no-GT acceptance only; GPU still locked pending independent reviews','GT_opened':False,'scoring_performed':False,'GPU_authorized':False}
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('random acceptance source/test plan frozen; no GPU authorization')
if __name__=='__main__':main()
