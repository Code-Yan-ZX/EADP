"""Freeze CPU-only diagnostic code; never reads image features or scores."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 proof=D/'cpu_proof001.json';plan=D/'code_plan001.json'
 if proof.exists() or plan.exists():raise ValueError('already frozen')
 paths=[D/x for x in ('PLAN.md','plan_freeze001.json','freeze_plan.py','features.py','gate.py','extract.py','test_features.py','test_entry.py','freeze_code.py')];before={str(p.resolve()):sha(p) for p in paths}
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_features','test_entry'],cwd=D,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));after={str(p.resolve()):sha(p) for p in paths}
 if before!=after or result.returncode:raise ValueError('synthetic test/source failure '+result.stdout)
 data={'status':'passed','returncode':result.returncode,'source_before':before,'source_after':after,'tests':result.stdout,'GT_opened':False,'predictions_opened':False,'feature_or_score_values_opened':False,'GPU_used':False}
 with proof.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result={'status':'frozen','plan_freeze_sha256':sha(D/'plan_freeze001.json'),'source_sha256':dict(after,**{str(proof.resolve()):sha(proof)}),'GT_opened':False,'predictions_opened':False,'feature_or_score_values_opened':False,'GPU_authorized':False,'scope':'synthetic CPU preparation only; no extraction or score access'}
 with plan.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
