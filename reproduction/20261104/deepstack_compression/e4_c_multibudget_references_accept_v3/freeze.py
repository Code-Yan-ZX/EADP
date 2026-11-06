"""Freeze 248-reference post-run verifier prior to its GPU run."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dst=HERE/'verification_plan001.json'
 if dst.exists():raise ValueError('verification plan already frozen')
 paths=[HERE/'verify.py',HERE/'snapshot.py',HERE/'test_verify.py',HERE/'freeze.py',BASE/'e4_c_multibudget_references_v3/execution_plan001.json',BASE/'e4_c_multibudget_references_v3/gate.py',BASE/'e4_c_multibudget_references_v3/core.py',BASE/'e4_c_multibudget_fixed32/core.py',BASE/'e4_c_multibudget_worker/references.py']
 before={str(p.resolve()):sha(p) for p in paths};r=subprocess.run([sys.executable,'-m','unittest','-v','test_verify'],cwd=HERE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
 if before!={str(p.resolve()):sha(p) for p in paths}:raise ValueError('verification source changed during tests')
 if r.returncode:raise ValueError('verification tests failed: '+r.stdout)
 result={'status':'prepared','run_id':'001','GPU_authorized':False,'created_unix':time.time(),'source_sha256':before,'tests':r.stdout,
         'scope':'formal 248-reference verification after complete GPU/container/supervisor exit; no GT/scoring; post-run snapshot cannot exclude transient source changes'}
 with dst.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('248 reference verification plan frozen',sha(dst),flush=True)
if __name__=='__main__':main()
