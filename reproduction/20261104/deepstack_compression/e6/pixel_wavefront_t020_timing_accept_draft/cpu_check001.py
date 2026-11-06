"""Freeze CPU tests for the acceptance draft only, not formal admission."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
N=('verify_candidate.py','window_overlap.py','test_verify_candidate.py','test_window_overlap.py','cpu_check001.py')
def sources():return {str((D/n).resolve()):hashlib.sha256((D/n).read_bytes()).hexdigest() for n in N}
def main():
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_verify_candidate','test_window_overlap'],cwd=D,env=env,capture_output=True,text=True,timeout=60);after=sources();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 value={'status':'passed' if r.returncode==0 and before==after else 'failed','scope':'formal acceptance design CPU draft, no experiment proof','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'GPU_authorized':False,'GT_opened':False,'predictions_opened':False}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
