"""CPU synthetic formal-verifier checks; no GPU, labels, or predictions."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {str(p.resolve()):sha(p) for p in sorted(D.glob('*.py'))}
def main():
 before=sources();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');tests=['test_formal_locks','test_pipeline','test_docker_contract','test_completion_contract','test_verify_candidate','test_window_overlap'];r=subprocess.run([sys.executable,'-m','unittest','-v']+tests,cwd=D,env=env,capture_output=True,text=True,timeout=60);after=sources();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 value={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'GPU_used':False,'predictions_opened':False,'scope':'formal acceptance synthetic CPU tests only'}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if value['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
