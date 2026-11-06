"""Synthetic-only safety contracts; no model, predictions, truth or GPU."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
 from pilot_admission import sources
 files=sources()
 before={str(f.resolve()):sha(f) for f in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],env=env,capture_output=True,text=True,timeout=90)
 (a.out/'stderr.log').write_text(result.stderr);(a.out/'stdout.log').write_text(result.stdout);after={str(f.resolve()):sha(f) for f in files}
 proof={'status':'passed' if result.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':sha(a.out/'stderr.log'),'gpu_used':False,'GT_opened':False,'predictions_opened':False,'model_loaded':False,'scope':'synthetic safety contracts only; real GPU admission remains locked'}
 with (a.out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if proof['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
