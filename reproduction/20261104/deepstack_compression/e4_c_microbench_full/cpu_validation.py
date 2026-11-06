"""Source-bound CPU synthetic tests; never starts a model or GPU."""
import argparse,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
from full_admission import sources
from identity_gate import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);files=sources();before={str(f.resolve()):sha(f) for f in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],env=env,capture_output=True,text=True,timeout=120);(a.out/'stdout.log').write_text(r.stdout);(a.out/'stderr.log').write_text(r.stderr);after={str(f.resolve()):sha(f) for f in files}
 result={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':sha(a.out/'stderr.log'),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'model_loaded':False}
 with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
