"""Freeze tests without CUDA; no new predictions or labels."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from source_inventory import hashes
D=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();before=hashes();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],env=env,text=True,capture_output=True,timeout=240);after=hashes();ok=r.returncode==0 and before==after
 a.out.mkdir(exist_ok=False)
 with (a.out/'verification.json').open('x') as f:json.dump({'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests_returncode':r.returncode,'tests_stdout':r.stdout,'tests_stderr':r.stderr,'GT_opened':False,'gpu_used':False,'model_loaded':False,'existing_DEV_predictions_opened':True,'scope':'synthetic CPU tests; source hashing opens prior DEV prediction evidence'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
