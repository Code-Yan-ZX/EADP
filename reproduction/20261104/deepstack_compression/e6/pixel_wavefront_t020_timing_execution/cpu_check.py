"""Freeze candidate and CPU preflight-source tests, no GPU or model load."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import gate
D=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=D/'cpu_check001':raise ValueError('unplanned CPU output')
 before=gate.sources();r=subprocess.run([sys.executable,str(D/'test_preflight.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60);after=gate.sources();ok=r.returncode==0 and before==after;a.out.mkdir(exist_ok=False)
 with (a.out/'verification.json').open('x') as f:json.dump({'status':'passed' if ok else 'failed','returncode':r.returncode,'test_stdout':r.stdout,'test_stderr':r.stderr,'source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'existing_DEV_predictions_opened':True,'model_loaded':False,'gpu_used':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
