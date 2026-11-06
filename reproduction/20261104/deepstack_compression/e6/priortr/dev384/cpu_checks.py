"""Bounded CPU preparation tests and complete before/after source hashes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import gate,source_inventory
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 files=source_inventory.sources();before={p:gate.sha(p) for p in files};up=gate.upstream();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'];started=time.time()
 try:r=subprocess.run(cmd,cwd=HERE,env=env,text=True,capture_output=True,timeout=180);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
 except subprocess.TimeoutExpired as exc:code=None;stdout=str(exc.stdout or '');stderr=str(exc.stderr or '');timed=True
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={p:gate.sha(p) for p in files}
 passed=code==0 and not timed and before==after and 'Ran 16 tests' in stderr and stderr.rstrip().endswith('OK')
 result={'status':'passed' if passed else 'failed','returncode':code,'timed_out':timed,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':gate.sha(out/'stdout.log'),'stderr_sha256':gate.sha(out/'stderr.log'),'upstream':up,'started_unix':started,'finished_unix':time.time(),'GT_opened':False,'scoring_performed':False,'gpu_used':False}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':result['status']}),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
