"""Bounded CPU-only synthetic preflight; not real model authorization."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 files=list(HERE.glob('*.py'))+[HERE/'plan.json',BASE/'m3_oracle_exec/selection_adapter.py']+[BASE/'e4'/x for x in ('budget_adapter.py','off_adapter.py','skip_deepstack.py','test_off_adapter.py')]
 before={str(x.resolve()):sha(x) for x in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.time()
 try:r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True,env=env,timeout=120);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timeout=False
 except subprocess.TimeoutExpired as e:code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timeout=True
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={str(x.resolve()):sha(x) for x in files}
 ok=code==0 and not timeout and before==after and 'Ran 6 tests' in stderr and stderr.rstrip().endswith('OK')
 proof={'status':'passed' if ok else 'failed','returncode':code,'timed_out':timeout,'source_sha256_before':before,'source_sha256_after':after,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
        'started_unix':start,'finished_unix':time.time(),'GT_opened':False,'gpu_used':False,'generation_performed':False,'scope':'synthetic CPU only; actual worker container preflight and model smoke not performed'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
