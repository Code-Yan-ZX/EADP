"""Bounded synthetic CPU tests with full source identity before and after."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from latency_worker import sources
from real_worker import HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid CPU check ID')
 out=HERE/('cpu_check'+args.run_id);out.mkdir(exist_ok=False);before={str(p):sha(p) for p in sources()};start=time.time()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
 result=subprocess.run([sys.executable,'-m','unittest','-v','test_latency','test_lifecycle'],cwd=HERE,env=env,capture_output=True,text=True,timeout=120)
 (out/'stdout.log').write_text(result.stdout);(out/'stderr.log').write_text(result.stderr);after={str(p):sha(p) for p in sources()}
 ok=result.returncode==0 and before==after and 'Ran 14 tests' in result.stderr and result.stderr.rstrip().endswith('OK')
 proof={'status':'passed' if ok else 'failed','returncode':result.returncode,'source_sha256_before':before,'source_sha256_after':after,
        'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'started_unix':start,'finished_unix':time.time(),
        'scope':'synthetic schedule, first-token boundary, failure and monitor overlap only; no GPU, GT or real performance'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
