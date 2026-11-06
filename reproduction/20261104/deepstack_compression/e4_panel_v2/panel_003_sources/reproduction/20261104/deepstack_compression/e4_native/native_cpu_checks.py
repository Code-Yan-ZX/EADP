"""Isolated synthetic stock-native CPU tests, source hashes before/after."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from native_contract import HERE,E4
from native_gate import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((E4/'cpu_check_023/verification.json').read_text())
    paths=[Path(x) for x in previous['source_sha256_before']]
    paths += [HERE/x for x in ('native_worker.py','native_launch.py','native_gate.py','native_contract.py',
                               'native_capture.py','native_verify.py','native_cpu_checks.py','test_native.py')]
    before={str(p):sha(p) for p in paths}
    if any(before[p]!=digest for p,digest in previous['source_sha256_after'].items()):raise ValueError('reviewed E4 source changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-m','unittest','-v','test_native'];start=time.time()
    try:
        proc=subprocess.run(cmd,cwd=HERE,env=env,capture_output=True,text=True,timeout=180)
        code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr;timeout=False
    except subprocess.TimeoutExpired as e:
        code=None;timeout=True
        stdout=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        stderr=e.stderr.decode(errors='replace') if isinstance(e.stderr,bytes) else (e.stderr or '')
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
    after={str(p):sha(p) for p in paths}
    passed=code==0 and not timeout and before==after and 'Ran 5 tests' in stderr and stderr.rstrip().endswith('OK')
    proof={'status':'passed' if passed else 'failed','returncode':code,'timeout':timeout,'started_unix':start,'finished_unix':time.time(),
           'command':cmd,'cuda_visible_devices':'','source_sha256_before':before,'source_sha256_after':after,
           'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
           'scope':'synthetic CPU only; no real model, GPU, predictions, GT or scores'}
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'status':proof['status'],'returncode':code}))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
