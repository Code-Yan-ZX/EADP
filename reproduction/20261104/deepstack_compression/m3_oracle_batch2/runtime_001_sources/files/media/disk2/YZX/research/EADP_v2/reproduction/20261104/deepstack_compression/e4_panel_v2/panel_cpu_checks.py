"""Bounded CPU-only tests and source binding for the full DEV panel snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4';NATIVE=HERE.parent/'e4_native'
sys.path.insert(1,str(E4))
from identity_worker import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((NATIVE/'cpu_check_002/verification.json').read_text())
    paths=[Path(x) for x in previous['source_sha256_after']]
    paths += [HERE/x for x in ('panel_worker.py','panel_launch.py','panel_gate.py','panel_verify.py',
                               'native_panel_gate.py','panel_v2_imports.py','import_regression.py','test_panel.py','panel_cpu_checks.py','budget_plan.json')]
    paths += [HERE.parent/'initial_plan.json']
    before={str(p):sha(p) for p in paths}
    if any(before[p]!=digest for p,digest in previous['source_sha256_after'].items()):raise ValueError('reviewed upstream source changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','unittest','-v','test_panel'];start=time.time()
    try:
        proc=subprocess.run(command,cwd=HERE,env=env,capture_output=True,text=True,timeout=180)
        code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr;timeout=False
    except subprocess.TimeoutExpired as e:
        code=None;timeout=True;stdout=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        stderr=e.stderr.decode(errors='replace') if isinstance(e.stderr,bytes) else (e.stderr or '')
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
    after={str(p):sha(p) for p in paths}
    passed=code==0 and not timeout and before==after and 'Ran 10 tests' in stderr and stderr.rstrip().endswith('OK')
    proof={'status':'passed' if passed else 'failed','returncode':code,'timeout':timeout,'started_unix':start,'finished_unix':time.time(),
        'command':command,'cuda_visible_devices':'','source_sha256_before':before,'source_sha256_after':after,
        'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
        'scope':'synthetic CPU only; no real model, GPU, predictions, GT or scoring'}
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'status':proof['status'],'returncode':code}))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
