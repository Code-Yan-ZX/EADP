"""Source-bound synthetic tests for M3 real-model feasibility entry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from local_imports import local
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sha=local('contract').sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    names=('plan.json','contract.py','core.py','gate.py','worker.py','launcher.py','verify.py','preflight.py','local_imports.py',
           'cpu_checks.py','cpu_tests.py','review_oracle.py','test_contract.py','test_core.py','test_gate.py','test_launcher.py')
    paths=[HERE/x for x in names]
    oracle=json.loads((HERE/'oracle_review_001.json').read_text())
    paths += [Path(x) for x in oracle['source_sha256']]
    previous=json.loads((ROOT/'e4_panel_v2/cpu_check_005/verification.json').read_text())
    paths += [Path(x) for x in previous['source_sha256_after']];paths=list(dict.fromkeys(paths))
    before={str(x):sha(x) for x in paths}
    if any(before[p]!=v for p,v in oracle['source_sha256'].items()) or any(before[p]!=v for p,v in previous['source_sha256_after'].items()):
        raise ValueError('reviewed upstream source changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(HERE/'cpu_tests.py')];begin=time.time()
    try:r=subprocess.run(command,cwd=HERE,env=env,text=True,capture_output=True,timeout=180);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
    except subprocess.TimeoutExpired as e:code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timed=True
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={str(x):sha(x) for x in paths}
    passed=code==0 and not timed and before==after and 'Ran 10 tests' in stderr and stderr.rstrip().endswith('OK')
    result={'status':'passed' if passed else 'failed','returncode':code,'timeout':timed,'command':command,
        'source_sha256_before':before,'source_sha256_after':after,'started_unix':begin,'finished_unix':time.time(),
        'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'cuda_visible_devices':'',
        'scope':'synthetic CPU fake model only; no real model, GPU, predictions, GT or scoring'}
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':result['status']}));return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
