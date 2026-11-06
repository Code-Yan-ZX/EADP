"""Bounded, source-bound synthetic tests; never launches a GPU or reads GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4';NATIVE=HERE.parent/'e4_native'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=[HERE/x for x in ('README.md','b_gate.py','b_imports.py','b_worker.py','b_launch.py','b_verify.py','b_preflight.py','b_import_regression.py','budget_plan.json','arm.py','off_capture.py','pair_contract.py',
           'verify_records.py','cpu_checks.py','test_arm.py','test_b_launch.py','test_b_verify.py','test_b_gate.py','test_off_capture.py','test_pair_contract.py','test_verify_records.py')]
    paths += [E4/x for x in ('off_adapter.py','skip_deepstack.py','identity_records.py','decode_observer.py')]
    paths += [NATIVE/x for x in ('native_contract.py','native_capture.py','native_gate.py')]
    previous=json.loads((HERE.parent/'e4_panel_v2/cpu_check_005/verification.json').read_text())
    paths += [Path(x) for x in previous['source_sha256_after']]
    paths += [HERE.parent/'e4_analysis'/x for x in ('score_e4.py','paired.py','plan.json')]
    paths=list(dict.fromkeys(paths))
    before={str(x):sha(x) for x in paths};command=[sys.executable,'-m','unittest','-v','test_arm','test_b_launch','test_b_verify','test_b_gate','test_off_capture','test_pair_contract','test_verify_records']
    if any(before[p]!=v for p,v in previous['source_sha256_after'].items()):raise ValueError('reviewed upstream source changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');started=time.time()
    try:
        proc=subprocess.run(command,cwd=HERE,env=env,text=True,capture_output=True,timeout=180)
        code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr;timeout=False
    except subprocess.TimeoutExpired as e:
        code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timeout=True
    (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr);after={str(x):sha(x) for x in paths}
    passed=code==0 and not timeout and before==after and 'Ran 16 tests' in stderr and stderr.rstrip().endswith('OK')
    proof={'status':'passed' if passed else 'failed','returncode':code,'timeout':timeout,'command':command,
           'cuda_visible_devices':'','source_sha256_before':before,'source_sha256_after':after,
           'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),'started_unix':started,'finished_unix':time.time(),
           'scope':'synthetic CPU only; no model, GPU, real B predictions, labels, or performance measurement'}
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'status':proof['status']}))
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
