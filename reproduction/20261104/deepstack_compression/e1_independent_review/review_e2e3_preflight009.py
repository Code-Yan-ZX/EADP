"""Independent CPU/source review for E2/E3 preflight only; no predictions or GT."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent.parent
SMOKE = BASE / 'e2_e3_smoke'
PROOF = SMOKE / 'cpu_check_009/verification.json'
OUT = SMOKE / 'independent_review_009_preflight.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cpu = json.loads(PROOF.read_text())
    terminal = json.loads((SMOKE / 'monitor_cpu009/terminal.json').read_text())
    if cpu['status'] != 'passed' or cpu['returncode'] != 0 or terminal['reason'] != 'completed' or terminal['worker_returncode'] != 0 or terminal['remaining_process_group']:
        raise ValueError('CPU proof or supervised test did not complete')
    before = cpu['source_sha256_before']
    if before != cpu['source_sha256_after'] or any(sha(p) != h for p, h in before.items()):
        raise ValueError('CPU-tested source changed')
    runs = []
    for folder in (BASE / 'e2_e3_contract', SMOKE):
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(folder), '-p', 'test_*.py', '-v']
        p = subprocess.run(command, capture_output=True, text=True, timeout=180)
        runs.append({'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
        if p.returncode != 0:
            raise ValueError('independent unit tests failed: ' + p.stderr)
    if any(sha(p) != h for p, h in before.items()):
        raise ValueError('source changed during independent tests')
    proof = {'status':'passed', 'independent_review':True, 'scope':'CPU contract and source review; only allow non-GPU container preflight; no model run or predictions',
             'cpu_proof_sha256':sha(PROOF), 'source_sha256':before, 'cpu_terminal_sha256':sha(SMOKE/'monitor_cpu009/terminal.json'),
             'reviewer_sha256':sha(__file__), 'tests':runs, 'finished_unix':time.time()}
    with OUT.open('x') as f:
        json.dump(proof, f, indent=2);f.write('\n')

if __name__ == '__main__':
    main()
