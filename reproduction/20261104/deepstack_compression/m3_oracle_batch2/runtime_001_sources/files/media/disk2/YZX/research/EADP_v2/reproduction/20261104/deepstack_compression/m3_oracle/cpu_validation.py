"""Run source-bound CPU contracts without ground truth or model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCES = [HERE/x for x in ('freeze_sample.py','proposals.py','saliency.py','teacher_contract.py','oracle_teacher.py',
                             'test_proposals.py','test_saliency.py','test_teacher_contract.py','cpu_validation.py')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    before = {str(p): sha(p) for p in SOURCES}
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v']
    run = subprocess.run(command, capture_output=True, text=True, timeout=180)
    after = {str(p): sha(p) for p in SOURCES}
    proof = {'status': 'passed' if run.returncode == 0 and before == after else 'failed',
             'returncode': run.returncode, 'source_sha256_before': before, 'source_sha256_after': after,
             'command': command, 'stdout': run.stdout, 'stderr': run.stderr, 'finished_unix': time.time(),
             'scope': 'CPU synthetic/tiny random Qwen3-VL text model; no reference answers, real weights or GPU'}
    out = HERE/'cpu_validation001.json'
    with out.open('x') as f:
        json.dump(proof, f, indent=2);f.write('\n')
    return 0 if proof['status'] == 'passed' else 1

if __name__ == '__main__':
    sys.exit(main())
