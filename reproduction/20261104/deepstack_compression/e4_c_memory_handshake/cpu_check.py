"""Run rehearsal contract tests with before/after source identity."""
import json
import os
import subprocess
import sys
from pathlib import Path
import gate


def main():
    out = gate.HERE/'cpu_check005'
    out.mkdir(exist_ok=False)
    before = gate.sources()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(gate.HERE))
    p = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_handshake'],
                       cwd=gate.BASE, env=env, capture_output=True, text=True, timeout=60)
    (out/'tests.log').write_text(p.stdout+p.stderr)
    after = gate.sources()
    result = {'status':'passed' if p.returncode == 0 and before == after else 'failed',
              'source_sha256_before':before, 'source_sha256_after':after,
              'gpu_used':False, 'model_loaded':False, 'test_exit_code':p.returncode}
    with (out/'verification.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
