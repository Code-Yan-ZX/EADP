"""Bounded, low-frequency host process snapshot; no GPU queries."""
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
start = time.monotonic()
with (D/'process_samples001.jsonl').open('x') as out:
    while time.monotonic()-start < 14400 and not (D/'stop001').exists():
        t0 = time.monotonic_ns()
        try:
            p = subprocess.run(['ps','-eo','user,pid,pcpu,pmem,stat,etime,args','--sort=-pcpu'],
                               text=True,capture_output=True,timeout=10)
            record = {'query_start_ns':t0,'query_end_ns':time.monotonic_ns(),
                      'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        except Exception as e:
            record = {'query_start_ns':t0,'query_end_ns':time.monotonic_ns(),
                      'error_type':type(e).__name__,'error':str(e)}
        out.write(json.dumps(record)+'\n'); out.flush(); os.fsync(out.fileno())
        for _ in range(15):
            if (D/'stop001').exists(): break
            time.sleep(1)
