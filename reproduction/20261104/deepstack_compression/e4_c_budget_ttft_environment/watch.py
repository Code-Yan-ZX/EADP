"""Bounded low-frequency process snapshots; no GPU or application mutation."""
import json
import os
import subprocess
import time
from pathlib import Path

D=Path(__file__).resolve().parent
os.nice(10)
start=time.monotonic()
with (D/'process_samples001.jsonl').open('x') as f:
 while time.monotonic()-start<1800 and not (D/'stop001').exists():
  t0=time.monotonic_ns()
  try:
   p=subprocess.run(['ps','-eo','user,pid,pcpu,pmem,stat,etime,comm','--sort=-pcpu'],capture_output=True,text=True,timeout=10)
   sample={'event':'snapshot','query_start_ns':t0,'query_end_ns':time.monotonic_ns(),
           'time_namespace':os.readlink('/proc/self/ns/time'),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  except Exception as e:
   sample={'event':'snapshot','query_start_ns':t0,'query_end_ns':time.monotonic_ns(),
           'error_type':type(e).__name__,'error':str(e),'time_namespace':os.readlink('/proc/self/ns/time')}
  f.write(json.dumps(sample)+'\n');f.flush();os.fsync(f.fileno())
  # The interval covers the ps query and first log fsync. This second line's
  # own write/fsync is not included in the recorded end and may also perturb timing.
  f.write(json.dumps({'event':'snapshot_logging_complete','query_start_ns':t0,
                      'logging_end_ns':time.monotonic_ns()})+'\n');f.flush();os.fsync(f.fileno())
  for _ in range(60):
   if (D/'stop001').exists():break
   time.sleep(1)
