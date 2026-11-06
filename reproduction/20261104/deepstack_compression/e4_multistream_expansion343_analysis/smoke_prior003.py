"""Real prior evidence and complete 343 proof boundary; never open 343 record/norm."""
import json,os,sys
from pathlib import Path
import gate
D=Path(__file__).resolve().parent
class BoundaryReached(Exception):pass
original=gate.sha
def sha(p):
 if Path(p).resolve().is_relative_to(gate.R.resolve()):raise BoundaryReached('343 capture content not opened')
 return original(p)
gate.sha=sha
gate.prelaunch=lambda: None
for _ in range(2):
 try:gate.accepted()
 except BoundaryReached:pass
 else:raise ValueError('expected capture boundary')
 if '_full_multistream_prior_gate' in sys.modules:raise ValueError('prior module left cached')
with (D/'smoke_prior003.json').open('x') as f:json.dump({'status':'passed','real_prior_and_dual_proof_gate_calls':2,'343_capture_opened':False,'GT_opened':False,'scoring_performed':False,'prelaunch_bypassed_only_for_this_bounded_cpu_smoke':True},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
