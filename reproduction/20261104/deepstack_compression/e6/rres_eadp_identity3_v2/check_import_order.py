"""Exercise the actual accepted reference import followed by the timing loader."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
sys.path.insert(0,str(TIMING))
import reference_inputs
refs=reference_inputs.load()
import runner as e4_runner
if Path(e4_runner.__file__).resolve()!=(ROOT/'e4_c_latency/runner.py').resolve():raise ValueError('actual E4 runner absent')
import timing_imports
request,core,timing_runner=timing_imports.load()
if sys.modules['runner'] is not e4_runner or request.runner is not timing_runner or core.runner is not timing_runner or Path(timing_runner.__file__).resolve()!=(TIMING/'paired_runner.py').resolve() or len(refs['rows'])!=62 or len(refs['references'])!=124:raise ValueError('real import ordering/bindings')
print('passed: accepted references; E4 runner restored; timing runner bound; no GPU or GT')
