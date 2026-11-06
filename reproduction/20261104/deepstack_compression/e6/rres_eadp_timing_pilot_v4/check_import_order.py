"""Actual accepted E4 import then the real paired runner and its persist API."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import reference_inputs
refs=reference_inputs.load()
import runner as e4_runner
import paired_runner,qwen_request,process_core
if Path(e4_runner.__file__).resolve()!=(ROOT/'e4_c_latency/runner.py').resolve() or Path(paired_runner.__file__).resolve()!=(HERE/'paired_runner.py').resolve() or qwen_request.runner is not paired_runner or process_core.runner is not paired_runner or len(refs['rows'])!=62 or len(refs['references'])!=124:raise ValueError('actual import order')
class IDs:shape=(1,1)
saved=[]
def invoke(b):
 b(IDs(),None);b.generation_returned('raw');return 'response'
response,timing=paired_runner.measure(invoke,lambda:None,persist=saved.append)
if response!='response' or saved!=['raw'] or not timing['raw_persisted_after_all_endpoints']:raise ValueError('actual runner call')
print('passed actual E4 and paired runner path/object/persist call; 62 inputs/124 same-arm references; no GT/GPU')
