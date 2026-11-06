"""CPU-only record of actual generation implementation used for TTFT tests."""
import hashlib,json,sys
from pathlib import Path
import torch,transformers
from transformers.generation import utils,stopping_criteria

def identity(module):
 path=Path(module.__file__).resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
result={'status':'passed','torch':torch.__version__,'transformers':transformers.__version__,'python':sys.executable,
 'generation_utils':identity(utils),'stopping_criteria':identity(stopping_criteria),'cuda_initialized':torch.cuda.is_initialized(),
 'scope':'CPU source provenance; no weights, GPU initialization or generation'}
if result['cuda_initialized']:raise ValueError('unexpected CUDA initialization')
print(json.dumps(result,sort_keys=True))
