"""CPU-only import inside frozen EADP container, no weights/image or GPU."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
plan=json.loads((BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())
sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
import torch,transformers
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
from light_adapter import LightAdapter
from timer import FirstTokenBoundary,measure
from identity_gate import require_gpu_identity
assert not torch.cuda.is_initialized()
try:require_gpu_identity()
except ValueError as e:
 if 'locked' not in str(e):raise
else:raise ValueError('real GPU path unexpectedly unlocked')
print(json.dumps({'status':'passed','torch':torch.__version__,'transformers':transformers.__version__,'cuda_initialized':torch.cuda.is_initialized(),'python':sys.executable,'author_module':sys.modules[Qwen3VLChatEADP.__module__].__file__,'scope':'CPU imports only; no trained model, generation or GPU'}))
