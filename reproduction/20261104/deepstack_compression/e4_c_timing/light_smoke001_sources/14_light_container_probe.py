"""CPU-only actual container import proof; never initialize CUDA or load data."""
import importlib.metadata,inspect,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'e4'))
from real_worker import PLAN
plan=json.loads(PLAN.read_text())
sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
import torch
from model import pruner
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
import budget_runner,light_arm,light_smoke_core,light_smoke_worker,light_smoke_launch,light_admission,light_verify_records,light_verify_core,light_verify
versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
expected_wrapper=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=(Path(plan['author_root'])/'model/pruner.py').resolve() or
    Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=expected_wrapper or Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected_wrapper or
    Path(budget_runner.__file__).resolve()!=(BASE/'e4/budget_runner.py').resolve() or Path(light_arm.__file__).resolve()!=(HERE/'light_arm.py').resolve() or
    torch.cuda.is_initialized()):raise ValueError('CPU import or version identity failed')
print(json.dumps({'status':'passed','versions':versions,'cuda_initialized':False,'model_loaded':False,'predictions':False,
                  'author_wrapper':str(expected_wrapper),'budget_runner':budget_runner.__file__,'light_arm':light_arm.__file__,
                  'scope':'actual container CPU imports only; no images, weights, model or GPU'}),flush=True)
