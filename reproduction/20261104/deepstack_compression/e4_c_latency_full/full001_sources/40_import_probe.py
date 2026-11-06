"""Real container import test; no model, predictions, truth or GPU."""
import importlib.metadata,inspect,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PILOT=HERE.parent/'e4_c_latency';sys.path.insert(0,str(PILOT))
from real_worker import PLAN
plan=json.loads(PLAN.read_text());sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
import torch
from model import pruner
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
import full_worker,full_admission,pilot_gate,latency_worker,runner,light_adapter
versions={n:importlib.metadata.version(n) for n in plan['expected_versions']};wrapper=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=(Path(plan['author_root'])/'model/pruner.py').resolve() or Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=wrapper or Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=wrapper or
    Path(full_worker.__file__).resolve()!=(HERE/'full_worker.py').resolve() or Path(full_admission.__file__).resolve()!=(HERE/'full_admission.py').resolve() or Path(latency_worker.__file__).resolve()!=(PILOT/'latency_worker.py').resolve() or Path(runner.__file__).resolve()!=(PILOT/'runner.py').resolve() or Path(light_adapter.__file__).resolve()!=(PILOT/'light_adapter.py').resolve() or torch.cuda.is_initialized()):raise ValueError('full latency CPU import identity failed')
print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'predictions':False,'versions':versions,'scope':'real container imports only'}),flush=True)
