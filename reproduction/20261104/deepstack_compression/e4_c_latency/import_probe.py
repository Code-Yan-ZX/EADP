"""Real author-container imports only; no models, inputs, predictions or GPU."""
import importlib.metadata,inspect,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
from real_worker import PLAN,sha
plan=json.loads(PLAN.read_text());sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
import torch
from model import pruner
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
import latency_worker,latency_gate,runner,stage_runner,light_adapter
versions={n:importlib.metadata.version(n) for n in plan['expected_versions']};expected=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=(Path(plan['author_root'])/'model/pruner.py').resolve() or
    Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=expected or Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected or
    Path(runner.__file__).resolve()!=(HERE/'runner.py').resolve() or Path(light_adapter.__file__).resolve()!=(HERE/'light_adapter.py').resolve() or
    sha(HERE/'light_adapter.py')!=sha(BASE/'e4_c_timing/light_adapter.py') or torch.cuda.is_initialized()):raise ValueError('latency CPU import identity failed')
print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'predictions':False,'versions':versions,
                  'runner':runner.__file__,'light_adapter':light_adapter.__file__,'scope':'container CPU imports only, no GPU or trained model'}),flush=True)
