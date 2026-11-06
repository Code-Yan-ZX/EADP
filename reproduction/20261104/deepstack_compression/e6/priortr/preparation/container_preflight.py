"""CPU-only PriorTR container preflight, no model weights or CUDA initialization."""
import json,sys
from pathlib import Path
import torch,transformers,visual_token_pruning
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen
from smoke_worker import admit_review,INSTALL
from admission import admit
proof=admit_review(Path(__file__).parent/'independent_review004.json');e1,rows=admit()
if (Path(sys.prefix).resolve()!=(INSTALL/'env').resolve() or torch.__version__!='2.10.0+cu128' or transformers.__version__!='5.2.0.dev0' or
 Path(visual_token_pruning.__file__).resolve()!=(INSTALL/'source/visual_token_pruning/__init__.py').resolve() or
 Path(qwen.__file__).resolve()!=(INSTALL/'source/qwen3/modeling_qwen3_vl.py').resolve() or torch.cuda.is_initialized()):raise ValueError('isolated runtime or no-GPU precondition failed')
print(json.dumps({'status':'passed','scope':'CPU-only container import and reviewed admission; no GPU, weights or generation','torch':torch.__version__,'transformers':transformers.__version__,'priortr':visual_token_pruning.__file__,'qwen':qwen.__file__,'cuda_initialized':torch.cuda.is_initialized(),'n_frozen_samples':len(rows),'review':proof}))
