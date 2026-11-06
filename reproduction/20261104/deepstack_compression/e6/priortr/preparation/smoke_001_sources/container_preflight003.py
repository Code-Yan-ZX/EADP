"""Revised CPU-only container preflight: no model weights, no GPU generation."""
import getpass,json,os,sys
from pathlib import Path
import torch,transformers,visual_token_pruning
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen
from env_contract import verify_env,INSTALL
from admission import admit,sha
fingerprint=verify_env();proof,rows=admit();here=Path(__file__).resolve().parent
if (Path(sys.prefix).resolve()!=(INSTALL/'env').resolve() or torch.__version__!='2.10.0+cu128' or transformers.__version__!='5.2.0.dev0' or
 Path(visual_token_pruning.__file__).resolve()!=(INSTALL/'source/visual_token_pruning/__init__.py').resolve() or
 Path(qwen.__file__).resolve()!=(INSTALL/'source/qwen3/modeling_qwen3_vl.py').resolve() or torch.cuda.is_initialized() or
 getpass.getuser()!='eadp' or not os.environ.get('TORCHINDUCTOR_CACHE_DIR','').startswith(str(here/'container_preflight003_cache'))):raise ValueError('isolated container runtime mismatch')
print(json.dumps({'status':'passed','scope':'CPU-only container import and E1 admission; no GPU, weights or generation','torch':torch.__version__,'transformers':transformers.__version__,'priortr':visual_token_pruning.__file__,'qwen':qwen.__file__,'cuda_initialized':torch.cuda.is_initialized(),'username':getpass.getuser(),'inductor_cache':os.environ['TORCHINDUCTOR_CACHE_DIR'],'n_frozen_samples':len(rows),'fingerprint_sha256':fingerprint,'e1_proof_sha256':sha(here.parents[2]/'e1/e1full001_completion.json'),'launcher_sha256':sha(here/'smoke_launch.py')}))
