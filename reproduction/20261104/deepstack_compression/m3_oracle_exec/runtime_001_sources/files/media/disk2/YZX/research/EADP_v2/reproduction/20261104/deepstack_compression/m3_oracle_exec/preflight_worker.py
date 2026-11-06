"""No-GPU, no-data import/identity preflight inside the production image."""
import importlib.metadata,inspect,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT/'m3_oracle'),str(ROOT/'e4')]
import torch
import teacher_contract,oracle_teacher,saliency,target_contract
import one_question,selection_adapter,generate_backend,search,extract,baseline,e1_baseline,e1_admission,run_control,panel,lazy_truth,runtime_gate,runtime_worker,runtime_launch,verify_contract,verify,container_identity
plan=json.loads((ROOT/'e4/identity_plan.json').read_text()) if (ROOT/'e4/identity_plan.json').is_file() else None
# Resolve recipe and expected library versions through the already accepted E4 identity plan.
from identity_worker import PLAN
author=json.loads(PLAN.read_text());sys.path[:0]=[author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
from transformers import AutoTokenizer
source=(Path(author['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
if Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=source:raise ValueError('unexpected author wrapper source')
versions={k:importlib.metadata.version(k) for k in author['expected_versions']}
if versions!=author['expected_versions'] or torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise ValueError('unexpected runtime or CUDA state')
# Tokenizer metadata only, no sample text, labels or model weights.
tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
print(json.dumps({'status':'passed','cuda_initialized':torch.cuda.is_initialized(),'model_loaded':False,'GT_opened':False,
                  'predictions_opened':False,'tokenizer_length':len(tokenizer),'author_wrapper':str(source),
                  'versions':versions}),flush=True)
