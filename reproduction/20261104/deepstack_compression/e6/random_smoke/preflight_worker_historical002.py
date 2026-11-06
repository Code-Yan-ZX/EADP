"""Same-container no-GPU real import/weight/input preflight; never loads a model."""
import argparse,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
if hashlib.sha256((D/'runtime_gate.py').read_bytes()).hexdigest()!='083e0a69b4296c7f02c82d31c96d567d2f4026fabd2be7a40c644c4ff43d31b4':raise ValueError('runtime gate bytes changed')
sys.path.insert(0,str(D))
if 'runtime_gate' in sys.modules and Path(getattr(sys.modules['runtime_gate'],'__file__','')).resolve()!=D/'runtime_gate.py':raise ValueError('foreign runtime gate')
import runtime_gate as gate
if Path(gate.__file__).resolve()!=D/'runtime_gate.py':raise ValueError('wrong runtime gate')
def checked(name,path):
 mod=sys.modules.get(name)
 if mod is not None and (not getattr(mod,'__file__',None) or Path(mod.__file__).resolve()!=Path(path).resolve()):raise ValueError('foreign cached '+name)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();gate.verify_plan();gate.require_review()
 import torch,transformers
 if torch.cuda.is_initialized():raise ValueError('preinitialized CUDA')
 author=ROOT/'_reproduction/author_eadp/Qwen_vl';e4=B/'e4';rb=B/'e6/random_baselines';sys.path[:0]=[str(rb),str(e4),str(author),str(author/'VLMEvalKit')]
 for name,path in {'selector':rb/'selector.py','off_adapter':e4/'off_adapter.py','skip_deepstack':e4/'skip_deepstack.py','budget_adapter':e4/'budget_adapter.py','identity_records':e4/'identity_records.py','decode_observer':e4/'decode_observer.py'}.items():checked(name,path)
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 if Path(pruner.__file__).resolve()!=(author/'model/pruner.py').resolve() or any(Path(inspect.getfile(c)).resolve()!=(author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() for c in (Qwen3VLChatEADP,Qwen3VLChatCDPruner)):raise ValueError('author import mismatch')
 spec=importlib.util.spec_from_file_location('eadp_random_preflight_adapter',rb/'adapter.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if Path(inspect.getfile(mod.RandomAdapter)).resolve()!=rb/'adapter.py' or Path(sys.modules['selector'].__file__).resolve()!=rb/'selector.py':raise ValueError('random module mismatch')
 weights=gate.read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];observed=[]
 if len(weights)!=4:raise ValueError('weight count')
 for x in weights:
  q=Path(x['path']);before=q.stat()
  if before.st_size!=x['bytes'] or x['sha256']!=x['expected_lfs_sha256'] or gate.sha(q)!=x['expected_lfs_sha256']:raise ValueError('weight mismatch')
  after=q.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('weight changed')
  observed.append({'path':str(q),'bytes':before.st_size,'sha256':x['expected_lfs_sha256']})
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 gate.verify_plan();out={'status':'passed','GT_opened':False,'predictions_opened':True,'prediction_scope':'previously accepted DEV mixed prediction files opened only for source SHA; not parsed','model_loaded':False,'cuda_initialized':False,'weight_count':len(observed),'weights':observed,'torch':torch.__version__,'transformers':transformers.__version__,'runtime_plan_sha256':gate.sha(gate.PLAN),'scope':'imports and four weight payloads only; does not establish model identity or GPU approval'}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
