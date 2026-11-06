"""CPU-only source and real mismatch input check; no GPU entry is provided."""
import argparse,hashlib,importlib.util,inspect,json,os
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[2]
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/model/pruner.py'
CASE=BASE/'e5/selector_real32/run_001/image_017/case_512.pt'
PROOF=BASE/'e5/selector_real32/run_001_verification.json'
JOURNAL=BASE/'e5/selector_real32/run_001/journal.jsonl'
AUTHOR_SHA='bf809614c207142b8a4fc945f29a8828688c88a2896ebd349d55c4a0b1a917e6'
CASE_SHA='f651a859849d1e0c84119a7b708ffff794cbe8beaf60e0d9a8bf63d5ae24f95d'
PROOF_SHA='d6f22f3deec8c6ceae8305adb01e6e03653d65933bac954d542dd21bdb0d682d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check():
    expected_torch=Path('/home/dell/miniconda3/envs/qwen3vl_serving/lib/python3.10/site-packages/torch/__init__.py')
    if torch.__version__!='2.8.0+cu128' or Path(torch.__file__).resolve()!=expected_torch.resolve():
        raise ValueError('unexpected torch import or version')
    if torch.cuda.is_initialized():raise ValueError('CUDA was initialized')
    if (sha(AUTHOR)!=AUTHOR_SHA or sha(CASE)!=CASE_SHA or sha(PROOF)!=PROOF_SHA):
        raise ValueError('frozen source or evidence identity changed')
    proof=json.loads(PROOF.read_text())
    if (proof.get('status')!='passed' or proof.get('GT_opened') is not False
        or proof.get('scoring_performed') is not False
        or proof.get('array_sha256',{}).get(str(CASE))!=CASE_SHA
        or proof.get('files_sha256',{}).get(str(JOURNAL))!=sha(JOURNAL)):
        raise ValueError('real32 evidence incomplete')
    matches=[]
    with JOURNAL.open() as f:
        for line in f:
            r=json.loads(line)
            if r.get('status')=='case_completed' and r.get('image_index')==17 and r.get('budget')==512:
                matches.append(r)
    if len(matches)!=1:raise ValueError('real mismatch case not unique')
    identity=matches[0]['result']['identity']
    if (identity.get('author_equals_endpoint') is not True or identity.get('fused_equals_author') is not False
        or identity.get('graph_equals_author') is not False):raise ValueError('real mismatch evidence differs')
    spec=importlib.util.spec_from_file_location('eadp_fixed_author_pruner',AUTHOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(inspect.getfile(module._greed_select_impl)).resolve()!=AUTHOR.resolve():raise ValueError('author import mismatch')
    case=torch.load(CASE,map_location='cpu',weights_only=True)
    importance=case['importance'];sim=case['similarity'];order=case['ordered']
    if (importance.shape!=(1,1024) or sim.shape!=(1,1024,1024) or order.shape!=(1,512)
        or importance.dtype!=torch.float32 or sim.dtype!=torch.float32 or order.dtype!=torch.int64
        or not torch.isfinite(importance).all() or not torch.isfinite(sim).all()
        or len(set(order[0].tolist()))!=512 or int(order.min())<0 or int(order.max())>=1024):
        raise ValueError('archived case tensor contract failed')
    if torch.cuda.is_initialized():raise ValueError('CUDA unexpectedly initialized')
    return {'status':'passed','scope':'CPU-only frozen author and archived case identity; no GPU/graph execution',
            'author_sha256':AUTHOR_SHA,'case_sha256':CASE_SHA,'real32_proof_sha256':PROOF_SHA,
            'real32_journal_sha256':sha(JOURNAL),'image_index':17,'k':512,
            'importance_shape':list(importance.shape),'similarity_shape':list(sim.shape),
            'stored_order_shape':list(order.shape),'torch_version':torch.__version__,
            'torch_file':str(Path(torch.__file__).resolve()),'cuda_initialized':False,'GT_opened':False,
            'new_predictions_opened':False,'scoring_performed':False,'gpu_authorized':False,
            'limitations':['does not execute author greedy, graph capture or replay',
                           'real32 formal evidence binds archive; independent real32 review must also precede later GPU admission']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.resolve()!=HERE/'cpu_preflight003.json':raise ValueError('unexpected output path')
    result=check()
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__':main()
