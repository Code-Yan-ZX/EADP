"""16-generation DEV identity worker, gated by independent approval.

It makes no scoring or timing claims. The host launcher must create the
identity-bound launch record before this worker may initialize a GPU.
"""
import argparse, importlib.metadata, inspect, json, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
E4=BASE/'e4'
from real_worker import PLAN, WEIGHTS, sha, source_and_weights, preflight_one

def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def continue_smoke(reference_path,out,preflight_only=False):
    """Called only by a future reviewed admission path; caller owns GPU lock."""
    reference_path=Path(reference_path).resolve();out=Path(out).resolve()
    ref=json.loads(reference_path.read_text())
    if ref.get('status')!='passed' or len(ref.get('rows',[]))!=3 or len(ref.get('b_records',[]))!=8 or len(ref.get('old_on',[]))!=4:
        raise ValueError('invalid accepted reference inventory')
    if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','worker_complete.json','generated.jsonl','records.jsonl','arrays')):
        raise ValueError('invalid or reused output')
    plan=json.loads(PLAN.read_text());recipe=dict(plan['recipe']);recipe['visual_token_num']=256
    expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('recipe changed')
    weights_before=source_and_weights(plan)
    from light_admission import sources
    sources=sources()
    frozen={str(p):sha(p) for p in sources}
    sys.path[:0]=[str(E4),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from model import pruner
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    from budget_runner import run_arm as fresh_runner
    from light_arm import run_arm as light_runner,CacheFingerprints
    from light_smoke_core import run_diagnostics
    versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
    expected_pruner=(Path(plan['author_root'])/'model/pruner.py').resolve()
    expected_wrapper=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
    if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=expected_pruner or
        Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=expected_wrapper or
        Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected_wrapper or
        Path(inspect.getfile(fresh_runner)).resolve()!=(E4/'budget_runner.py').resolve() or
        Path(inspect.getfile(light_runner)).resolve()!=(HERE/'light_arm.py').resolve()):raise ValueError('runtime import identity changed')
    if preflight_only:
        if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
        result={'status':'passed','cuda_initialized':False,'model_loaded':False,'predictions':False,'reference_sha256':sha(reference_path),
                'source_sha256':frozen,'weight_stats_before':weights_before,'versions':versions,'plan_sha256':sha(PLAN),'weights_identity_sha256':sha(WEIGHTS),
                'scope':'complete worker model-before-load CPU path; no trained model, generation, GT or GPU'}
        save(out/'worker_preflight.json',result);print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False}),flush=True)
        return
    if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
    (out/'arrays').mkdir()
    save(out/'frozen.json',{'stage':'light_adapter_identity','references_path':str(reference_path),'references_sha256':sha(reference_path),
         'source_sha256':frozen,'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':weights_before,
         'plan_sha256':sha(PLAN),'jobs':ref['plan']['jobs'],'gt_read':False,'scoring':False,'timing_claims':False})
    wrapper=Qwen3VLChatEADP(**recipe)
    if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend changed')
    for row in ref['rows']:
        preflight_one(torch,wrapper,row,{'processor':row['processor_expected'],'image_start':row['image_start']})
    import gc
    gc.collect();torch.cuda.synchronize()
    tokens=wrapper.processor.tokenizer.encode(' the',add_special_tokens=False)
    if not tokens:raise ValueError('forced token missing')
    token=int(tokens[0])
    def generate(w,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
    def fresh_with_cache(torch,w,pruner,row,budget,ds_on,generate,journal,archive,forced_token,capture_prefill_logits=False):
        with CacheFingerprints(w,forced_token is not None) as kv:
            rec=fresh_runner(torch,w,pruner,row,budget,ds_on,generate,journal,archive,forced_token,capture_prefill_logits=capture_prefill_logits)
        rec['cache_fingerprints']=kv.fingerprints
        return rec
    def archive_factory(name):
        def archive(tensors):
            import numpy as np
            dst=out/'arrays'/(name+'.npz')
            data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
            with dst.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
            return {'kind':'full_arrays','path':str(dst),'sha256':sha(dst)}
        return archive
    result=run_diagnostics(torch,wrapper,pruner,ref['rows'],ref['plan']['anchors'],ref['b_records'],ref['old_on'],out,generate,fresh_with_cache,light_runner,archive_factory,token)
    if any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('source changed during run')
    for path,before in weights_before.items():
        s=Path(path).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
    if sha(reference_path)!=json.loads((out/'frozen.json').read_text())['references_sha256']:raise ValueError('reference changed during run')
    save(out/'worker_complete.json',dict(result,finished_unix=time.time(),forced_token=token))

def main():
    p=argparse.ArgumentParser();p.add_argument('--references',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    from light_admission import admit
    approval=admit(preflight_only=args.preflight_only)
    if args.references.resolve()!=Path(approval['reference_path']).resolve() or sha(args.references)!=approval['reference_sha256']:raise ValueError('reference admission mismatch')
    launch=json.loads((args.out/'launch.json').read_text())
    if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__):raise ValueError('launcher identity mismatch')
    continue_smoke(args.references,args.out,preflight_only=args.preflight_only)

if __name__=='__main__':main()
