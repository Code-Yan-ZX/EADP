"""Locked, reviewed-plan M3 worker. Import/preflight does not execute main."""
import argparse,importlib.metadata,inspect,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import runtime_gate,e1_admission,panel,lazy_truth

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if not args.execute:raise ValueError('explicit execution required')
    approval=runtime_gate.validate(host=False,preflight_only=args.preflight_only) # Must precede model, prediction and reference access.
    out=args.out.resolve();launch=json.loads((out/'launch.json').read_text());receipt_path=out/'host_receipt.json'
    if launch.get('host_receipt')!={'path':str(receipt_path),'sha256':runtime_gate.sha(receipt_path)} or json.loads(receipt_path.read_text())!=approval:
        raise ValueError('missing host admission/inspect receipt')
    plan=json.loads((HERE/'runtime_plan.json').read_text());e1=e1_admission.validate(metadata_only=args.preflight_only)
    proofpath=ROOT/'e1/e1full001_completion.json';proof_bytes=proofpath.read_bytes();proof=json.loads(proof_bytes)
    if runtime_gate.sha(proofpath)!=e1.proof_sha:raise ValueError('E1 proof changed')
    samples=Path(plan['sample_path']).read_bytes();sample=json.loads(samples)
    inputs=Path(proof['files']['inputs']['path']).read_bytes()
    if __import__('hashlib').sha256(inputs).hexdigest()!=e1.inputs_sha:raise ValueError('E1 inputs changed')
    input_rows=[json.loads(x) for x in inputs.splitlines()]
    if len(input_rows)!=384 or len(sample.get('examples',[]))!=128:raise ValueError('frozen inventories incomplete')
    for item in sample['examples']:
        matches=[r for r in input_rows if (r.get('task'),str(r.get('qid')),r.get('group_id'))==(item['task'],str(item['qid']),item['group_id'])]
        if len(matches)!=1 or any(matches[0].get(k)!=item.get(k) for k in ('image_path','image_sha256','message_sha256')):
            raise ValueError('frozen sample input mismatch')
    source_meta=json.loads(Path(proof['files']['dataset_summary']['path']).read_text());truth=lazy_truth.LazyTruth(source_meta['sources'],sample['examples'])
    source_plan=json.loads(Path(proof['files']['plan']['path']).read_text());recipe=dict(source_plan['recipe']);recipe['visual_token_num']=256
    sys.path[:0]=[str(HERE),str(ROOT/'m3_oracle'),str(ROOT/'e4'),source_plan['author_root'],str(Path(source_plan['author_root'])/'VLMEvalKit')]
    import torch
    from model import pruner as native
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
    from transformers import AutoTokenizer
    import oracle_teacher,saliency,target_contract,teacher_contract
    if {n:importlib.metadata.version(n) for n in source_plan['expected_versions']}!=source_plan['expected_versions']:
        raise ValueError('runtime version mismatch')
    source_path=(Path(source_plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
    if Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=source_path:raise ValueError('author wrapper source mismatch')
    native_path=Path(native.__file__).resolve()
    if str(native_path) not in {**source_plan['source_sha256'],**source_plan['implementation_sha256']}:
        raise ValueError('author pruner source not frozen')
    tokenizer=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
    if args.preflight_only:
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized during preflight')
        print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,
                          'predictions_opened':False,'inputs_verified':384,'sample_verified':128,
                          'weights_sha256_verified':4,'pruner_path':str(native_path),'wrapper_path':str(source_path),
                          'tokenizer_length':len(tokenizer),'prediction_journal_byte_hash_deferred':True}),flush=True)
        return 0
    journal=Path(proof['files']['journal']['path']).read_bytes()
    if torch.cuda.get_device_name(0)!=plan['gpu_name'] or torch.is_inference_mode_enabled():raise ValueError('invalid GPU or autograd mode')
    wrapper=Qwen3VLChatEADP(**recipe)
    if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
    wrapper.model.eval();wrapper.model.requires_grad_(False)
    eos=wrapper.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
    scalars={k:recipe[k] for k in ('top_p','top_k','temperature','repetition_penalty')};scalars.update(do_sample=False,max_new_tokens=2048)
    def author_generate(w,row):
        with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
    def validate_input(row,inputs):
        if row.get('split')!='DEV' or not all(k in inputs for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')):
            raise ValueError('invalid frozen DEV processor inputs')
    metadata={'stage':'oracle_panel','approval':approval,'plan_sha256':runtime_gate.sha(HERE/'runtime_plan.json'),
              'sample_sha256':plan['sample_sha256'],'e1_completion_sha256':e1.proof_sha,'e1_review_sha256':e1.review_sha,
              'weights_identity_sha256':e1.weights_sha,'recipe_plan_sha256':e1.plan_sha,'GT_opened':False,'scoring':False}
    with (out/'frozen.json').open('x') as f:json.dump(metadata,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    panel.run(samples,plan['sample_sha256'],proof_bytes,journal,inputs,e1,out/'panel',plan['budget_end_unix'],
              plan['question_checkpoint_seconds'],wrapper,native,torch,oracle_teacher,saliency,target_contract,teacher_contract,
              truth,author_generate,scalars,eos,validate_input)
    return 0
if __name__=='__main__':sys.exit(main())
