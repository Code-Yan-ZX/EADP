"""One-image, forced-prefix diagnostic of PixelPrune on real Qwen3-VL 8B.

The published patch is applied only inside this process after an unpatched
control. No labels or quality measurements; explicit disabled is an identity
control and enabled threshold zero may still remove duplicate visual patches.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
BASE = ROOT / 'reproduction/20261104/deepstack_compression'
HERE = BASE / 'e6/pixelprune_gpu_smoke'
PIXEL = BASE / 'e6/external_sources/pixelprune_8d347eb'
AUTHOR = ROOT / '_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
MODEL = Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path[:0] = [str(AUTHOR), str(PIXEL)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def progress(path, event):
    with open(path, 'a') as f:
        f.write(json.dumps(event) + '\n'); f.flush(); os.fsync(f.fileno())


def weight_stats(weights):
    result = []
    for x in weights['files']:
        path = Path(x['path']); s = path.stat()
        if s.st_size != x['bytes'] or x['sha256'] != x['expected_lfs_sha256']:
            raise ValueError('weight identity')
        result.append({'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
                       'inode': s.st_ino, 'device': s.st_dev})
    if len(result) != 4: raise ValueError('weight count')
    return result


def preflight(plan):
    for path, digest in plan['sha256'].items():
        if sha(path) != digest: raise ValueError('source mismatch: ' + path)
    rows = [json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()]
    anchor = plan['anchor']
    found = [r for r in rows if r['task'] == anchor['task'] and str(r['qid']) == anchor['qid']]
    if len(found) != 1: raise ValueError('anchor uniqueness')
    row = found[0]
    digest = hashlib.sha256(json.dumps(row['message'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if (digest != anchor['message_sha256'] or digest != row['message_sha256']
            or sha(row['image_path']) != anchor['image_sha256']
            or [x['type'] for x in row['message']] != ['image', 'text']):
        raise ValueError('anchor mismatch')
    return row


def cache_shapes(cache):
    if cache is None: return []
    return [{'key': None if x.keys is None else list(x.keys.shape),
             'value': None if x.values is None else list(x.values.shape)} for x in cache.layers]


def digest_tensor(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def cache_prefix(cache, length):
    return [{'key': digest_tensor(x.keys[..., :length, :]), 'value': digest_tensor(x.values[..., :length, :])}
            for x in cache.layers]


def checks(arm, layers):
    n = arm.get('selected_count'); seq = arm['original_sequence_length']; trace = arm['trace']; post = arm['post']
    initial = None if n is None else seq - 1024 + n
    c = {'selected_count_valid': type(n) is int and 1 <= n <= 1024 and len(arm.get('selected_indices', [])) == n and
             arm.get('selected_indices') == sorted(set(arm.get('selected_indices', []))) and
             all(0 <= x < 1024 for x in arm.get('selected_indices', [])),
         'forced_33_tokens': arm.get('status') == 'completed' and arm.get('forced_tokens_match') is True,
         'one_prefill_32_decode': len(trace) == len(post) == 33,
         'prefill_positions': arm.get('prefill_positions_match') is True,
         'prefill_main_and_text': arm.get('prefill_main_match') is True and arm.get('prefill_text_match') is True,
         'prefill_three_ds': arm.get('prefill_ds_matches') == [True] * 3,
         'finite_logits': len(post) == 33 and all(x['finite_logits'] for x in post),
         'rope_delta': len(post) == 33 and all(x['rope_delta'] == arm['original_rope_delta'] for x in post),
         'decode_position': len(trace) == 33 and all(x['positions'] == [[[arm['original_max_position']+i]]] * 3
                                                     for i, x in enumerate(trace) if i > 0),
         'cache_lengths': initial is not None and len(post) == 33 and all(
             len(x['layers']) == layers and all(z['key'] is not None and z['value'] is not None
                and z['key'][-2] == initial+i and z['value'][-2] == initial+i for z in x['layers'])
             for i, x in enumerate(post)),
         'cache_prefix_unchanged': len(arm['prefix']) == 2 and arm['prefix'][0] == arm['prefix'][1]}
    if arm['name'] == 'patched_disabled':
        c['disabled_logits_match_unpatched'] = len(arm['logit_checks']) == 33 and all(
            x.get('comparable') and x.get('allclose') and x.get('argmax_equal') for x in arm['logit_checks'])
    return c


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads((HERE/'plan001.json').read_text()); row = preflight(plan)
    import torch
    import transformers
    from transformers import AutoProcessor, LogitsProcessor, Qwen3VLForConditionalGeneration
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
    from pixelprune import apply_pixelprune
    import pixelprune.patches.qwen3_vl_hf as patch
    if Path(inspect.getfile(patch)).resolve() != PIXEL/'pixelprune/patches/qwen3_vl_hf.py': raise ValueError('wrong pixel source')
    weights = json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text()); stats = weight_stats(weights)
    if args.preflight_only:
        if torch.cuda.is_initialized(): raise ValueError('CUDA initialized')
        for x in weights['files']:
            if sha(x['path']) != x['expected_lfs_sha256']: raise ValueError('weight SHA mismatch')
        if weight_stats(weights) != stats: raise ValueError('weights changed')
        save(args.out, {'status':'passed', 'source_plan_sha256':sha(HERE/'plan001.json'), 'cuda_initialized':False,
             'model_loaded':False, 'GT_opened':False, 'weight_sha256_verified':4, 'weight_stats':stats,
             'torch':torch.__version__, 'transformers':transformers.__version__}); return
    if stats != json.loads((HERE/'preflight001.json').read_text())['weight_stats'] or torch.cuda.get_device_name(0) != 'NVIDIA A40':
        raise ValueError('weight/GPU mismatch')
    args.out.mkdir(exist_ok=False); progress(args.out/'progress.jsonl', {'event':'worker_started'})
    save(args.out/'started.json', {'plan_sha256':sha(HERE/'plan001.json'), 'task':row['task'], 'qid':row['qid'], 'GT_opened':False})
    os.environ.update(PIXELPRUNE_ENABLED='false', PIXELPRUNE_METHOD='pred_2d', PIXELPRUNE_METRIC='max', PIXELPRUNE_THRESHOLD='0.0', PIXELPRUNE_VIT_LAYER='0', PIXELPRUNE_VERBOSE='false')
    os.environ.pop('PIXELPRUNE_LOG_FILE', None)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
             attn_implementation='flash_attention_2').to('cuda').eval()
    wrapper = object.__new__(Qwen3VLChatFixedRes); wrapper.system_prompt=None; wrapper.min_pixels=wrapper.max_pixels=1048576
    wrapper.total_pixels=None; wrapper.processor=processor; wrapper.model=SimpleNamespace(device='cuda', dtype=torch.bfloat16)
    inputs=wrapper._processor_inputs(wrapper._build_messages(row['message'], dataset=row['task'])); ids=inputs['input_ids']; seq=ids.shape[1]
    orig, delta=model.model.get_rope_index(ids, inputs['image_grid_thw'], None, inputs['attention_mask'])
    image_pos=torch.where(ids[0] == model.config.image_token_id)[0]
    if len(image_pos) != 1024: raise ValueError('expected 1024 input image tokens')
    text_embeds=model.get_input_embeddings()(ids).detach(); token=processor.tokenizer.encode(' the',add_special_tokens=False)[0]
    class Force(LogitsProcessor):
        def __call__(self, _ids, scores):
            out=torch.full_like(scores, -float('inf')); out[:, token]=0; return out
    baseline=[]; arms=[]
    for name in ('unpatched', 'patched_disabled', 'patched_enabled'):
        if name == 'patched_disabled': apply_pixelprune(model='qwen3_vl')
        os.environ['PIXELPRUNE_ENABLED'] = 'true' if name == 'patched_enabled' else 'false'
        model.model.rope_deltas=None
        trace=[]; post=[]; prefix=[]; logit_checks=[]; vision=[]; selections=[]; prefill={}
        save(args.out/(name+'_started.json'), {'arm':name, 'pixelprune_enabled':name=='patched_enabled'})
        progress(args.out/'progress.jsonl', {'event':'arm_started','arm':name})
        def pre_vision(_module, _args, kw):
            keep=kw.get('keep_indices')
            if keep is None:
                selections.append(list(range(1024)))
            else:
                if len(keep)!=1: raise ValueError('unexpected image keep lists')
                patch_indices=keep[0].detach().cpu().tolist()
                if len(patch_indices)%4: raise ValueError('incomplete merged patch group')
                picked=[patch_indices[i]//4 for i in range(0,len(patch_indices),4)]
                if patch_indices != [4*j+k for j in picked for k in range(4)] or picked != sorted(set(picked)):
                    raise ValueError('invalid patch grouping or order')
                selections.append(picked)
        def post_vision(_module, _args, _kw, out):
            vision.append((out[0].detach(), [x.detach() for x in out[1]]))
        def pre_lm(_module, _args, kw):
            pos=kw.get('position_ids'); emb=kw.get('inputs_embeds'); mask=kw.get('visual_pos_masks'); ds=kw.get('deepstack_visual_embeds')
            if not trace:
                if len(selections)!=1 or len(vision)!=1 or emb is None or mask is None or ds is None: raise ValueError('missing prefill capture')
                picked=selections[0]; indices=torch.tensor(picked, device=ids.device, dtype=torch.long)
                keep=torch.ones(seq,device=ids.device,dtype=torch.bool); keep[image_pos]=False; keep[image_pos[indices]]=True
                v, ds_original=vision[0]; vm=mask[0].bool(); text_mask=ids[0]!=model.config.image_token_id
                prefill.update(positions_match=pos is not None and bool(torch.equal(pos,orig[:,:,keep])),
                    main_match=bool(torch.equal(emb[0,vm],v.to(emb.dtype))),
                    text_match=bool(torch.equal(emb[0,~vm],text_embeds[0,text_mask])),
                    ds_matches=[bool(torch.equal(a,b.to(a.dtype))) for a,b in zip(ds,ds_original)],
                    ds_count=len(ds), visual_count=int(mask.sum()))
            trace.append({'positions':None if pos is None else pos.detach().cpu().tolist(),
                          'input_length':None if emb is None else int(emb.shape[1])})
        def post_model(_module, _args, out):
            i=len(post); value=out.logits[:,-1,:].detach().float().cpu(); cache=out.past_key_values
            post.append({'finite_logits':bool(torch.isfinite(value).all()), 'rope_delta':None if out.rope_deltas is None else out.rope_deltas.detach().cpu().tolist(),
                         'layers':cache_shapes(cache)})
            if i in (0,32):
                initial=seq-1024+len(selections[0]); prefix.append(cache_prefix(cache, initial))
            if name=='unpatched': baseline.append(value)
            elif name=='patched_disabled':
                if i>=len(baseline) or value.shape!=baseline[i].shape: logit_checks.append({'comparable':False})
                else: logit_checks.append({'comparable':True, 'allclose':bool(torch.allclose(value,baseline[i],atol=.01,rtol=.01)),
                    'argmax_equal':bool(torch.equal(value.argmax(-1),baseline[i].argmax(-1))),
                    'max_abs_difference':float((value-baseline[i]).abs().max())})
            progress(args.out/'progress.jsonl', {'event':'forward_complete','arm':name,'call_index':i})
        handles=[model.model.visual.register_forward_pre_hook(pre_vision,with_kwargs=True),
                 model.model.visual.register_forward_hook(post_vision,with_kwargs=True),
                 model.model.language_model.register_forward_pre_hook(pre_lm,with_kwargs=True), model.register_forward_hook(post_model)]
        arm={'name':name, 'original_sequence_length':int(seq), 'original_rope_delta':delta.detach().cpu().tolist(),
             'original_max_position':int(orig.max()), 'threshold':0.0}
        try:
            with torch.inference_mode(): result=model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=33,
                min_new_tokens=33, logits_processor=[Force()], return_dict_in_generate=True)
            torch.cuda.synchronize(); tokens=result.sequences[0,seq:].tolist()
            save(args.out/(name+'_generated.json'),{'tokens':tokens,'forced_token_id':token})
            arm.update(status='completed',forced_tokens_match=tokens==[token]*33, generated_token_count=len(tokens))
        except Exception as exc:
            arm.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            save(args.out/(name+'_exception.json'),{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        finally:
            for h in handles: h.remove()
        arm.update(selected_indices=selections[0] if selections else [], selected_count=len(selections[0]) if selections else None,
                   prefill_positions_match=prefill.get('positions_match'), prefill_main_match=prefill.get('main_match'),
                   prefill_text_match=prefill.get('text_match'), prefill_ds_matches=prefill.get('ds_matches'),
                   trace=trace,post=post,prefix=prefix,logit_checks=logit_checks)
        arm['diagnostic_checks']=checks(arm,model.config.text_config.num_hidden_layers)
        save(args.out/(name+'.json'),arm); progress(args.out/'progress.jsonl',{'event':'arm_recorded','arm':name,'status':arm['status']}); arms.append(arm)
    if weight_stats(weights)!=stats or any(sha(p)!=h for p,h in plan['sha256'].items()): raise ValueError('weights/sources changed')
    save(args.out/'worker_complete.json',{'status':'diagnostic_completed','arm_statuses':{x['name']:x['status'] for x in arms},
         'warning':'attempts only; enabled uses variable visual budget and is not quality or speed evaluation'})


if __name__ == '__main__': main()
