"""E0 diagnostic worker; preserves output before interpreting identity checks."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
BASE = ROOT / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
PANEL = ROOT / 'reproduction/20261028/qwen/full1024_two_path_panel/plan.json'
OLD = ROOT / 'reproduction/20261028/qwen/doc_full_k256_ds_pair'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main(args):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from adapter import Adapt
    sys.path.insert(0, str(OLD))
    from selection_observer import SelectionObserver
    plan = json.loads((BASE/'run_plan.json').read_text())
    for group in ('source_sha256', 'implementation_sha256'):
        for path, digest in plan[group].items():
            if sha(path) != digest: raise ValueError('source mismatch: '+path)
    versions = {n:importlib.metadata.version(n) for n in plan['expected_versions']}
    if versions != plan['expected_versions']: raise ValueError('environment mismatch')
    sys.path[:0] = [plan['author_root'], str(Path(plan['author_root'])/'VLMEvalKit')]
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes, Qwen3VLChatCDPruner, Qwen3VLChatEADP
    from model import pruner as eadp_pruner
    torch.set_num_threads(4)
    if torch.cuda.get_device_name(0) != 'NVIDIA A40': raise ValueError('unexpected GPU')
    recipe = dict(plan['recipe']); recipe['visual_token_num'] = 1024
    w = Qwen3VLChatEADP(**recipe)
    if w.model.dtype != torch.bfloat16 or w.model.config._attn_implementation != 'flash_attention_2':
        raise ValueError('unexpected model configuration')
    panel = json.loads(PANEL.read_text())['panel'][:args.limit]
    write(args.out/'runtime.json', dict(recipe=recipe, versions=versions, pid=os.getpid(),
        gpu=torch.cuda.get_device_name(0), model_instances=1, panel_sha256=sha(PANEL),
        source_plan_sha256=sha(BASE/'run_plan.json'), worker_sha256=sha(__file__),
        adapter_sha256=sha(Path(__file__).with_name('adapter.py')), logits_abs_tolerance=0.0,
        teacher_steps=32, scoring=False, gt_read=False, start_unix=time.time()))

    def cpu(x):
        if isinstance(x, torch.Tensor): return x.detach().to('cpu').contiguous()
        if isinstance(x, (tuple, list)): return [cpu(v) for v in x]
        return x
    def meta(x):
        if x is None: return None
        if isinstance(x, list): return [meta(y) for y in x]
        return dict(dtype=str(x.dtype), shape=list(x.shape), sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
    class Fixed(LogitsProcessor):
        def __init__(self, token): self.token = token
        def __call__(self, input_ids, scores):
            scores.fill_(-float('inf')); scores[:, self.token] = 0
            return scores

    def run(item, mode, forced):
        vals = {'llm': [], 'logits': []}; handles = []
        def vis(_m, _a, out): vals['visual'] = cpu(out)
        def llm(_m, _a, kw):
            vals['llm'].append({k:cpu(kw.get(k)) for k in ('inputs_embeds','attention_mask','position_ids','cache_position','visual_pos_masks','deepstack_visual_embeds')})
            vals.setdefault('deltas', []).append(cpu(w.model.model.rope_deltas))
        def logits(_m, _a, out): vals['logits'].append(cpu(out[:, -1, :]))
        orig_generate = w.model.generate
        def generate(*a, **kw):
            if forced:
                kw.update(max_new_tokens=32, min_new_tokens=32,
                    logits_processor=LogitsProcessorList([Fixed(forced_id)]), eos_token_id=[])
            result = orig_generate(*a, **kw)
            vals['output_ids'] = cpu(result)
            return result
        w.model.generate = generate
        try:
            w.visual_token_num = w.pruner.visual_token_num = 1024 if mode in ('native','on') else 256
            keep = w.visual_token_num
            observer = SelectionObserver(eadp_pruner) if keep == 256 else None
            ob = observer if observer is not None else contextlib.nullcontext()
            intervention = Adapt(torch, w, keep, {'on':'on','off':'off','zero':'zero'}[mode], observer) if mode != 'native' else contextlib.nullcontext()
            with ob, intervention as ad:
                handles.append(w.model.visual.register_forward_hook(vis))
                handles.append(w.model.model.language_model.register_forward_pre_hook(llm, with_kwargs=True))
                handles.append(w.model.lm_head.register_forward_hook(logits))
                method = Qwen3VLChatFixedRes.generate_inner_transformers_fixed if mode == 'native' else Qwen3VLChatCDPruner.generate_inner_transformers_fixed
                with torch.inference_mode():
                    response = method(w, item['message'], dataset='DocVQA_VAL')
                if ad is not None:
                    for k in ('ordered', 'spatial', 'retained', 'full_delta', 'adjusted_delta'):
                        vals[k] = cpu(ad.state[k])
            if mode == 'native':
                vals['output_ids'] = vals['output_ids'][:, vals['llm'][0]['inputs_embeds'].shape[1]:]
            vals['response'] = response
            return vals
        finally:
            w.model.generate = orig_generate
            for h in handles: h.remove()

    forced_id = w.processor.tokenizer.encode(' the', add_special_tokens=False)[0]
    for i,item in enumerate(panel):
        if sha(item['image_path']) != item['image_sha256']: raise ValueError('image mismatch')
        for forced in (False, True):
            all_runs = {}
            for mode in ('native','on','off','zero'):
                t = time.monotonic(); value = run(item, mode, forced); value['seconds'] = time.monotonic()-t
                all_runs[mode] = value
                summary = {k:meta(v) for k,v in value.items() if k not in ('llm','logits','response','seconds')}
                summary.update(mode=mode, forced=forced, response=value['response'], seconds=value['seconds'],
                    output_ids=value['output_ids'].tolist(), llm=[{k:meta(v) for k,v in e.items()} for e in value['llm']],
                    logits=[meta(v) for v in value['logits']])
                write(args.out/f'{i:02d}_{mode}_{"forced" if forced else "natural"}.json', summary)
            a,b = all_runs['native'],all_runs['on']
            off,zero = all_runs['off'],all_runs['zero']
            def comp(x,y):
                if len(x['logits']) != len(y['logits']): return {'equal_length':False}
                diffs = [(u.float()-v.float()).abs().max().item() for u,v in zip(x['logits'],y['logits'])]
                return dict(equal_length=True, logits_max_abs=diffs,
                    logits_exact=all(d == 0 for d in diffs),
                    argmax_equal=[torch.equal(u.argmax(-1),v.argmax(-1)) for u,v in zip(x['logits'],y['logits'])],
                    output_equal=torch.equal(x['output_ids'],y['output_ids']),
                    positions_equal=[torch.equal(u['position_ids'],v['position_ids']) for u,v in zip(x['llm'],y['llm'])],
                    cache_equal=[torch.equal(u['cache_position'],v['cache_position']) for u,v in zip(x['llm'],y['llm'])])
            result = dict(index=item['index'], forced=forced, native_on=comp(a,b), off_zero=comp(off,zero),
                off_zero_selection_equal=torch.equal(off['ordered'],zero['ordered']) and torch.equal(off['spatial'],zero['spatial']),
                visual_equal=all(torch.equal(x,y) for x,y in zip([a['visual'][0]]+a['visual'][1],[b['visual'][0]]+b['visual'][1])),
                prefill_embed_equal=torch.equal(a['llm'][0]['inputs_embeds'],b['llm'][0]['inputs_embeds']),
                prefill_mask_equal=torch.equal(a['llm'][0]['attention_mask'],b['llm'][0]['attention_mask']),
                forced_token=forced_id if forced else None)
            write(args.out/f'{i:02d}_comparison_{"forced" if forced else "natural"}.json', result)
    write(args.out/'worker_done.json', dict(n=args.limit, forced_steps=32, complete=True, end_unix=time.time()))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--limit',type=int,default=1)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    main(a)
