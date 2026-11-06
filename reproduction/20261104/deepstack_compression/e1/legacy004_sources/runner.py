"""E1 four-arm generator. Requires frozen input mapping and passed E0 gate.

Importing or executing without --execute never imports torch or loads a model.
This generator does not score predictions. See README for prerequisites.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from contract import ARMS, read_jsonl, sha, validate_manifest, validate_anchor_exclusion, validate_quartet, validate_decode_trace

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
PROJECT = ROOT/'reproduction/20261104/deepstack_compression'
E0 = PROJECT/'e0_identity'
PLAN = ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
OLD = ROOT/'reproduction/20261028/qwen/doc_full_k256_ds_pair'
OLD_PANEL = ROOT/'reproduction/20261028/qwen/full1024_two_path_panel/plan.json'


def validate_gate(proof_path):
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('status') != 'passed' or proof.get('n_expected') != 16 or proof.get('n_verified') != 16 or proof.get('errors'):
        raise ValueError('E0 full anchor gate is not passed')
    for field, path in (('worker_sha256', E0/'array_worker.py'), ('adapter_sha256', E0/'adapter.py'),
                        ('run_manifest_sha256', E0/'array16_launch.json'), ('verifier_sha256', E0/'verify_arrays.py'),
                        ('supervisor_terminal_sha256', E0/'monitor_array16/terminal.json'),
                        ('container_exit_sha256', E0/'array16_exit.json'),
                        ('evidence_manifest_sha256',E0/'anchor16_evidence_manifest.json'),
                        ('weight_postrun_sha256',E0/'array16_weights_postrun.json'),
                        ('worker_midrun_sha256',E0/'array16_worker_midrun.json'),
                        ('weight_midrun_sha256',E0/'array16_weight_midrun.json')):
        if proof.get(field) != sha(path):
            raise ValueError(f'E0 {field} differs from currently used implementation')
    terminal=json.loads((E0/'monitor_array16/terminal.json').read_text())
    container=json.loads((E0/'array16_exit.json').read_text())
    if terminal['reason']!='completed' or terminal['worker_returncode']!=0 or terminal['remaining_process_group']:
        raise ValueError('E0 supervision has not exited cleanly')
    verifier_terminal=json.loads((E0/'monitor_verify_array16/terminal.json').read_text())
    if verifier_terminal['reason']!='completed' or verifier_terminal['worker_returncode']!=0 or verifier_terminal['remaining_process_group']:
        raise ValueError('E0 verifier has not exited cleanly')
    if container.get('returncode')!=0 or container.get('timed_out') or container.get('gpu_pids_after'):
        raise ValueError('E0 container exit not clean')
    if not json.loads((E0/'array16_weights_postrun.json').read_text()).get('all_match'):
        raise ValueError('E0 postrun weight hash evidence failed')
    evidence=json.loads((E0/'anchor16_evidence_manifest.json').read_text())
    expected={str(E0/'array16'/f'{i:02d}_{arm}_{kind}.{ext}')
              for i in range(16) for arm in ('native','on','off','zero','k256_on')
              for kind in ('natural','forced') for ext in ('json','npz')}
    expected.update(str(E0/'array16'/f'{i:02d}_comparison_{kind}.json') for i in range(16) for kind in ('natural','forced'))
    if set(evidence)!=expected or any(sha(path)!=digest for path,digest in evidence.items()):
        raise ValueError('E0 anchor evidence differs from verified manifest')
    return proof


def validate_input_records(panel, records, mapping):
    if len(panel) != len(records):
        raise ValueError('input mapping and DEV panel size differ')
    indexed = {}
    for x in mapping:
        if x.get('split') != 'DEV': continue
        key=(x['task'],str(x['qid']))
        if key in indexed: raise ValueError('duplicate audited runtime path')
        indexed[key]=x
    if len(indexed)!=len(panel): raise ValueError('audited mapping coverage mismatch')
    for p, r in zip(panel, records):
        if any(str(p[k]) != str(r[k]) for k in ('task', 'qid', 'group_id')):
            raise ValueError('input mapping must exactly follow frozen DEV panel')
        if not isinstance(r.get('message'), list) or len([x for x in r['message'] if x.get('type') == 'image']) != 1:
            raise ValueError('input mapping requires one image message')
        paths = [x['value'] for x in r['message'] if x['type'] == 'image']
        approved=indexed[(r['task'],str(r['qid']))]
        if (approved['group_id']!=r['group_id'] or approved['status']!='found' or
            approved['expected_path']!=r['image_path'] or approved['cache_byte_sha256']!=r['image_sha256']):
            raise ValueError('image does not match audited task/qid mapping')
        if paths != [r['image_path']] or sha(r['image_path']) != r['image_sha256']:
            raise ValueError('runtime image path or SHA mismatch')
        digest = hashlib.sha256(json.dumps(r['message'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest != r['message_sha256']:
            raise ValueError('prompt/message identity mismatch')
        if any(k in r for k in ('answer', 'answers', 'ground_truth', 'prediction')):
            raise ValueError('generation input mapping must not contain answers')


def meta(x):
    if x is None: return None
    a = x.detach().contiguous().cpu()
    return {'dtype': str(a.dtype), 'shape': list(a.shape),
            'sha256': hashlib.sha256(a.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}


def append(path, record):
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
        f.flush()
        os.fsync(f.fileno())


def verified_inputs(manifest_dir, input_path, proof_path):
    panel=validate_manifest(manifest_dir)
    validate_anchor_exclusion(manifest_dir, OLD_PANEL)
    proof=json.loads(proof_path.read_text())
    if proof.get('status')!='passed' or proof.get('n_verified')!=384 or proof.get('inputs_sha256')!=sha(input_path):
        raise ValueError('explicitly verified task prompt/image mapping required')
    checks={'verifier_sha256':Path(__file__).with_name('verify_inputs.py'),
            'mapping_sha256':manifest_dir/'panel_runtime_paths.jsonl',
            'path_summary_sha256':manifest_dir/'panel_runtime_paths_summary.json',
            'author_prompt_sha256':ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py',
            'localized_tsv_sha256':Path('/media/disk2/YZX/LMUData/TextVQA_VAL_local.tsv'),
            'original_textvqa_sha256':Path('/media/disk2/YZX/LMUData/TextVQA_VAL.tsv')}
    for key,path in checks.items():
        if proof.get(key)!=sha(path): raise ValueError('input proof source changed: '+key)
    if proof.get('original_textvqa_verified_questions')!=128: raise ValueError('original/localized TextVQA check missing')
    if proof.get('panel_sha256')!=sha(manifest_dir/'panel_manifest.jsonl'): raise ValueError('different panel')
    records=read_jsonl(input_path)
    validate_input_records(panel,records,read_jsonl(manifest_dir/'panel_runtime_paths.jsonl'))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--e0-proof', type=Path, required=True)
    p.add_argument('--manifest-dir', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--input-proof', type=Path, required=True)
    p.add_argument('--stage', choices=('smoke','full'), required=True)
    p.add_argument('--smoke-proof', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not a.execute:
        raise ValueError('execution disabled without --execute; use tests for CPU checks')
    e0 = validate_gate(a.e0_proof)
    records=verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
    if a.stage=='full':
        if a.smoke_proof is None: raise ValueError('full run requires independent smoke and timing proof')
        # Fail closed until a separate smoke verifier and its interface are implemented.
        raise ValueError('full run locked: independent smoke verifier not yet implemented')
    # P1 equivalence to the unmodified legacy wrapper is a separate 16-anchor
    # generation gate; never infer it from the compact-position formula.
    raise ValueError('E1 smoke locked: legacy P1 16-anchor gate not yet implemented')
    records = [r for task,n in (('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10))
               for r in [x for x in records if x['task']==task][:n]]
    if len(records)!=32: raise ValueError('32-question stratified smoke required')
    plan = json.loads(PLAN.read_text())
    for group in ('source_sha256', 'implementation_sha256'):
        for path, digest in plan[group].items():
            if sha(path) != digest: raise ValueError('upstream source changed: '+path)
    weights_path=ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json'
    weights=json.loads(weights_path.read_text())
    if len(weights['files'])!=4: raise ValueError('expected four checkpoint shards')
    weight_stats={}
    for info in weights['files']:
        path=Path(info['path']); st=path.stat()
        if st.st_size != info['bytes'] or sha(path) != info['expected_lfs_sha256']:
            raise ValueError('checkpoint weight payload mismatch: '+str(path))
        weight_stats[str(path)]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
    a.out.mkdir(parents=True, exist_ok=False)
    frozen = {'e0_proof_sha256': sha(a.e0_proof), 'e0_adapter_sha256': e0['adapter_sha256'],
              'e0_verifier_terminal_sha256':sha(E0/'monitor_verify_array16/terminal.json'),
              'panel_sha256': sha(a.manifest_dir/'panel_manifest.jsonl'), 'inputs_sha256': sha(a.inputs),
              'anchor_exclusion':validate_anchor_exclusion(a.manifest_dir,OLD_PANEL),
              'input_proof_sha256': sha(a.input_proof), 'plan_sha256': sha(PLAN),
              'runner_sha256': sha(__file__), 'contract_sha256': sha(Path(__file__).with_name('contract.py')),
              'position_adapter_sha256':sha(Path(__file__).with_name('position_adapter.py')),
              'weights_identity_sha256':sha(weights_path), 'weight_stats_before':weight_stats,
              'arms': ARMS, 'n_questions': len(records), 'stage':a.stage, 'gt_read': False, 'scoring': False}
    (a.out/'frozen.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n')

    sys.path[:0] = [str(E0), str(OLD), plan['author_root'], str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from position_adapter import FourArm
    from selection_observer import SelectionObserver
    from model import pruner as eadp_pruner
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner, Qwen3VLChatEADP
    if {n: importlib.metadata.version(n) for n in plan['expected_versions']} != plan['expected_versions']:
        raise ValueError('runtime versions differ from frozen author recipe')
    if torch.cuda.get_device_name(0) != 'NVIDIA A40': raise ValueError('unexpected GPU')
    recipe = dict(plan['recipe'])
    recipe['visual_token_num'] = 256
    w = Qwen3VLChatEADP(**recipe)
    if w.model.dtype != torch.bfloat16 or w.model.config._attn_implementation != 'flash_attention_2':
        raise ValueError('unexpected dtype or attention')

    journal = a.out/'raw_predictions.jsonl'
    class ForceToken(LogitsProcessor):
        def __init__(self, token): self.token=token
        def __call__(self, input_ids, scores):
            scores.fill_(-float('inf')); scores[:,self.token]=0
            return scores
    forced_id=w.processor.tokenizer.encode(' the',add_special_tokens=False)[0]
    for row, forced in ((r, forced) for r in records for forced in (False, True)):
        group=[]
        for arm in ARMS:
            seen = {'trace':[]}
            def observe(_module, _args, kw):
                if 'prefill' not in seen:
                    seen['prefill'] = {k:kw.get(k) for k in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')}
                    seen['delta'] = w.model.model.rope_deltas.detach().cpu().tolist()
                if forced:
                    cache=kw.get('past_key_values')
                    seen['trace'].append({'position_ids':kw['position_ids'].detach().cpu().tolist(),
                         'cache_position':kw['cache_position'].detach().cpu().tolist(),
                         'layer_cache_lengths':[] if cache is None else [cache.get_seq_length(i) for i in range(len(w.model.model.language_model.layers))]})
            original_generate = w.model.generate
            def capture_generate(*args, **kw):
                if forced:
                    kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken(forced_id)]))
                eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id)
                seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
                generated = original_generate(*args, **kw)
                seen['output_ids'] = generated.detach().cpu().tolist()[0]
                return generated
            w.model.generate = capture_generate
            observer = SelectionObserver(eadp_pruner)
            handler = None
            started = time.monotonic()
            try:
                with observer, FourArm(torch, w, arm, observer) as ad:
                    handler = w.model.model.language_model.register_forward_pre_hook(observe, with_kwargs=True)
                    with torch.inference_mode():
                        answer = Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w, row['message'], dataset=row['task'])
                    s = ad.state
                    inp = s['inputs']; pf = seen['prefill']; start = s['start']
                    full_pos, _ = w.model.model.get_rope_index(inp['input_ids'], inp['image_grid_thw'], None, attention_mask=inp['attention_mask'])
                    selected = [meta(z) for z in s['selected']]
                    actual_ds = None if pf['deepstack_visual_embeds'] is None else [meta(z) for z in pf['deepstack_visual_embeds']]
                    rec = {k:row[k] for k in ('task','qid','group_id','message_sha256')}
                    rec.update(arm=arm, processor={k:meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
                        image_start=start, original_length=inp['input_ids'].shape[1],
                        ordered_ids=s['ordered'].detach().cpu().tolist(), spatial_ids=s['spatial'].detach().cpu().tolist(),
                        retained_ids=s['retained'].detach().cpu().tolist(), original_position_ids=full_pos.detach().cpu().tolist(),
                        full_rope_delta=s['full_delta'].detach().cpu().tolist(),
                        visual_sha256=meta(s['visual']), pruned_sha256=meta(s['pruned']),
                        deepstack_full_sha256=[meta(z) for z in s['deepstack']],
                        prefill_embed_sha256=meta(pf['inputs_embeds']), attention_mask_sha256=meta(pf['attention_mask']),
                        prefill_position_ids=pf['position_ids'].detach().cpu().tolist(), rope_delta=seen['delta'],
                        injected_mask=None if pf['visual_pos_masks'] is None else pf['visual_pos_masks'].detach().cpu().tolist(),
                        injected_deepstack=actual_ds, selected_deepstack=selected,
                        output_ids=seen['output_ids'], answer=answer, forced=forced, eos_ids=seen['eos_ids'],
                        eos=bool(seen['output_ids'] and seen['output_ids'][-1] in seen['eos_ids']),
                        truncated=bool(not forced and len(seen['output_ids'])>=2048 and (not seen['output_ids'] or seen['output_ids'][-1] not in seen['eos_ids'])),
                        decode_trace=seen['trace'] if forced else None, wall_seconds=time.monotonic()-started)
                    if forced:
                        if rec['output_ids'] != [forced_id]*33: raise ValueError('fixed generation prefix mismatch')
                        validate_decode_trace(rec['decode_trace'],len(rec['retained_ids']),rec['rope_delta'][0][0])
                append(journal, rec)
                group.append(rec)
            finally:
                if handler is not None: handler.remove()
                w.model.generate = original_generate
        validate_quartet(group)
        append(a.out/'validated_quartets.jsonl', {'task':row['task'], 'qid':row['qid'], 'group_id':row['group_id'],'forced':forced})
    for path, before in weight_stats.items():
        st=Path(path).stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] != before:
            raise ValueError('checkpoint weight stat changed during run: '+path)
    (a.out/'worker_complete.json').write_text(json.dumps({'n':len(records), 'end_unix':time.time()})+'\n')


if __name__ == '__main__': main()
