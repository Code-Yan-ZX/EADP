"""E1 full DEV natural-generation runner; only after both smoke verifiers pass.

Importing or executing without --execute never imports torch or loads a model.
This generator does not score predictions. See README for prerequisites.
"""
import argparse
import contextlib
import fcntl
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


def validate_legacy_gate():
    base=Path(__file__).resolve().parent
    proof_path=base/'legacy004_verification.json'
    review_path=PROJECT/'e1_independent_review/legacy004_review.json'
    proof=json.loads(proof_path.read_text()); review=json.loads(review_path.read_text())
    if proof.get('status')!='passed' or proof.get('n_verified')!=16 or proof.get('errors'):
        raise ValueError('legacy 16-question gate failed')
    if review.get('status')!='passed' or review.get('n_expected')!=16 or review.get('errors') or review.get('run_proof_sha256')!=sha(proof_path):
        raise ValueError('legacy independent review absent or mismatched')
    if review.get('reviewer_sha256')!=sha(PROJECT/'e1_independent_review/review.py'):
        raise ValueError('legacy independent reviewer source changed')
    for path in (base/'monitorverifylegacy004/terminal.json',PROJECT/'e1_independent_review/monitor_legacy004/terminal.json'):
        t=json.loads(path.read_text())
        if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group'):
            raise ValueError('legacy verifier/reviewer has not exited cleanly')
    if review.get('verifier_terminal_sha256')!=sha(base/'monitorverifylegacy004/terminal.json'):
        raise ValueError('legacy verifier terminal hash mismatch')
    run=base/'legacy004'; frozen=json.loads((run/'frozen.json').read_text())
    for key,path in (('comparisons_sha256',run/'comparisons.jsonl'),('frozen_sha256',run/'frozen.json'),
                     ('worker_complete_sha256',run/'worker_complete.json'),('terminal_sha256',base/'monitorlegacy004/terminal.json'),
                     ('container_exit_sha256',base/'legacy004_exit.json'),('launch_sha256',base/'legacy004_launch.json')):
        if proof.get(key)!=sha(path): raise ValueError('legacy evidence changed: '+key)
    sources=base/'legacy004_sources'
    if proof.get('verifier_sha256')!=sha(sources/'verify_legacy.py'):
        raise ValueError('legacy verifier snapshot changed')
    for key,name in (('position_adapter_sha256','position_adapter.py'),('worker_sha256','legacy_anchor.py'),
                     ('runner_sha256','runner.py'),('contract_sha256','contract.py')):
        if frozen.get(key)!=sha(sources/name): raise ValueError('legacy source snapshot changed: '+name)
    if sha(Path(__file__).with_name('position_adapter.py'))!=frozen['position_adapter_sha256']:
        raise ValueError('P1 adapter changed since legacy gate')
    observations=read_jsonl(run/'comparisons.jsonl')
    if len(observations)!=32: raise ValueError('legacy comparison count changed')
    for rec in observations:
        for label in ('legacy','p1'):
            item=rec[label]; path=Path(item['array_path'])
            if path.parent!=run or sha(path)!=item['array_sha256']:
                raise ValueError('legacy diagnostic array changed')
    return {'proof_sha256':sha(proof_path),'review_sha256':sha(review_path),
            'review_terminal_sha256':sha(PROJECT/'e1_independent_review/monitor_legacy004/terminal.json')}


def validate_smoke_gate():
    base=Path(__file__).resolve().parent
    proof_path=base/'e1smoke001_verification.json'
    review_path=PROJECT/'e1_independent_review/e1smoke001_review.json'
    proof=json.loads(proof_path.read_text()); review=json.loads(review_path.read_text())
    if proof.get('status')!='passed' or proof.get('n_verified')!=32 or proof.get('errors'):
        raise ValueError('smoke verification not passed')
    if review.get('status')!='passed' or review.get('n_verified')!=32 or review.get('errors'):
        raise ValueError('smoke independent review not passed')
    if review.get('reviewer_sha256')!=sha(base/'e1smoke001_sources/review_smoke.py'):
        raise ValueError('smoke reviewer source snapshot changed')
    if review.get('run_proof_sha256')!=sha(proof_path):
        raise ValueError('smoke reviewer does not bind verifier proof')
    for path in (base/'monitorverifye1smoke001/terminal.json',PROJECT/'e1_independent_review/monitor_e1smoke001/terminal.json'):
        t=json.loads(path.read_text())
        if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group'):
            raise ValueError('smoke verifier/reviewer still running or failed')
    if review.get('verifier_terminal_sha256')!=sha(base/'monitorverifye1smoke001/terminal.json'):
        raise ValueError('smoke reviewer verifier terminal mismatch')
    run=base/'e1smoke001'
    for key,path in (('frozen_sha256',run/'frozen.json'),('raw_predictions_sha256',run/'raw_predictions.jsonl'),
                     ('generated_arms_sha256',run/'generated_arms.jsonl'),('validated_quartets_sha256',run/'validated_quartets.jsonl'),
                     ('worker_complete_sha256',run/'worker_complete.json'),('launch_sha256',base/'e1smoke001_launch.json'),
                     ('container_exit_sha256',base/'e1smoke001_exit.json'),('supervisor_terminal_sha256',base/'monitore1smoke001/terminal.json')):
        if proof.get(key)!=sha(path): raise ValueError('smoke evidence changed: '+key)
    source=base/'e1smoke001_sources'; launch=json.loads((base/'e1smoke001_launch.json').read_text())
    for path,digest in launch['source_sha256'].items():
        if sha(source/Path(path).name)!=digest: raise ValueError('smoke source snapshot changed')
    if proof.get('verifier_sha256')!=sha(source/'verify_smoke.py'): raise ValueError('smoke verifier snapshot changed')
    if sha(Path(__file__).with_name('position_adapter.py'))!=sha(source/'position_adapter.py'):
        raise ValueError('position adapter differs from validated smoke')
    return {'proof_sha256':sha(proof_path),'review_sha256':sha(review_path),
            'verifier_terminal_sha256':sha(base/'monitorverifye1smoke001/terminal.json'),
            'review_terminal_sha256':sha(PROJECT/'e1_independent_review/monitor_e1smoke001/terminal.json')}


def validate_prefix(panel, records, generated, quartets):
    if len(records)!=len(generated) or len(records)>4*len(panel):
        raise ValueError('partial/unpaired generation journal; preserve and inspect before resume')
    for index,rec in enumerate(records):
        row=panel[index//4]
        if any(rec.get(k)!=row[k] for k in ('task','qid','group_id','message_sha256')) or rec.get('arm')!=ARMS[index%4] or rec.get('forced') is not False:
            raise ValueError('journal is not the frozen natural-arm prefix')
        if generated[index]!={k:rec[k] for k in ('task','qid','group_id','arm','forced','output_ids','eos_ids')}:
            raise ValueError('generation and observation journals differ')
    if len(quartets)!=len(records)//4:
        raise ValueError('quartet journal incomplete; preserve and inspect before resume')
    for i,q in enumerate(quartets):
        row=panel[i]
        if q!={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'forced':False}:
            raise ValueError('quartet identity mismatch')
        validate_quartet(records[4*i:4*i+4])
    return len(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--e0-proof', type=Path, required=True)
    p.add_argument('--manifest-dir', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--input-proof', type=Path, required=True)
    p.add_argument('--stage', choices=('full',), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.resume:
        raise ValueError('resume disabled: preserve interrupted output and start a new run')
    if not a.execute:
        raise ValueError('execution disabled without --execute; use tests for CPU checks')
    e0 = validate_gate(a.e0_proof)
    records=verified_inputs(a.manifest_dir,a.inputs,a.input_proof)
    smoke=validate_smoke_gate()
    legacy=validate_legacy_gate()
    if len(records)!=384: raise ValueError('384 DEV questions required')
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
    if not a.resume: a.out.mkdir(parents=True, exist_ok=False)
    elif not a.out.is_dir(): raise ValueError('resume directory missing')
    lock=(a.out/'worker.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise ValueError('another worker already owns this output')
    frozen = {'e0_proof_sha256': sha(a.e0_proof), 'e0_adapter_sha256': e0['adapter_sha256'],
              'e0_verifier_terminal_sha256':sha(E0/'monitor_verify_array16/terminal.json'),
              'panel_sha256': sha(a.manifest_dir/'panel_manifest.jsonl'), 'inputs_sha256': sha(a.inputs),
              'anchor_exclusion':validate_anchor_exclusion(a.manifest_dir,OLD_PANEL),
              'legacy_gate':legacy,
              'smoke_gate':smoke,
              'input_proof_sha256': sha(a.input_proof), 'plan_sha256': sha(PLAN),
              'runner_sha256': sha(__file__), 'contract_sha256': sha(Path(__file__).with_name('contract.py')),
              'position_adapter_sha256':sha(Path(__file__).with_name('position_adapter.py')),
              'weights_identity_sha256':sha(weights_path), 'weight_stats_before':weight_stats,
              'arms': ARMS, 'n_questions': len(records), 'stage':a.stage, 'gt_read': False, 'scoring': False}
    identity_path=a.out/'frozen.json'
    if a.resume:
        if not identity_path.exists() or json.loads(identity_path.read_text())!=json.loads(json.dumps(frozen)):
            raise ValueError('resume identity differs from original run')
        if (a.out/'worker_complete.json').exists(): raise ValueError('run already completed')
    else:
        with identity_path.open('x') as f: json.dump(frozen,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    raw_path=a.out/'raw_predictions.jsonl'; gen_path=a.out/'generated_arms.jsonl'; quartet_path=a.out/'validated_quartets.jsonl'
    previous=read_jsonl(raw_path) if raw_path.exists() else []
    generated=read_jsonl(gen_path) if gen_path.exists() else []
    quartets=read_jsonl(quartet_path) if quartet_path.exists() else []
    prefix=validate_prefix(records,previous,generated,quartets)

    sys.path[:0] = [str(E0), str(OLD), plan['author_root'], str(Path(plan['author_root'])/'VLMEvalKit')]
    import torch
    import numpy as np
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

    journal = raw_path
    class ForceToken(LogitsProcessor):
        def __init__(self, token): self.token=token
        def __call__(self, input_ids, scores):
            scores.fill_(-float('inf')); scores[:,self.token]=0
            return scores
    forced_id=w.processor.tokenizer.encode(' the',add_special_tokens=False)[0]
    for row_index,row in enumerate(records):
        forced=False
        if (row_index+1)*4<=prefix: continue
        group=previous[row_index*4:min(prefix,(row_index+1)*4)]
        for arm_index,arm in enumerate(ARMS):
            if row_index*4+arm_index<prefix: continue
            seen = {'trace':[], 'logits':[]}
            def observe(_module, _args, kw):
                if 'prefill' not in seen:
                    def clone(x):
                        if x is None: return None
                        if isinstance(x,(list,tuple)): return [clone(z) for z in x]
                        return x.detach().clone()
                    seen['prefill'] = {k:clone(kw.get(k)) for k in ('inputs_embeds','attention_mask','position_ids','cache_position','visual_pos_masks','deepstack_visual_embeds')}
                    seen['delta'] = w.model.model.rope_deltas.detach().cpu().tolist()
                if forced:
                    cache=kw.get('past_key_values')
                    seen['trace'].append({'position_ids':kw['position_ids'].detach().cpu().tolist(),
                         'cache_position':kw['cache_position'].detach().cpu().tolist(),
                         'rope_delta':w.model.model.rope_deltas.detach().cpu().tolist(),
                         'layer_cache_lengths':[] if cache is None else [cache.get_seq_length(i) for i in range(len(w.model.model.language_model.layers))]})
            def observe_after(_module,_args,_kwargs,out):
                if forced:
                    cache=out.past_key_values
                    seen['trace'][-1]['layer_cache_lengths_after']=[] if cache is None else [cache.get_seq_length(i) for i in range(len(w.model.model.language_model.layers))]
            def observe_logits(_module,_args,out):
                if forced or not seen['logits']:
                    seen['logits'].append(out[:,-1,:].detach().contiguous().cpu())
            original_generate = w.model.generate
            def capture_generate(*args, **kw):
                if forced:
                    kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=LogitsProcessorList([ForceToken(forced_id)]))
                scalar_names=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')
                seen['generation_scalars']={key:kw.get(key) for key in scalar_names}
                expected={'do_sample':False,'max_new_tokens':33 if forced else 2048,'top_p':recipe['top_p'],
                          'top_k':recipe['top_k'],'temperature':recipe['temperature'],'repetition_penalty':recipe['repetition_penalty']}
                if seen['generation_scalars']!=expected: raise ValueError('actual generation recipe mismatch')
                eos=kw.get('eos_token_id',w.model.generation_config.eos_token_id)
                seen['eos_ids']=[eos] if isinstance(eos,int) else list(eos or [])
                generated = original_generate(*args, **kw)
                seen['output_ids'] = generated.detach().cpu().tolist()[0]
                append(a.out/'generated_arms.jsonl',{'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],
                       'arm':arm,'forced':forced,'output_ids':seen['output_ids'],'eos_ids':seen['eos_ids']})
                return generated
            w.model.generate = capture_generate
            observer = SelectionObserver(eadp_pruner)
            handlers = []
            started = time.monotonic()
            try:
                with observer, FourArm(torch, w, arm, observer) as ad:
                    handlers.append(w.model.model.language_model.register_forward_pre_hook(observe, with_kwargs=True))
                    handlers.append(w.model.model.language_model.register_forward_hook(observe_after, with_kwargs=True))
                    handlers.append(w.model.lm_head.register_forward_hook(observe_logits))
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
                        generation_scalars=seen['generation_scalars'], forced_token=forced_id if forced else None,
                        eos=bool(seen['output_ids'] and seen['output_ids'][-1] in seen['eos_ids']),
                        truncated=bool(not forced and len(seen['output_ids'])>=2048 and (not seen['output_ids'] or seen['output_ids'][-1] not in seen['eos_ids'])),
                        decode_trace=seen['trace'] if forced else None, wall_seconds=time.monotonic()-started)
                    rec['logits']=[meta(z) for z in seen['logits']]
                    rec['wall_seconds_through_observation']=time.monotonic()-started
                    if forced:
                        if rec['output_ids'] != [forced_id]*33: raise ValueError('fixed generation prefix mismatch')
                        validate_decode_trace(rec['decode_trace'],len(rec['retained_ids']),rec['rope_delta'][0][0])
                        for step,entry in enumerate(rec['decode_trace']):
                            if entry['rope_delta'] != rec['rope_delta']: raise ValueError('decode rope delta drift')
                            expected=len(rec['retained_ids'])+step
                            if len(entry['layer_cache_lengths_after'])!=36 or any(x!=expected for x in entry['layer_cache_lengths_after']):
                                raise ValueError('post-forward per-layer cache length mismatch')
                append(journal, rec)
                group.append(rec)
            finally:
                for handler in handlers: handler.remove()
                w.model.generate = original_generate
        validate_quartet(group)
        append(a.out/'validated_quartets.jsonl', {'task':row['task'], 'qid':row['qid'], 'group_id':row['group_id'],'forced':forced})
    for path, before in weight_stats.items():
        st=Path(path).stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] != before:
            raise ValueError('checkpoint weight stat changed during run: '+path)
    (a.out/'worker_complete.json').write_text(json.dumps({'n':len(records), 'end_unix':time.time()})+'\n')


if __name__ == '__main__': main()
