#!/usr/bin/env python3
"""Prepared native16 AnyRes mediation diagnostic; default is CPU-only validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = HERE / 'run_plan.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stat_guard(path):
    path = Path(path); s = path.stat()
    return dict(resolved=str(path.resolve()), bytes=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def frozen_guards(plan):
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256']:
        for path, expected in plan[group].items():
            assert sha(path) == expected, path
    for path, expected in plan['model_stat_guards'].items():
        assert stat_guard(path) == {k: v for k, v in expected.items() if k != 'prior_verified_sha256'}, path
    assert all(sha(row['image_path']) == row['image_sha256'] for row in plan['panel'])


def validate(plan):
    frozen_guards(plan)
    assert len(plan['panel']) == 16 and len({row['image'] for row in plan['panel']}) == 16
    assert len(plan['recipes']) == 5 and plan['planned_case_slots'] == 320
    assert {(r['K'], r['beta']) for r in plan['recipes']} == {(128, 1.), (32, 1.), (128, 2.), (64, 2.), (32, 2.)}
    assert all(row['alpha'] == .5 for row in plan['recipes'])
    from anyres_helpers import common
    assert common.source_parts(Path(plan['arch_path']))[-1] == plan['source_ast_proof']
    canonical = [json.loads(line) for line in Path(plan['question_file']).read_text().splitlines()]
    for row in plan['panel']:
        q = canonical[row['full_question_position']]
        assert q['question_id'] == row['question_id'] and q['image'] == row['image'] and q['text'] == row['prompt']
    assert 'torch' not in sys.modules


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True)
    return sorted({int(x) for x in result.stdout.splitlines() if x.strip()})


def execute(plan, note_path):
    note = json.loads(note_path.read_text())
    assert note['authorization'] == 'explicit_root_gpu_allocation' and note['run_plan_sha256'] == sha(PLAN)
    assert note['run_id'] == plan['run_id'] and note['GPU_slot'] in {'A', 'B'}
    live = gpu_pids()
    assert live == sorted(note['allowed_other_gpu_pids']) and len(live) < 2
    assert Path(sys.executable).resolve() == Path(plan['python']).resolve()
    out = HERE / 'execution'; out.mkdir(exist_ok=False)
    identity = {'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().split()[21]),
                'session_id': os.getsid(0), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan_sha256': sha(PLAN), 'allocation_note_sha256': sha(note_path)}
    (out / 'process_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    (out / 'allocation_note.json').write_bytes(note_path.read_bytes())
    for prefix in plan['cleared_environment_prefixes']:
        for name in list(os.environ):
            if name.startswith(prefix): os.environ.pop(name)
    os.environ.update(plan['environment_overrides'])
    try:
        gpu_diagnostic(plan, out, identity)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps({'error': repr(error), 'automatic_retry': False}, indent=2) + '\n')
        raise


def gpu_diagnostic(plan, out, identity):
    import numpy as np
    import torch
    from PIL import Image
    from anyres_helpers import common, importance, traced_select, effective_mask, effective_ids, RaggedQuantile
    torch.set_num_threads(4)
    assert torch.__version__ == '2.1.2+cu118' and __import__('transformers').__version__ == '4.37.2'
    sys.path.insert(0, plan['llava_root'])
    sys.path.insert(0, plan['pope_root'])
    from wait_only_patch import install
    patch = install()
    (out / 'wait_patch_proof.json').write_text(json.dumps(patch, indent=2) + '\n')
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer, model, processor, context = load_pretrained_model(
        plan['model_path'], None, get_model_name_from_path(plan['model_path']), visual_token_num=128, alpha=.5, beta=1.)
    tower = model.get_vision_tower()
    dtype = lambda module: sorted({str(p.dtype) for p in module.parameters()})
    runtime = {'identity': identity, 'dtype': {name: dtype(module) for name, module in {
        'vision_body': tower.vision_tower.vision_model, 'visual_projection': tower.vision_tower.visual_projection,
        'text_tower': tower.text_tower, 'mm_projector': model.get_model().mm_projector}.items()},
        'backend': getattr(model.config, '_attn_implementation', None), 'default_dtype': str(torch.get_default_dtype()),
        'autocast_enabled': torch.is_autocast_enabled(), 'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_TF32': torch.backends.cudnn.allow_tf32, 'GPU': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
        'mm_patch_merge_type': model.config.mm_patch_merge_type, 'image_aspect_ratio': model.config.image_aspect_ratio,
        'image_grid_pinpoints': model.config.image_grid_pinpoints,
        'native_alpha': model.alpha, 'native_beta': model.beta, 'native_K': model.visual_token_num,
        'source_paths': {name: sys.modules[name].__file__ for name in ['llava.model.builder', 'llava.model.llava_arch',
                                                                  'llava.model.multimodal_encoder.clip_encoder']},
        'language_model_generation_calls': 0}
    assert runtime['dtype'] == plan['expected_dtype'] and runtime['backend'] == 'sdpa'
    assert runtime['default_dtype'] == 'torch.float32' and not runtime['autocast_enabled']
    assert not runtime['matmul_TF32'] and runtime['image_aspect_ratio'] == 'anyres'
    assert all(sha(path) == plan['source_sha256'][path] for path in runtime['source_paths'].values())
    (out / 'loaded_runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    helper, source, eq10, traced, proof = common.bind_source(torch, Path(plan['arch_path']))
    capture = {}
    native_cross, native_greedy = model.sim_cross, model.greed_select
    def cross(*args, **kwargs):
        value = native_cross(*args, **kwargs); capture['global'], capture['dense'] = value
        return value
    def greedy(weights, sim, *args, **kwargs):
        value = native_greedy(weights, sim, *args, **kwargs)
        capture.update(importance=weights, sim=sim, ids=value[0], quotas=value[1]); return value
    model.sim_cross, model.greed_select = cross, greedy
    records = []; measured = 0; skipped = 0
    with torch.inference_mode(), (out / 'per_image.jsonl').open('x') as stream:
        for index, panel in enumerate(plan['panel']):
            capture.clear()
            image = Image.open(panel['image_path']).convert('RGB')
            original_pixels = process_images([image], processor, model.config)
            assert isinstance(original_pixels, torch.Tensor) and tuple(original_pixels.shape) == (1, 5, 3, 336, 336)
            original_pixels = original_pixels.to(device='cuda', dtype=torch.float16, non_blocking=True)
            # This is the unchanged native ndim5 branch's flattening operation.
            split_sizes = [item.shape[0] for item in original_pixels]
            images = torch.cat([item for item in original_pixels], dim=0)
            guidance = panel['prompt'].replace('\nAnswer the question using a single word or phrase.', '')
            projected, native_mask = model.encode_images(images, texts=guidance, split_sizes=split_sizes)
            assert tuple(projected.shape) == (5, 576, 4096) and projected.dtype == torch.float16
            assert capture['global'].shape[:2] == (5, 576) and capture['dense'].shape[:2] == (5, 576)
            assert capture['global'].dtype == capture['dense'].dtype == capture['sim'].dtype == torch.float32
            for tensor in [projected, capture['global'], capture['dense'], capture['sim'], capture['importance']]:
                assert bool(torch.isfinite(tensor).all())
            cache = {name: value.cpu().numpy() for name, value in {
                'projected_fp16': projected, 'visual_sim_fp32': capture['sim'], 'global_fp32': capture['global'],
                'dense_fp32': capture['dense'], 'native_importance': capture['importance'],
                'native_internal_ids': capture['ids'], 'native_quotas': capture['quotas'],
                'native_final_mask': native_mask, 'actual_input_fp16': original_pixels}.items()}
            cases = []
            for recipe in plan['recipes']:
                helper.visual_token_num = recipe['K']
                base, _ = importance(torch, helper, source, eq10, capture['global'], capture['dense'], recipe['beta'], 'source')
                base_ids, base_quota = helper.greed_select(base, capture['sim'], K=5, strategy='importance_based')
                base_mask = effective_mask(torch, base_ids, base_quota)
                if recipe['K'] == 128 and recipe['beta'] == 1.:
                    assert torch.equal(base, capture['importance']) and torch.equal(base_ids, capture['ids'])
                    assert torch.equal(base_quota, capture['quotas']) and torch.equal(base_mask, native_mask)
                base_prefix, base_spatial = effective_ids(base_mask, base_ids, base_quota)
                baseline_gains = None
                for variant in plan['variants']:
                    try:
                        weights, text_details = importance(torch, helper, source, eq10, capture['global'], capture['dense'], recipe['beta'], variant)
                    except RaggedQuantile as error:
                        assert variant == 'eq6_quantile_membership_only'
                        cases.append({'recipe': recipe, 'variant': variant, 'status': 'unmeasured_ragged_quantile',
                                      'text_filter': error.details, 'no_implicit_padding_or_reduction_change': True})
                        skipped += 1; continue
                    assert bool(torch.isfinite(weights).all())
                    ids, quotas = helper.greed_select(weights, capture['sim'], K=5, strategy='importance_based')
                    seen_ids, seen_quota, gains = traced_select(torch, helper, traced, weights, capture['sim'])
                    assert torch.equal(ids, seen_ids) and torch.equal(quotas, seen_quota) and bool(torch.isfinite(gains).all())
                    mask = effective_mask(torch, ids, quotas)
                    prefixes, spatial = effective_ids(mask, ids, quotas)
                    if variant == 'source': baseline_gains = gains
                    if variant == 'source_members_order_identity':
                        assert torch.equal(weights, base) and torch.equal(gains, baseline_gains)
                        assert torch.equal(ids, base_ids) and torch.equal(quotas, base_quota) and torch.equal(mask, base_mask)
                    divergences = []
                    for crop in range(5):
                        first = next((t for t, (x, y) in enumerate(zip(prefixes[crop], base_prefix[crop])) if x != y), None)
                        if first is not None:
                            picks = [base_prefix[crop][first], prefixes[crop][first]]
                            divergences.append({'crop': crop, 'first_effective_step_zero_based': first,
                                'same_prefix_before_step': prefixes[crop][:first] == base_prefix[crop][:first],
                                'picks_baseline_variant': picks,
                                'baseline_gains_for_picks': [float(baseline_gains[first, crop, k]) for k in picks],
                                'variant_gains_for_picks': [float(gains[first, crop, k]) for k in picks]})
                    cases.append({'recipe': recipe, 'variant': variant, 'status': 'measured',
                        'importance_exact_equal': torch.equal(weights, base), 'importance_max_abs_delta': float((weights - base).abs().max()),
                        'quota': quotas.tolist(), 'baseline_quota': base_quota.tolist(), 'quota_exact_equal': torch.equal(quotas, base_quota),
                        'actual_retained_count': int(mask.sum()), 'baseline_retained_count': int(base_mask.sum()),
                        'effective_prefix_ids': prefixes, 'baseline_effective_prefix_ids': base_prefix,
                        'effective_prefix_order_equal': prefixes == base_prefix, 'retained_ids_spatial_order_by_crop': spatial,
                        'baseline_retained_ids_spatial_order_by_crop': base_spatial, 'final_mask_exact_equal': torch.equal(mask, base_mask),
                        'retained_overlap_by_crop': [len(set(spatial[b]) & set(base_spatial[b])) for b in range(5)],
                        'retained_added_by_crop': [sorted(set(spatial[b]) - set(base_spatial[b])) for b in range(5)],
                        'retained_removed_by_crop': [sorted(set(base_spatial[b]) - set(spatial[b])) for b in range(5)],
                        'effective_length_changes_by_crop': [{'crop': b, 'baseline_quota': len(base_prefix[b]),
                            'variant_quota': len(prefixes[b])} for b in range(5) if len(base_prefix[b]) != len(prefixes[b])],
                        'internal_maxT_sequence_equal': torch.equal(ids, base_ids),
                        'internal_discarded_changes_are_not_model_input_changes': True,
                        'first_effective_divergences': divergences,
                        'text_filter': text_details, 'traced_greedy_matches_unmodified': True})
                    name = recipe['name'] + '__' + variant
                    for suffix, value in [('importance', weights), ('internal_ids', ids), ('quotas', quotas), ('final_mask', mask), ('gain_trace', gains)]:
                        cache[name + '__' + suffix] = value.cpu().numpy()
                    measured += 1
                repeat, _ = importance(torch, helper, source, eq10, capture['global'], capture['dense'], recipe['beta'], 'source')
                repeat_ids, repeat_quota = helper.greed_select(repeat, capture['sim'], K=5, strategy='importance_based')
                assert torch.equal(repeat, base) and torch.equal(repeat_ids, base_ids) and torch.equal(repeat_quota, base_quota)
            artifact = out / ('image_%02d.npz' % index); np.savez_compressed(artifact, **cache)
            record = {'index': index, 'panel': panel, 'native_source_anchor_matches': True,
                      'native_feature_extractions': 1, 'split_sizes': split_sizes,
                      'cross_crop_normalization_shape': [1, 2880], 'actual_dense_shape': list(capture['dense'].shape),
                      'cache_path': str(artifact), 'cache_sha256': sha(artifact), 'cache_bytes': artifact.stat().st_size, 'cases': cases}
            records.append(record); stream.write(json.dumps(record) + '\n'); stream.flush()
            print(json.dumps({'completed_images': len(records), 'maximum_images': 16, 'measured_cases': measured, 'unmeasured_ragged_cases': skipped}), flush=True)
    del model.sim_cross, model.greed_select
    frozen_guards(plan)
    assert measured + skipped == 320 and len(records) == 16
    aggregate = []
    for recipe in plan['recipes']:
        for variant in plan['variants']:
            values = [case for row in records for case in row['cases'] if case['recipe'] == recipe and case['variant'] == variant]
            valid = [x for x in values if x['status'] == 'measured']
            aggregate.append({'recipe': recipe['name'], 'variant': variant, 'case_slots': len(values),
                'measured': len(valid), 'unmeasured_ragged': len(values) - len(valid),
                'importance_changed_images': sum(not x['importance_exact_equal'] for x in valid),
                'quota_changed_images': sum(not x['quota_exact_equal'] for x in valid),
                'effective_prefix_changed_images': sum(not x['effective_prefix_order_equal'] for x in valid),
                'actual_final_mask_changed_images': sum(not x['final_mask_exact_equal'] for x in valid),
                'internal_sequence_changed_images': sum(not x['internal_maxT_sequence_equal'] for x in valid)})
    summary = {'status': 'complete' if skipped == 0 else 'complete_with_unmeasured_ragged_eq6',
        'plan_sha256': sha(PLAN), 'n_images': 16, 'feature_extractions': 16, 'planned_case_slots': 320,
        'measured_cases': measured, 'unmeasured_ragged_cases': skipped, 'source_anchor_images': 16,
        'identity_controls': 80, 'aggregate': aggregate, 'all_numeric_tensors_finite': True,
        'terminal_source_model_image_guards_passed': True, 'generated_answers': 0, 'GT_read': False,
        'benchmark_scores_computed': False, 'source_ast_proof': proof,
        'per_image_sha256': sha(out / 'per_image.jsonl'), 'loaded_runtime_sha256': sha(out / 'loaded_runtime.json'),
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'claim_boundary': plan['claim_boundary']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--allocation-note', type=Path)
    args = parser.parse_args(); plan = json.loads(PLAN.read_text()); validate(plan)
    if not args.execute:
        print(json.dumps({'status': 'prepared_only_not_allocated', 'CPU_validation': 'passed',
                          'torch_imported': False, 'GPU_started': False, 'run_plan_sha256': sha(PLAN)})); return
    assert args.allocation_note is not None
    execute(plan, args.allocation_note)


if __name__ == '__main__':
    main()
