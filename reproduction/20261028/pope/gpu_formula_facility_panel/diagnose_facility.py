#!/usr/bin/env python3
"""Prepared 16-image diagnostic. Default is CPU-only validation; no automatic launch."""
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
POPE = HERE.parent
PLAN = HERE / 'run_plan.json'


def sha(path):
    with Path(path).open('rb') as f:
        value = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stat_guard(path):
    path = Path(path)
    s = path.stat()
    return dict(resolved=str(path.resolve()), bytes=s.st_size, device=s.st_dev, inode=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def validate(plan):
    assert plan['execution_status'] == 'prepared_only_not_allocated'
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256']:
        for path, expected in plan[group].items():
            assert sha(path) == expected, path
    for path, expected in plan['model_stat_guards'].items():
        assert stat_guard(path) == {k: v for k, v in expected.items() if k != 'prior_verified_sha256'}, path
    for row in plan['panel']:
        assert sha(row['image_path']) == row['image_sha256']
    assert len(plan['panel']) == 16 and len({x['image'] for x in plan['panel']}) == 16
    from formula_helpers import source_parts
    assert source_parts(Path(plan['arch_path']))[-1] == plan['source_ast_proof']
    canonical = {r['question_id']: r for r in map(json.loads, Path(plan['question_file']).read_text().splitlines())}
    for row in plan['panel']:
        assert canonical[row['question_id']]['image'] == row['image']
        assert canonical[row['question_id']]['text'] == row['prompt']
    assert 'torch' not in sys.modules, 'Validation must not import the model/tensor runtime'


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True)
    return sorted({int(x.strip()) for x in result.stdout.splitlines() if x.strip()})


def execute(plan, note_path):
    # This entry has not been invoked during preparation. A future root allocation
    # must name every other actual GPU PID; a spare slot is never inferred silently.
    note = json.loads(note_path.read_text())
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['run_plan_sha256'] == sha(PLAN)
    assert note['run_id'] == plan['run_id'] and note['GPU_slot'] in {'A', 'B'}
    assert gpu_pids() == sorted(note['allowed_other_gpu_pids']) and len(gpu_pids()) < 2
    assert Path(sys.executable).resolve() == Path(plan['python']).resolve()
    for prefix in plan['cleared_environment_prefixes']:
        for key in list(os.environ):
            if key.startswith(prefix):
                os.environ.pop(key)
    os.environ.update(plan['environment_overrides'])
    output = HERE / 'execution'
    output.mkdir(exist_ok=False)  # Single attempt; any partial result is preserved.
    (output / 'allocation_note.json').write_text(note_path.read_text())
    identity = {'pid': os.getpid(), 'start_ticks': int(Path('/proc/self/stat').read_text().split()[21]),
                'session_id': os.getsid(0), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan_sha256': sha(PLAN), 'allocation_note_sha256': sha(note_path)}
    (output / 'process_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    try:
        gpu_diagnostic(plan, output, identity)
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps({'error': repr(error), 'automatic_retry': False}, indent=2) + '\n')
        raise


def gpu_diagnostic(plan, output, identity):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    assert torch.__version__ == plan['runtime_versions']['torch']
    assert __import__('transformers').__version__ == plan['runtime_versions']['transformers']
    sys.path.insert(0, plan['llava_root'])
    sys.path.insert(0, str(POPE))
    from wait_only_patch import install
    from formula_helpers import bind_source, importance, traced_selection
    patch = install()
    (output / 'wait_patch_proof.json').write_text(json.dumps(patch, indent=2) + '\n')
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    tokenizer, model, processor, context = load_pretrained_model(
        plan['model_path'], None, get_model_name_from_path(plan['model_path']), visual_token_num=32, alpha=1., beta=1.)
    assert model.visual_token_num == 32 and model.alpha == 1. and model.beta == 1.
    tower = model.get_vision_tower()
    dtype = lambda module: sorted({str(p.dtype) for p in module.parameters()})
    runtime = {'identity': identity, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
               'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name(),
               'attention_backend': getattr(model.config, '_attn_implementation', None),
               'dtype': {n: dtype(m) for n, m in {'vision': tower.vision_tower, 'text': tower.text_tower,
                        'projection': tower.vision_tower.visual_projection, 'mm_projector': model.get_model().mm_projector}.items()},
               'TF32_matmul': torch.backends.cuda.matmul.allow_tf32,
               'TF32_cudnn': torch.backends.cudnn.allow_tf32, 'default_dtype': str(torch.get_default_dtype()),
               'autocast_enabled': torch.is_autocast_enabled(),
               'source_paths': {n: sys.modules[n].__file__ for n in ['llava.model.builder', 'llava.model.llava_arch',
                                'llava.model.multimodal_encoder.clip_encoder']},
               'context_len_unused': context, 'language_model_generation_calls': 0}
    assert runtime['attention_backend'] == 'sdpa'
    assert runtime['default_dtype'] == 'torch.float32' and not runtime['autocast_enabled']
    assert runtime['dtype'] == {'vision': ['torch.float32'], 'text': ['torch.float32'],
                                'projection': ['torch.float32'], 'mm_projector': ['torch.float16']}
    for path in runtime['source_paths'].values():
        assert sha(path) == plan['source_sha256'][path]
    (output / 'loaded_runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    helper, source_program, eq10, traced, proof = bind_source(torch, Path(plan['arch_path']))
    capture = {}
    native_cross, native_greedy = model.sim_cross, model.greed_select
    def observed_cross(*args, **kwargs):
        result = native_cross(*args, **kwargs)
        capture['global'], capture['dense'] = result
        return result
    def observed_greedy(values, sim, *args, **kwargs):
        result = native_greedy(values, sim, *args, **kwargs)
        capture.update(importance=values, sim=sim, indices=result[0], quotas=result[1])
        return result
    model.sim_cross, model.greed_select = observed_cross, observed_greedy
    rows = []
    with torch.inference_mode(), (output / 'per_image.jsonl').open('x') as stream:
        for index, panel in enumerate(plan['panel']):
            capture.clear()
            image = Image.open(panel['image_path']).convert('RGB')
            pixels = process_images([image], processor, model.config)
            assert isinstance(pixels, torch.Tensor) and tuple(pixels.shape) == (1, 3, 336, 336)
            inputs = pixels.to(dtype=torch.float16, device='cuda', non_blocking=True)
            guidance = panel['prompt'].replace('\nAnswer the question using a single word or phrase.', '')
            projected, native_mask = model.encode_images(inputs, texts=guidance)
            assert projected.dtype == torch.float16 and capture['global'].dtype == torch.float32
            assert capture['dense'].dtype == torch.float32 and capture['sim'].dtype == torch.float32
            assert tuple(projected.shape) == (1, 576, 4096) and int(native_mask.sum()) == 32
            for tensor in [projected, capture['global'], capture['dense'], capture['sim'], capture['importance']]:
                assert bool(torch.isfinite(tensor).all()), 'Native nonfinite tensor is a terminal diagnostic failure'
            arrays = {'projected_fp16': projected.cpu().numpy(), 'visual_sim_fp32': capture['sim'].cpu().numpy(),
                      'global_fp32': capture['global'].cpu().numpy(), 'dense_fp32': capture['dense'].cpu().numpy(),
                      'native_importance': capture['importance'].cpu().numpy(),
                      'native_selected_ids': capture['indices'].cpu().numpy(), 'native_final_mask': native_mask.cpu().numpy(),
                      'input_fp16': inputs.cpu().numpy()}
            image_rows = []
            for recipe in plan['recipes']:
                a, b = recipe['alpha'], recipe['beta']
                baseline, _ = importance(torch, helper, source_program, eq10, capture['global'], capture['dense'], a, b, 'source')
                assert bool(torch.isfinite(baseline).all())
                base_ids, base_quota = helper.greed_select(baseline, capture['sim'], K=1, strategy='importance_based')
                if a == b == 1.:
                    assert torch.equal(baseline, capture['importance']) and torch.equal(base_ids, capture['indices'])
                    assert torch.equal(base_quota, capture['quotas'])
                    mask = torch.zeros_like(native_mask)
                    mask.scatter_(1, base_ids, True)
                    assert torch.equal(mask, native_mask)
                baseline_gains = None
                for variant in ['source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only']:
                    values, details = importance(torch, helper, source_program, eq10, capture['global'], capture['dense'], a, b, variant)
                    assert bool(torch.isfinite(values).all()), 'Nonfinite importance must not reach greedy argmax'
                    native_ids, quotas = helper.greed_select(values, capture['sim'], K=1, strategy='importance_based')
                    ids, trace_quota, gain_trace = traced_selection(torch, helper, traced, values, capture['sim'])
                    assert bool(torch.isfinite(gain_trace).all()), 'Nonfinite gains are a terminal diagnostic failure'
                    assert torch.equal(ids, native_ids) and torch.equal(quotas, trace_quota)
                    if variant == 'source':
                        baseline_gains = gain_trace
                    if variant == 'source_members_order_identity':
                        assert torch.equal(values, baseline) and torch.equal(ids, base_ids)
                        assert torch.equal(gain_trace, baseline_gains)
                    if variant == 'eq6_quantile_membership_only' and a == 1.:
                        assert torch.equal(values, baseline) and torch.equal(ids, base_ids)
                    best_two = gain_trace.topk(2, dim=-1).values
                    margins = (best_two[..., 0] - best_two[..., 1]).flatten()
                    selected = ids[0].tolist()
                    base_selected = base_ids[0].tolist()
                    first = next((i for i, (x, y) in enumerate(zip(selected, base_selected)) if x != y), None)
                    divergence = None
                    if first is not None:
                        original_pick, changed_pick = base_selected[first], selected[first]
                        divergence = {'baseline_pick': original_pick, 'variant_pick': changed_pick,
                                      'same_greedy_prefix_before_step': selected[:first] == base_selected[:first],
                                      'baseline_gains_for_both_picks': [float(baseline_gains[first, 0, k]) for k in [original_pick, changed_pick]],
                                      'variant_gains_for_both_picks': [float(gain_trace[first, 0, k]) for k in [original_pick, changed_pick]],
                                      'step_gain_max_abs_delta': float((gain_trace[first] - baseline_gains[first]).abs().max())}
                    def objective(chosen, weights):
                        coverage = torch.maximum(capture['sim'][:, chosen, :].max(dim=1).values,
                                                 torch.zeros_like(weights))
                        return float((coverage * weights).sum())
                    row = {'recipe': recipe, 'variant': variant, 'importance_exact_equal': torch.equal(values, baseline),
                           'importance_max_abs_delta': float((values - baseline).abs().max()),
                           'selected_ids_ordered': selected, 'baseline_ids_ordered': base_selected,
                           'final_retained_ids_spatial_order': sorted(selected),
                           'selected_set_overlap': len(set(selected) & set(base_selected)),
                           'ordered_sequence_equal': selected == base_selected, 'first_divergence_step_zero_based': first,
                           'first_divergence_comparison': divergence,
                           'baseline_objective_baseline_selection': objective(base_selected, baseline),
                           'baseline_objective_variant_selection': objective(selected, baseline),
                           'variant_objective_baseline_selection': objective(base_selected, values),
                           'variant_objective_variant_selection': objective(selected, values),
                           'quota': quotas.tolist(), 'greedy_step_top2_gain_margins': margins.tolist(),
                           'exact_gain_ties_per_step': (gain_trace == gain_trace.max(dim=-1, keepdim=True).values).sum(-1).flatten().tolist(),
                           'text_filter': details, 'traced_greedy_matches_unmodified': True}
                    image_rows.append(row)
                    key = recipe['name'] + '__' + variant
                    arrays[key + '__importance'] = values.cpu().numpy()
                    arrays[key + '__ids'] = ids.cpu().numpy()
                    arrays[key + '__gain_trace'] = gain_trace.cpu().numpy()
                    final_mask = torch.zeros_like(native_mask)
                    final_mask.scatter_(1, ids, True)
                    arrays[key + '__final_mask'] = final_mask.cpu().numpy()
                repeat, _ = importance(torch, helper, source_program, eq10, capture['global'], capture['dense'], a, b, 'source')
                repeat_ids, repeat_quota = helper.greed_select(repeat, capture['sim'], K=1, strategy='importance_based')
                assert torch.equal(baseline, repeat) and torch.equal(base_ids, repeat_ids) and torch.equal(base_quota, repeat_quota)
            artifact = output / ('image_%02d.npz' % index)
            np.savez_compressed(artifact, **arrays)
            record = {'index': index, 'panel': panel, 'native_source_anchor_matches': True,
                      'native_feature_extractions': 1, 'cache_path': str(artifact), 'cache_sha256': sha(artifact),
                      'cache_bytes': artifact.stat().st_size, 'cases': image_rows}
            rows.append(record)
            stream.write(json.dumps(record) + '\n'); stream.flush()
            print(json.dumps({'completed_images': len(rows), 'maximum_images': 16}), flush=True)
    del model.sim_cross, model.greed_select
    assert len(rows) == 16
    # Frozen source/input guards close the run; validation itself imports no models.
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256']:
        assert all(sha(path) == value for path, value in plan[group].items())
    for path, expected in plan['model_stat_guards'].items():
        assert stat_guard(path) == {k: v for k, v in expected.items() if k != 'prior_verified_sha256'}, path
    assert all(sha(panel['image_path']) == panel['image_sha256'] for panel in plan['panel'])
    aggregate = []
    for recipe in plan['recipes']:
        for variant in plan['variants']:
            values = [case for record in rows for case in record['cases']
                      if case['recipe']['name'] == recipe['name'] and case['variant'] == variant]
            assert len(values) == 16
            aggregate.append({'recipe': recipe['name'], 'variant': variant,
                              'importance_changed_images': sum(not x['importance_exact_equal'] for x in values),
                              'ordered_sequence_changed_images': sum(not x['ordered_sequence_equal'] for x in values),
                              'retained_set_changed_images': sum(x['selected_set_overlap'] != len(x['baseline_ids_ordered']) for x in values),
                              'minimum_retained_set_overlap': min(x['selected_set_overlap'] for x in values),
                              'max_importance_abs_delta': max(x['importance_max_abs_delta'] for x in values)})
    summary = {'status': 'complete', 'n_images': 16, 'feature_extractions': 16, 'cases': 192,
               'benchmark_scores_computed': False, 'generated_answers': 0, 'plan_sha256': sha(PLAN),
               'per_image_sha256': sha(output / 'per_image.jsonl'), 'loaded_runtime_sha256': sha(output / 'loaded_runtime.json'),
               'source_ast_proof': proof, 'scope': plan['claim_boundary'], 'aggregate': aggregate,
               'all_numeric_tensors_finite': True, 'terminal_model_stats_and_all16_image_sha_match': True,
               'finished_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allocation-note', type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    validate(plan)
    if not args.execute:
        print(json.dumps({'status': 'prepared_only_not_allocated', 'CPU_validation': 'passed',
                          'torch_imported': False, 'run_plan_sha256': sha(PLAN), 'GPU_started': False}))
        return
    assert args.allocation_note is not None, 'A separate explicit root GPU allocation is required'
    execute(plan, args.allocation_note)


if __name__ == '__main__':
    main()
