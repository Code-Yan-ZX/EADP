"""Bounded native Qwen feature/selection diagnostic. Default imports no torch."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PLAN = HERE / 'run_plan.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stat(path):
    p = Path(path); value = p.stat()
    return dict(resolved=str(p.resolve()), bytes=value.st_size, device=value.st_dev, inode=value.st_ino,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def validate(plan, namespace='host'):
    for group in ('source_sha256', 'implementation_sha256', 'reference_sha256'):
        for path, expected in plan[group].items():
            assert sha(path) == expected, path
    from namespace_guards import check_model_stats
    check_model_stats(plan, stat, namespace)
    assert plan['recipe']['visual_token_num'] == 256 and plan['recipe']['alpha'] == .5 and plan['recipe']['beta'] == 2.
    positions = [((2 * i + 1) * 5349) // 32 for i in range(16)]
    assert [item['position'] for item in plan['panel']] == positions
    assert len({item['index'] for item in plan['panel']}) == 16 and plan['planned_case_slots'] == 64
    assert plan['variants'] == ['source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only']
    for item in plan['panel']:
        assert set(item) == {'position', 'index', 'image_path', 'image_sha256', 'message', 'source_messages_sha256',
                             'processor_expected', 'question_sha256'}
        assert sha(item['image_path']) == item['image_sha256']
        assert [m['type'] for m in item['message']] == ['image', 'text']
        assert item['message'][0]['value'] == item['image_path']
    from formula_helpers import parts
    assert parts(plan['pruner_path'])[-1] == plan['source_AST_proof']
    assert 'torch' not in sys.modules, 'Default CPU checking must not import a tensor runtime'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def gpu_diagnostic(plan, note, note_sha):
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['run_plan_sha256'] == sha(PLAN) and note['run_id'] == plan['run_id']
    assert note['GPU_slot'] in ('A', 'B')
    assert Path(sys.executable).resolve() == Path(plan['python']).resolve()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, plan['author_root'])
    sys.path.insert(0, str(Path(plan['author_root']) / 'VLMEvalKit'))
    from vlmeval.config import supported_VLM
    from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
    import model.pruner as native
    from formula_helpers import bind, RaggedQuantile
    versions = {name: importlib.metadata.version(name) for name in plan['expected_versions']}
    assert versions == plan['expected_versions']
    factory = supported_VLM[plan['model_name']]
    assert factory.func is Qwen3VLChatEADP and factory.keywords == plan['recipe']
    compute, select, proof = bind(torch, native, plan['pruner_path'])
    assert proof == plan['source_AST_proof']
    out = HERE / 'execution'; out.mkdir(exist_ok=False)
    save(out / 'identity.json', dict(container_pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                                  plan_sha256=sha(PLAN), allocation=note, allocation_note_sha256=note_sha,
                                  started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        wrapper = factory()
        assert isinstance(wrapper, Qwen3VLChatEADP)
        assert wrapper.visual_token_num == wrapper.pruner.visual_token_num == 256
        assert wrapper.alpha == wrapper.pruner.alpha == .5 and wrapper.beta == wrapper.pruner.beta == 2.
        assert wrapper.model.config._attn_implementation == 'flash_attention_2'
        assert wrapper.model.visual.config._attn_implementation == 'flash_attention_2'
        assert wrapper.model.dtype == wrapper.model.visual.dtype == torch.bfloat16
        assert {str(p.device) for p in wrapper.model.parameters()} == {'cuda:0'}
        assert wrapper.pruner.spatial_merge_size == 2
        source_modules = {name: sys.modules[name].__file__ for name in ('vlmeval.config', 'model.pruner',
                         'vlmeval.vlm.qwen3_vl.model', 'vlmeval.vlm.qwen3_vl.model_fixed_res')}
        assert all(sha(path) == plan['source_sha256'][path] for path in source_modules.values())
        modeling = sys.modules['transformers.models.qwen3_vl.modeling_qwen3_vl'].__file__
        assert Path(modeling).resolve() == Path(plan['modeling_path']).resolve()
        assert sha(modeling) == plan['source_sha256'][plan['modeling_path']]
        runtime = dict(versions=versions, torch_cuda=torch.version.cuda, backend=wrapper.model.config._attn_implementation,
                       vision_backend=wrapper.model.visual.config._attn_implementation,
                       model_dtype=str(wrapper.model.dtype), visual_dtype=str(wrapper.model.visual.dtype),
                       input_embedding_dtype=str(wrapper.model.get_input_embeddings().weight.dtype),
                       default_dtype=str(torch.get_default_dtype()), autocast=torch.is_autocast_enabled(),
                       matmul_TF32=torch.backends.cuda.matmul.allow_tf32, source_modules=source_modules,
                       modeling_source=dict(path=modeling, sha256=sha(modeling)),
                       GPU=torch.cuda.get_device_name(), generate_calls=0,
                       deepstack='Unchanged native unwrap_visual_output; no DeepStack restoration or adapter')
        assert runtime['input_embedding_dtype'] == 'torch.bfloat16'
        assert runtime['default_dtype'] == 'torch.float32' and not runtime['autocast'] and not runtime['matmul_TF32']
        save(out / 'runtime.json', runtime)
        def forbidden(*args, **kwargs):
            raise RuntimeError('Answer generation is forbidden in this feature-only plan')
        wrapper.generate = forbidden; wrapper.model.generate = forbidden
        records, measured, unmeasured = [], 0, 0
        with torch.inference_mode(), (out / 'per_question.jsonl').open('x') as stream:
            for number, item in enumerate(plan['panel']):
                message = item['message']; messages = wrapper._build_messages(message, dataset='DocVQA_VAL')
                assert hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == item['source_messages_sha256']
                inputs = wrapper._processor_inputs(messages)
                expected = item['processor_expected']
                def tensor_meta(value):
                    actual = value.detach().cpu().contiguous()
                    array = actual.view(torch.uint16).numpy() if actual.dtype == torch.bfloat16 else actual.numpy()
                    return dict(dtype=str(actual.dtype), shape=list(actual.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())
                assert set(inputs) == {'pixel_values', 'input_ids', 'attention_mask', 'image_grid_thw'}
                assert {key: tensor_meta(value) for key, value in inputs.items()} == expected
                assert inputs['image_grid_thw'].tolist() == [[1, 64, 64]]
                capture, counters = {}, {}
                originals = {name: getattr(native, name) for name in ('_sim_cross', '_sim_visual_impl', '_greed_select_impl')}
                def observe(name):
                    def call(*args, **kwargs):
                        value = originals[name](*args, **kwargs)
                        capture[name] = value; counters[name] = counters.get(name, 0) + 1
                        return value
                    return call
                for name in originals: setattr(native, name, observe(name))
                old_compute = wrapper.pruner.compute_importance
                old_forward = wrapper.pruner.forward
                old_tokenize = wrapper._tokenize_instruction
                def observed_compute(*args, **kwargs):
                    value = old_compute(*args, **kwargs); capture['importance'] = value
                    counters['compute'] = counters.get('compute', 0) + 1
                    return value
                def observed_forward(*args, **kwargs):
                    value = old_forward(*args, **kwargs); capture['pruner_args'] = args
                    counters['pruner'] = counters.get('pruner', 0) + 1
                    return value
                def observed_tokenize(*args, **kwargs):
                    value = old_tokenize(*args, **kwargs); capture['instruction_ids'] = value
                    counters['tokenize'] = counters.get('tokenize', 0) + 1
                    return value
                wrapper.pruner.compute_importance = observed_compute; wrapper.pruner.forward = observed_forward
                wrapper._tokenize_instruction = observed_tokenize
                try:
                    sequence = wrapper._get_instruction_sequence_embedding(message, dataset='DocVQA_VAL')
                    pruned, splits = wrapper._get_pruned_image_features(inputs['pixel_values'], inputs['image_grid_thw'],
                        instruction_embeds_seq=sequence, message=message, dataset='DocVQA_VAL')
                finally:
                    for name, value in originals.items(): setattr(native, name, value)
                    wrapper.pruner.compute_importance = old_compute; wrapper.pruner.forward = old_forward
                    wrapper._tokenize_instruction = old_tokenize
                assert counters == {name: 1 for name in ('_sim_cross', '_sim_visual_impl', '_greed_select_impl', 'compute', 'pruner', 'tokenize')}
                features, global_text, dense_text, grid = capture['pruner_args']
                assert tuple(features.shape) == (1024, 4096) and features.dtype == torch.bfloat16
                assert sequence.dtype == global_text.dtype == dense_text.dtype == torch.bfloat16
                assert tuple(global_text.shape) == (1, 4096) and tuple(dense_text.shape) == tuple(sequence.shape)
                assert torch.equal(global_text, sequence.mean(dim=1)) and torch.equal(dense_text, sequence)
                global_sim, dense_sim = capture['_sim_cross']; sim = capture['_sim_visual_impl']
                base_ids, base_quota = capture['_greed_select_impl']; spatial = base_ids[0].sort().values
                assert global_sim.dtype == dense_sim.dtype == sim.dtype == capture['importance'].dtype == torch.float32
                assert splits == [256] and base_quota.tolist() == [256]
                assert base_ids.shape == (1, 256) and len(set(base_ids[0].tolist())) == 256
                assert torch.equal(pruned, features[spatial])
                assert torch.equal(originals['_sim_visual_impl'](features.unsqueeze(0)), sim)
                recomputed = originals['_sim_cross'](global_text, dense_text, features.unsqueeze(0))
                assert torch.equal(recomputed[0], global_sim) and torch.equal(recomputed[1], dense_sim)
                arrays, metadata = {}, {}
                def cache(name, value):
                    actual = value.detach().cpu().contiguous()
                    assert bool(torch.isfinite(actual).all())
                    arrays[name] = actual.view(torch.uint16).numpy() if actual.dtype == torch.bfloat16 else actual.numpy()
                    metadata[name] = tensor_meta(actual)
                for name, value in dict(visual_features_BF16_bits=features, text_sequence_BF16_bits=dense_text,
                    text_global_mean_BF16_bits=global_text, instruction_ids=capture['instruction_ids'],
                    global_sim=global_sim, dense_sim=dense_sim, visual_sim=sim, native_importance=capture['importance'],
                    native_ordered_ids=base_ids, native_spatial_ids=spatial, native_pruned_BF16_bits=pruned,
                    full_prompt_input_ids=inputs['input_ids'], image_grid_thw=grid).items(): cache(name, value)
                cases, source_gains = [], None
                for variant in plan['variants']:
                    try:
                        importance, details = compute(wrapper.pruner, features.unsqueeze(0), global_text, dense_text, 32, 32, variant)
                    except RaggedQuantile as error:
                        assert variant == 'eq6_quantile_membership_only'
                        cases.append(dict(variant=variant, status='unmeasured_ragged', text_filter=error.details)); unmeasured += 1
                        continue
                    ids, quota, gains = select(importance, sim, 256)
                    assert bool(torch.isfinite(importance).all()) and bool(torch.isfinite(gains).all())
                    assert quota.tolist() == [256] and len(set(ids[0].tolist())) == 256
                    sorted_ids = ids[0].sort().values
                    if variant == 'source':
                        assert torch.equal(importance, capture['importance']) and torch.equal(ids, base_ids) and torch.equal(quota, base_quota)
                        assert torch.equal(features[sorted_ids], pruned); source_gains = gains
                    if variant == 'source_members_order_identity':
                        assert torch.equal(importance, capture['importance']) and torch.equal(ids, base_ids)
                        assert torch.equal(gains, source_gains) and torch.equal(features[sorted_ids], pruned)
                    first = next((i for i, (a, b) in enumerate(zip(ids[0].tolist(), base_ids[0].tolist())) if a != b), None)
                    divergence = None
                    if first is not None:
                        picks = [int(base_ids[0, first]), int(ids[0, first])]
                        divergence = dict(step=first, same_prefix_before_step=torch.equal(ids[:, :first], base_ids[:, :first]),
                            baseline_variant_picks=picks, baseline_gains=[float(source_gains[first, 0, k]) for k in picks],
                            variant_gains=[float(gains[first, 0, k]) for k in picks])
                    cases.append(dict(variant=variant, status='measured', text_filter=details,
                        importance_exact_equal=torch.equal(importance, capture['importance']),
                        importance_max_abs_delta=float((importance - capture['importance']).abs().max()),
                        ordered_ids=ids[0].tolist(), spatial_ids=sorted_ids.tolist(), actual_retained=256,
                        ordered_ids_equal=torch.equal(ids, base_ids), final_spatial_ids_equal=torch.equal(sorted_ids, spatial),
                        final_spatial_features=tensor_meta(features[sorted_ids]), first_divergence=divergence,
                        added=sorted(set(sorted_ids.tolist()) - set(spatial.tolist())),
                        removed=sorted(set(spatial.tolist()) - set(sorted_ids.tolist())), traced_matches_native_greedy=True))
                    for suffix, value in [('importance', importance), ('ordered_ids', ids), ('spatial_ids', sorted_ids), ('gain_trace', gains)]:
                        cache(variant + '__' + suffix, value)
                    measured += 1
                artifact = out / ('question_%02d.npz' % number); np.savez_compressed(artifact, **arrays)
                record = dict(position=item['position'], index=item['index'], native_get_calls=1, native_pruner_anchor=True,
                    image_sha256=item['image_sha256'], actual_instruction_L=sequence.shape[1],
                    cache=dict(path=str(artifact), sha256=sha(artifact), bytes=artifact.stat().st_size), tensors=metadata, cases=cases)
                records.append(record); stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()
                print(json.dumps(dict(completed_questions=len(records), measured_cases=measured, unmeasured_cases=unmeasured)), flush=True)
        # Recheck sources/model inputs with the same no-runtime-import guard relaxed only here.
        for group in ('source_sha256', 'implementation_sha256', 'reference_sha256'):
            for path, expected_sha in plan[group].items(): assert sha(path) == expected_sha, path
        from namespace_guards import check_model_stats
        check_model_stats(plan, stat, 'container')
        for item in plan['panel']: assert sha(item['image_path']) == item['image_sha256']
        assert len(records) == 16 and measured + unmeasured == 64
        summary = dict(status='complete' if not unmeasured else 'complete_with_explicit_unmeasured',
            run_plan_sha256=sha(PLAN), n_questions=16, native_get_calls=16, native_anchors=16, identity_controls=16,
            planned_case_slots=64, measured_cases=measured, unmeasured_cases=unmeasured,
            source_and_input_guards_unchanged=True, generated_answers=0, benchmark_scores_computed=False, GT_read=False,
            per_question_sha256=sha(out / 'per_question.jsonl'), runtime_sha256=sha(out / 'runtime.json'),
            cache_bytes=sum(row['cache']['bytes'] for row in records), claim_boundary=plan['claim_boundary'],
            aggregate=[dict(variant=variant, measured=sum(c['status'] == 'measured' for r in records for c in r['cases'] if c['variant'] == variant),
                importance_changed=sum(not c['importance_exact_equal'] for r in records for c in r['cases'] if c['variant'] == variant and c['status'] == 'measured'),
                ordered_ids_changed=sum(not c['ordered_ids_equal'] for r in records for c in r['cases'] if c['variant'] == variant and c['status'] == 'measured'),
                final_spatial_ids_changed=sum(not c['final_spatial_ids_equal'] for r in records for c in r['cases'] if c['variant'] == variant and c['status'] == 'measured')) for variant in plan['variants']],
            finished_utc=datetime.now(timezone.utc).isoformat())
        save(out / 'summary.json', summary)
    except BaseException as error:
        save(out / 'failure.json', dict(status='terminal_failure_or_unmeasured_anchor', error=repr(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--allocation-note', type=Path)
    args = parser.parse_args(); plan = json.loads(PLAN.read_text()); validate(plan, 'container' if args.execute else 'host')
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_started', torch_imported=False, container_started=False, GPU_started=False, run_plan_sha256=sha(PLAN))))
        return
    assert args.allocation_note is not None
    host = json.loads((HERE / 'state.json').read_text())
    assert host['run_plan_sha256'] == sha(PLAN)
    note_sha = sha(args.allocation_note)
    assert note_sha == host['allocation_note_sha256']
    note = json.loads(args.allocation_note.read_text()); assert note == host['allocation']
    # No model imports occur until the host has checked the actual Docker
    # image, devices, read-only bind sources and init process identity.
    import time
    proof_path = HERE / 'container_mount_proof.json'
    deadline = time.monotonic() + 30
    while not proof_path.exists() and time.monotonic() < deadline: time.sleep(.05)
    mount_proof = json.loads(proof_path.read_text())
    assert mount_proof['run_plan_sha256'] == sha(PLAN)
    assert mount_proof['all_requested_bind_sources_and_readonly_flags_verified']
    assert mount_proof['GPU_devices_requested']
    host = json.loads((HERE / 'state.json').read_text())
    assert mount_proof['container_id'] == host['container_identity']['container_id']
    gpu_diagnostic(plan, note, note_sha)


if __name__ == '__main__': main()
