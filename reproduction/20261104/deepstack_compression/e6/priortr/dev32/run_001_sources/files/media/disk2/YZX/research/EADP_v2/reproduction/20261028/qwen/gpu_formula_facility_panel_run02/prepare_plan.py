"""Freeze positions and source/input guards on CPU, without importing model code."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
QWEN = HERE.parent
ROOT = QWEN.parents[2]
sys.path.insert(0, str(HERE))
from diagnose import sha, stat, validate, save
from formula_helpers import parts


def main():
    assert not (HERE / 'run_plan.json').exists() and not (HERE / 'execution').exists()
    run01_path = HERE / 'run01_frozen_plan_reference.json'
    run01 = json.loads(run01_path.read_text())
    assert sha(run01_path) == '7701cfd2e78fb72f6a83855887dff611601c3cab00b01be6df6ff3ef0e21c835'
    generation = json.loads((QWEN / 'public_fa2_full_run02_plan.json').read_text())
    probe = json.loads((QWEN / 'source_probe_plan.json').read_text())
    author = ROOT / '_reproduction/author_eadp/Qwen_vl'
    pruner = author / 'model/pruner.py'
    prompt = author / 'VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
    source = dict(generation['source_sha256'])
    assert all(sha(path) == expected for path, expected in source.items())
    canonical_path = QWEN / 'independent_audit/DocVQA_VAL.canonical.jsonl'
    canonical = [json.loads(line) for line in canonical_path.read_text().splitlines()]
    assert len(canonical) == 5349
    reference = QWEN / 'processor_input_identity/author_qwen_fullDoc/per_question.jsonl'
    positions = [((2 * i + 1) * 5349) // 32 for i in range(16)]
    expected = {}
    for i, line in enumerate(reference.open()):
        if i in positions: expected[i] = json.loads(line)
    assert set(expected) == set(positions)
    cache_audit = json.loads((QWEN / 'dataset_cache_audit.json').read_text())['datasets']['DocVQA_VAL']
    images = {row['image_path']: row['cached_sha256'] for row in cache_audit['images']}
    tree = ast.parse(prompt.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3VLPromptMixin')
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_vqa_prompt')
    namespace = {}; exec(compile(ast.Module(body=[build], type_ignores=[]), str(prompt), 'exec'), namespace)
    panel = []
    for position in positions:
        row = canonical[position]
        # Only index/question/image fields are copied. GT/predictions/scores
        # never enter this plan or its worker, and do not affect fixed positions.
        path = str(Path('/media/disk2/YZX/LMUData/images/DocVQA_VAL') / row['image_path'])
        context = types.SimpleNamespace(dump_image=lambda line, dataset: [path])
        message = namespace['_build_vqa_prompt'](context, {'question': row['question']}, 'DocVQA_VAL')
        assert path in images and sha(path) == images[path]
        old = expected[position]
        assert str(old['index']) == str(row['index']) and old['image_path'] == path
        panel.append(dict(position=position, index=str(row['index']), image_path=path, image_sha256=images[path],
                          message=message, source_messages_sha256=old['message_sha256'],
                          processor_expected=old['native_CPU_BF16_tensors'],
                          question_sha256=__import__('hashlib').sha256(row['question'].encode()).hexdigest()))
    terminal_preparation = QWEN / 'independent_audit/terminal_audit_preparation.json'
    prior = json.loads(terminal_preparation.read_text())['prior_SHA_and_current_model_stat_guards']
    guards = {}
    for item in prior:
        current = stat(item['path'])
        assert current['bytes'] == item['stat']['bytes'] and current['mtime_ns'] == item['stat']['mtime_ns']
        guards[item['path']] = dict(prior_verified_sha256=item['expected_sha256'], current_full_stat=current,
                                    prior_guard_source=str(terminal_preparation), prior_stat=item['stat'])
    model = Path(generation['recipe']['model_path'])
    model_bind_root = model.parents[1]
    namespace_probe = json.loads((HERE / 'namespace_probe_report.json').read_text())
    for path, guard in guards.items():
        assert guard['current_full_stat'] == run01['model_stat_guards'][path]['current_full_stat']
        observed = namespace_probe['model_stats'][path]['actual']
        assert {k:v for k,v in observed.items() if k!='resolved'} == {k:v for k,v in guard['current_full_stat'].items() if k!='resolved'}
        relative = Path(guard['current_full_stat']['resolved']).relative_to(model_bind_root.resolve())
        expected_container = str(model_bind_root / relative)
        assert observed['resolved'] == expected_container
        guard['container_resolved'] = expected_container
    source.update({str(p): sha(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja')})
    overlay = QWEN / 'import_overlay'
    source.update({str(p): sha(p) for p in [overlay / 'sitecustomize.py', overlay / 'result_transfer.py',
                                          QWEN / 'container_runtime.py']})
    library = ROOT / '_reproduction/envs/qwen_author/lib/python3.10/site-packages'
    modeling = library / 'transformers/models/qwen3_vl/modeling_qwen3_vl.py'
    source[str(modeling)] = sha(modeling)
    for name in ('libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-ptxjitcompiler.so.1'):
        driver = (Path('/usr/lib/x86_64-linux-gnu') / name).resolve(strict=True)
        source[str(driver)] = sha(driver)
    source['/usr/bin/nvidia-smi'] = sha('/usr/bin/nvidia-smi')
    references = [QWEN / 'public_fa2_full_run02_plan.json', QWEN / 'source_probe_plan.json',
                  QWEN / 'dataset_cache_audit.json', QWEN / 'qwen_weight_identity.json',
                  QWEN / 'qwen_processor_tokenizer_identity.json', terminal_preparation,
                  QWEN / 'processor_input_identity/fullDoc_comparison.json',
                  QWEN / 'processor_input_identity/author_qwen_fullDoc/report.json',
                  reference, HERE / 'cache_inventory.json', run01_path,
                  HERE / 'namespace_probe_report.json', HERE / 'namespace_probe_container_record.json',
                  HERE / 'driver_mount_mapping.json', QWEN / 'gpu_formula_facility_panel/root_failed_terminal_process_proof.json']
    implementations = [HERE / name for name in ['formula_helpers.py', 'diagnose.py', 'run_panel.py', 'prepare_plan.py', 'cpu_acceptance.py', 'README.md', 'namespace_guards.py', 'cpu_preflight.py', 'preflight.py']]
    plan = dict(run_id='Qwen_Doc_fixed16_LLMSpace_K256_a05_b2_formula_facility_run02',
        prepared_utc=datetime.now(timezone.utc).isoformat(), execution_status='prepared_not_started',
        python=str(ROOT / '_reproduction/envs/qwen_author/bin/python'), author_root=str(author),
        model_name=generation['model_name'], recipe=generation['recipe'], expected_versions=probe['versions'],
        pruner_path=str(pruner), modeling_path=str(modeling), source_AST_proof=parts(pruner)[-1],
        container_runtime_source=str(QWEN / 'container_runtime.py'),
        container_image='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf',
        container_name='eadp-qwen-doc16-formula-run02-20261028', import_overlay=str(overlay),
        source_sha256=source, implementation_sha256={str(p): sha(p) for p in implementations},
        reference_sha256={str(p): sha(p) for p in references}, model_stat_guards=guards,
        preparation_only_canonical_provenance=dict(path=str(canonical_path), sha256=sha(canonical_path),
            runtime_file_read=False, fields_copied=['index', 'question', 'image_path'],
            GT_or_prediction_or_score_used_for_selection=False),
        panel_rule='position=floor((2*i+1)*5349/32), i=0..15, fixed before numeric extraction; no outcome selection',
        panel=panel, variants=['source', 'source_members_order_identity', 'eq6_quantile_membership_only', 'eq10_numerator_only'],
        native_get_calls=16, planned_case_slots=64,
        Eq6=dict(q=.2, convention='torch.quantile linear interpolation; entropy <= threshold includes all ties',
                 candidates='All actual LLM instruction-token positions, actual dynamic L, exactly the native unmasked domain',
                 order='Preserve source top-floor members and order, append new threshold members by original token position',
                 ragged='Explicit unmeasured; no padding, rowwise reduction or batch replacement'),
        Eq10='Only remove numerator +1e-6; denominator +1e-6 and all other original arithmetic retained',
        dtype={'visual':'BF16 post-merger original Qwen3 visual', 'text':'Original LLM input embeddings BF16',
               'global':'Original instruction sequence mean in native dtype, not CLIP',
               'cross':'Source casts BF16 features/text to FP32 before norm and negative cosine',
               'visual_sim':'Source BF16 normalization then cast to FP32 matmul', 'importance':'FP32'},
        input_chain='Native _build_messages/_processor_inputs/_get_instruction_sequence_embedding/_get_pruned_image_features; no answer generation',
        native_anchor=['Once original _get_pruned_image_features and pruner', 'Exact original global/dense/visualsim/importance',
                       'Source reconstructed ordered256 IDs and sorted final BF16 features bit-identical',
                       'Strict identity all importance/gain/ordered256/final sequence equal; trace matches unmodified greedy'],
        save_policy='Native BF16 values stored as uint16 bit patterns with actual dtype/shape/SHA; FP32 arrays unchanged. Save full visual, instruction sequence/global, global/dense/sim/importance, IDs, native final features, gain traces; no repeated duplicate arrays for variant-selected BF16 features.',
        launch_policy=dict(root_review_required=True, separate_exact_root_allocation_required=True, max_GPU_models=2,
                           single_attempt=True, automatic_retry=False, watcher_or_queue_created=False, GPU_allocated=False),
        claim_boundary=['Only16 fixed Doc questions and one already-completed K256/.5/2 recipe; feature/facility mechanism only.',
                        'No generated answers, GT input, accuracy, external API or external server. No parameter search or automatic full extension.',
                        'Linear quantile is an explicit threshold convention, not evidence of unpublished paper implementation.',
                        'No CLIP adapter, contextual LLM forward, DeepStack restoration, mRoPE/position repair or source numerical edits.',
                        'A source or identity anchor mismatch is retained as failure/unmeasured, never interpreted as a formula outcome.',
                        'Historic caches exist but do not supply a qualified complete native anchor for these fixed questions; see inventory.',
                        'Current author runtime diagnostics do not reconstruct the September historical environment or certify paper argv.'])
    from run_panel import command
    plan['guard_namespace_mounts'] = json.loads((HERE / 'driver_mount_mapping.json').read_text())
    plan['model_bind_namespace'] = dict(source=str(model_bind_root), destination=str(model_bind_root),
                                       host_resolved_root=str(model_bind_root.resolve()), source_readonly=True,
                                       probe_report_sha256=sha(HERE/'namespace_probe_report.json'))
    plan['planned_container_command'] = command(plan, 'FUTURE_ROOT_REVIEWED_ALLOCATION.json')
    from namespace_guards import bindings
    plan['expected_bind_mounts'] = bindings(plan['planned_container_command'])
    assert sha(HERE/'formula_helpers.py') == run01['implementation_sha256'][str(QWEN/'gpu_formula_facility_panel/formula_helpers.py')]
    assert plan['panel']==run01['panel'] and plan['recipe']==run01['recipe'] and plan['source_AST_proof']==run01['source_AST_proof']
    plan['run01_preserved'] = dict(plan_sha256=sha(run01_path), original_directory=str(QWEN/'gpu_formula_facility_panel'),
        correction_scope='Driver/smi original-path readonly aliases and explicit model bind namespace only; formula/native feature/recipe/panel arithmetic unchanged')
    save(HERE / 'run_plan.json', plan); validate(plan)
    save(HERE / 'READY.json', dict(status='prepared_not_started', GPU_loaded=False, container_started=False,
         watcher_created=False, run_plan_sha256=sha(HERE / 'run_plan.json'), separate_root_allocation_required=True,
         command_for_review=['/home/dell/miniconda3/envs/llava_pruner/bin/python', str(HERE / 'run_panel.py'), 'run',
                             '--allocation-note', 'FUTURE_ROOT_REVIEWED_ALLOCATION.json']))
    print(json.dumps(dict(prepared=True, run_plan_sha256=sha(HERE / 'run_plan.json'), positions=positions,
                         n_questions=16, case_slots=64, torch_imported=False, container_started=False, GPU_started=False)))


if __name__ == '__main__': main()
