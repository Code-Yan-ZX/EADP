"""Read completed controls and CPU checks only; no generation or old-file edits."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
OUT = ROOT / 'Qwen_vl/outputs/audit_followup_20261008'
HERE = Path(__file__).resolve().parent
STABILITY = OUT / 'next_gap_diagnosis_20261008/stability'
ARMS = ('EADP_beta2', 'AZ_beta2')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use unchanged llava_pruner Python 3.10')
    output = HERE / 'next_gap_control_review.json'
    if output.exists():
        raise FileExistsError('Preserve previous independent review')
    cpu_path = HERE / 'json_count_distribution_overlay_cpu_validation.json'
    cpu = read(cpu_path)
    if cpu.get('success') is not True or not all(cpu['gates'].values()):
        raise AssertionError('All CPU strict gates must pass')
    plan_path = STABILITY / 'next_text_full_streamwait.controller.plan.json'
    state_path = STABILITY / 'next_text_full_streamwait.controller.state.json'
    plan, state = read(plan_path), read(state_path)
    paper_path = OUT / 'next_gap_diagnosis_20261008/paper_config/paper_config_audit.json'
    paper = read(paper_path)
    paper32 = next(row for row in paper['gap_rows'] if row['budget_parameter'] == 32)
    if paper32['paper_reported_accuracy'] != 54.2:
        raise AssertionError('Locked paper K32 reference changed')
    files = {str(path): sha(path) for path in (cpu_path, plan_path, state_path, paper_path, Path(__file__))}
    records = {}
    for arm in ARMS:
        job = plan['jobs'][arm]
        paths = job['output_paths']
        stored, marker, worker_state, protocol, native = (read(paths[name]) for name in
            ('score', 'finished', 'state', 'control_protocol', 'native_protocol'))
        exact = cpu['exact_full_scores'][arm]
        if any(stored.get(key) != value for key, value in exact.items()):
            raise AssertionError('CPU strict exact score differs')
        source_check = {path: sha(path) == digest for path, digest in protocol['source_and_input_sha256'].items()}
        if not all(source_check.values()):
            raise AssertionError('Frozen source or input drift')
        if not (stored['n'] == marker['n'] == protocol['question_count'] == native['question_rows'] == 5000
                and marker['success'] is True and marker['complete'] is True
                and worker_state['status'] == 'complete' and worker_state['generation_exit_code'] == 0
                and marker['prediction_sha256'] == sha(paths['prediction'])
                and marker['score_sha256'] == sha(paths['score'])
                and stored['runtime_sha256'] == sha(paths['runtime'])):
            raise AssertionError('Complete5000 marker/identity gate failed')
        for path in paths.values():
            if Path(path).is_file():
                files[path] = sha(path)
        files.update(protocol['source_and_input_sha256'])
        records[arm] = dict(n=5000, generation_exit_code=0, worker_complete=True, valid_success_marker=True,
            official_accuracy_percent=exact['accuracy_percent'], parameters=protocol['parameters'],
            prediction_sha256=exact['prediction_sha256'], runtime_sha256=exact['runtime_sha256'],
            score_sha256=sha(paths['score']), source_and_input_all_matching=True,
            source_and_input_count=len(source_check), source_and_input_match=source_check,
            official_evaluator_sha256=exact['official_evaluator_sha256'], GT_sha256=exact['GT_sha256'],
            actual_count_distribution=exact['actual_count_distribution'], all5000_image_tensor_shape=[1, 5, 3, 336, 336],
            native_protocol_unchanged=True, no_partial_prefix_scored=True,
            AST_validation=protocol['AST_validation'],
            loaded_parameter_boundary='Actual retained-count/geometry/prompt/budget traces verified; alpha/beta/lambda and dtype/class are recorded source/protocol values, not dynamic object snapshots')
    e = records['EADP_beta2']['official_accuracy_percent']
    az = records['AZ_beta2']['official_accuracy_percent']
    old_e = paper32['local_official_accuracy']
    old_az_path = ROOT / 'LLaVA/playground/data/eval/anchorzip_p3/next_textvqa_official_20261008/LRMAIN00625.jsonl'
    old_az = 53.174
    files[str(old_az_path)] = sha(old_az_path)
    overlay = HERE / 'json_count_distribution_overlay.py'
    validator = HERE / 'validate_json_count_distribution_overlay.py'
    files[str(overlay)] = sha(overlay)
    files[str(validator)] = sha(validator)
    report = dict(schema_version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Independent CPU review of complete NeXT TextVQA K32 wait-only E2/AZ2; new review files only',
        live_branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        live_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        handoff=dict(path=str(ROOT / 'docs/project_handoff.md'), full_read_lines=1029,
                     sha256=sha(ROOT / 'docs/project_handoff.md')),
        GPU_started=False, original_files_modified=False, frozen_queue_modified=False,
        threshold_policy=dict(user_deficit_tolerance_percentage_points=0.2,
            condition='Same-protocol full official EADP score at least paper minus0.2 is numerically accepted; above paper is accepted; parameter identity remains a separate evidence question',
            no_parameter_search=True, no_partial_prefix_scoring=True),
        paper_reference=dict(accuracy_percent=54.2, budget_parameter_per_crop=32,
            nominal_total_tokens=160, actual_retention_floor='156..159 matches official Eq26 floor/min1',
            source_url=paper32['paper_source_url'], source_path=str(paper_path), source_sha256=sha(paper_path)),
        completed_controls=records,
        threshold_result=dict(EADP_minus_paper_percentage_points=e-54.2,
            EADP_deficit_percentage_points=54.2-e, acceptable_minimum_accuracy_percent=54.0,
            EADP_numerical_acceptance=False, deficit_exceeds_point2=True,
            AnchorZip_minus_EADP_percentage_points=az-e,
            wait_EADP_minus_original_EADP_percentage_points=e-old_e,
            wait_AnchorZip_minus_original_AnchorZip_percentage_points=az-old_az,
            original_AZ_comparison_precision='Historical complete score53.174 is used to its recorded three decimals; no invented last-bit precision',
            statistical_claim='Complete point estimates only; no new paired bootstrap or significance claim'),
        confirmed_gate_bug=dict(
            source_controller=str(OUT / 'next_gap_diagnosis_20261008/schedule_next_full_controls.py') + ':143',
            source_publisher=str(ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot/publish_repair_results.py') + ':405',
            source_worker=str(STABILITY / 'next_text_full_streamwait.py') + ':164',
            mechanism='Frozen validator returns Counter(int) keys; serialized JSON score reload has string keys. Exact dict comparison falsely rejects both completed controls.',
            exact_same_interpreter='In actual Python3.10 gate environment accuracy and every other original return field are exactly equal; only actual_count_distribution key types differ',
            generation_status='Both actual workers exited0 and wrote valid complete5000 scores and success markers; controller failed solely during independent CPU verification, beta1 skipped its dependency',
            historical_controller_status_snapshot=state.get('status'),
            historical_job_status_snapshot={arm: job.get('status') for arm, job in state.get('jobs', {}).items()},
            not_an_accuracy_change=True,
            independent_python312_boundary='Python3.12 standalone rescoring also changes last-bit sums (~1e-13pp), but actual controller and publisher use Python3.10; this is not the incident cause and no tolerance is introduced'),
        minimal_recovery=dict(overlay_path=str(overlay), overlay_sha256=sha(overlay),
            worker_path=str(STABILITY / 'next_text_full_streamwait.py'),
            worker_SHA_pinned='80affa9c875fb25c5f78cea39a437efc81b36a9b8383a19e77166516f58cda63',
            only_changed_return_field='actual_count_distribution keys str(k)',
            source_and_score_and_prediction_preserved=True,
            unchanged='Original full validator, all values/math, strict source/input/runtime/denominator/marker checks, original CPU entry scripts and all generation workers',
            tests=dict(gates=cpu['gate_count'], all_pass=True, full5000_real_strict_score_checks=True,
                hashseeds=[0,731], wrong_SHA_refused=True, changed_count_refused=True,
                changed_denominator_refused=True, tiny_changed_accuracy_refused=True, no_tolerance=True),
            prepared_only=True, launcher_or_resume_not_run_by_this_audit=True,
            supervisor_state_recovery_owned_by_root=True),
        next_single_factor_causal_path=[
            dict(priority=0, action='Restore CPU verification using pinned JSON-key overlay after archiving failed supervisory state; retain both complete scores/predictions.',
                 expected='E2/AZ2 strictly validate without regeneration; both genuine full scores remain52.086/53.262.'),
            dict(priority=1, action='Complete the already preregistered unique EADP_beta1 same5000 K32 wait-only control, keeping alpha.5, input, guidance, decoding and environment fixed.',
                 reason='Official NeXT textvqa.sh:11 beta default1.0 differs from local2.0. Paper actual argv unknown.',
                 falsifiable='Unchanged output/score rejects an appreciable beta contribution; a changed score measures that one factor. Apply0.2pp rule to full score, never choose or search a best beta.',
                 limitation='Even numerical paper agreement would not establish that the authors used beta1, and does not transfer the increment to AZ.'),
            dict(priority=2, action='If beta1 still fails0.2pp, close actual paper task argv and pruning execution identity using primary released config and noninterfering post-generation parameter traces before any new full rerun.',
                 evidence='Official input/prompt/maxnew and exact core implementations already match; scoring suffix fix and wait dependency are applied. q=.2 is official hardcoded call, allocation floor/min1 and5crop are verified, sharedCLIP/NeXT checkpoint hashes match.',
                 limitation='Do not change q/alpha/crop budgets as a score search. Dynamic alpha/beta/lambda/dtype/model-class values were not recorded in old runtime, so protocol values alone cannot claim actual loaded object identity.'),
            dict(priority=3, action='Investigate remaining environment/precision differences only after a concrete divergence on fixed identical inputs is demonstrated; then freeze a separate one-factor control.',
                 evidence='tokenizers/safetensors/Pillow/sentencepiece versions differ; flash-attn andftfy absent; official builder default uses no externalFlash2, GenerationConfig metadata does not prove active backend.',
                 limitation='No causal accuracy effect established; FULL60.370 vs paper60.3 narrows global failures but cannot exclude pruning-specific effects. No env update or speculative GPU run authorized by this CPU artifact.'),
        ],
        method_side_effect_boundaries=[
            'Wait-only producer ordering has direct stability evidence, but full E2 accuracy gains only0.090pp from original51.996; it does not explain remaining2.114pp.',
            'Both E2/AZ2 share the same wait-only repair and complete5000 protocol; AZ−E=+1.176pp is a same-local-method point difference, not evidence of general dominance or a paper-matched baseline.',
            'Confirmed NeXT cross-crop Completion deviation is AZ-only; isolated per-crop adapter remains uninstalled and untested for GPU accuracy, and cannot explain EADP deficit.',
            'AZ first-text-segment RTG behavior is documented as unresolved M>1 design; do not call it a proven bug or silently change it.',
            'FULLnear-paper does not prove pruning, author beta, environment precision or current method math correctness.',
        ],
        evidence_sha256=files)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(path=str(output), EADP=e, AZ=az, deficit=54.2-e, AZ_minus_E=az-e,
        cpu_gates=cpu['gate_count'], sha256=sha(output)), indent=2))


if __name__ == '__main__':
    main()
