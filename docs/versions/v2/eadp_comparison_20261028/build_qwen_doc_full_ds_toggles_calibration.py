"""Export accepted FULL DeepStack diagnostics without rescoring or ledger edits."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUN = ROOT / 'reproduction/20261028/qwen/doc_full_ds_toggles'
BASE = ROOT / 'reproduction/20261028/qwen/doc_full_two_routes'
PLAN_SHA = '1eeeaa1f4a2cdbf12cdd1d6a17af4a7931ad221ba85021ec919596502323c482'
LOCKED = {
    RUN / 'plan.json': PLAN_SHA,
    RUN / 'RESULT.md': 'ffe93d8e2fd1988be145e7989c44860a780eb23f4abd981e287deb3df64b4bbf',
    RUN / 'score/score_report.json': 'f0a4d86377d946bbb44620bb6bbb6553af001e10e7f01d2905d50f25c54a4b44',
    RUN / 'root_terminal_process_proof.json': 'e1d5d1003e96aca181797018da09bdadbe9714e9ef35aabfec55217624e50724',
    RUN / 'score_terminal_process_proof.json': 'ed8f8ae84380c92f88a7083c219d74bec8cd2b15257a3c09c0311c8898377d42',
    RUN / 'execution/generation_verification.json': 'f05210380c9d78b40692f7578add4ea4f92bdc7d145d2725e44b9476fdb18173',
    BASE / 'score/score_report.json': '0b7843b902bc0fd50b05abc75791796373da5b3f4330387af5697ff18d0d37e0',
    BASE / 'root_terminal_process_proof.json': 'c6f7a6ab10f27e2d085ec1a7e80cbd5a31da620be65209cf5146e95e50851571',
    BASE / 'score_terminal_process_proof.json': '1b26e648136fe0862d5a0d84c28efc07b4126d0f7bc9416d6f47a3b6de4f8bbc',
}
ARMS = ('plain_ds_off', 'eadp1024_ds_on')


def artifact(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def complete_gates(report, generation, score_exit, verification):
    assert report['success'] and report['n'] == 5349 and report['full_denominators_preserved']
    assert report['all_generation_and_root_exit_before_first_score']
    assert report['plan_sha256'] == generation['plan_sha256'] == score_exit['plan_sha256'] == verification['plan_sha256'] == PLAN_SHA
    assert generation['root_owned_exec_actual_exit_code'] == 0 and generation['all_original_processes_exited']
    assert generation['container_not_running'] and generation['root_owned_exec_handle'] == 16152
    assert generation['n_questions'] == 5349 and generation['n_native_generations'] == 10698
    assert {row['role'] for row in generation['process_checks']} == {'controller', 'docker_client', 'host_init'}
    assert all(row['original_exited'] for row in generation['process_checks'])
    assert score_exit['root_owned_score_exec_actual_exit_code'] == 0 and score_exit['scorer_container_not_running']
    assert score_exit['root_owned_score_exec_handle'] == 11840 and score_exit['scored_questions_per_arm'] == 5349
    assert verification['success'] and verification['n_questions'] == 5349 and verification['n_generations'] == 10698
    assert verification['processor_identity_checks'] == verification['independent_token_redecodes'] == verification['prefill_intervention_checks'] == 10698
    assert verification['endpoint_anchors'] == verification['fixed16_variant_anchors'] == 32
    assert verification['high_dimensional_arrays_independently_reconstructed'] is False
    assert verification['GT_read'] is False and verification['scoring_started'] is False


def build_data():
    for path, digest in LOCKED.items():
        assert artifact(path)['sha256'] == digest, str(path)
    assert '独立只读终审通过' in (RUN / 'RESULT.md').read_text()
    report = read(RUN / 'score/score_report.json')
    generation = read(RUN / 'root_terminal_process_proof.json')
    score_exit = read(RUN / 'score_terminal_process_proof.json')
    verification = read(RUN / 'execution/generation_verification.json')
    complete_gates(report, generation, score_exit, verification)
    assert score_exit['generation_root_proof_sha256'] == LOCKED[RUN / 'root_terminal_process_proof.json']
    assert score_exit['score_report_sha256'] == LOCKED[RUN / 'score/score_report.json']
    assert generation['generation_verification_sha256'] == LOCKED[RUN / 'execution/generation_verification.json']
    state = read(RUN / 'state.json')
    release = read(RUN / 'execution/release.json')
    assert state['status'] == 'complete' and state['native_exit_code'] == state['CPU_verifier_exit_code'] == 0
    assert artifact(RUN / 'state.json')['sha256'] == generation['controller_state_sha256']
    assert artifact(RUN / 'execution/release.json')['sha256'] == generation['release_sha256'] == state['release_sha256']
    assert release['success'] and release['native_container_exited'] and release['GT_or_scoring_started'] is False
    assert release['generation_verification_sha256'] == generation['generation_verification_sha256']
    for name, key in [('root_gpu_allocation.json', 'allocation_sha256'), ('root_launch_process_snapshot.json', 'launch_snapshot_sha256')]:
        assert artifact(RUN / name)['sha256'] == generation[key]
    assert artifact(RUN / 'execution/native_complete.json')['sha256'] == verification['native_complete_sha256']
    assert artifact(RUN / 'execution/runtime.json')['sha256'] == verification['runtime_sha256']
    base_report = read(BASE / 'score/score_report.json')
    base_generation = read(BASE / 'root_terminal_process_proof.json')
    base_score_exit = read(BASE / 'score_terminal_process_proof.json')
    assert base_report['success'] and base_report['n'] == 5349 and base_report['full_denominators_preserved']
    assert base_generation['root_owned_exec_actual_exit_code'] == 0 and base_generation['all_original_processes_exited']
    assert base_score_exit['root_owned_score_exec_actual_exit_code'] == 0 and base_score_exit['scorer_container_not_running']
    assert base_score_exit['generation_root_proof_sha256'] == LOCKED[BASE / 'root_terminal_process_proof.json']
    assert base_score_exit['score_report_sha256'] == LOCKED[BASE / 'score/score_report.json']
    rows = []
    for arm, baseline, label in [
        ('plain_ds_off', 'plain', '原生 FULL：关闭 DeepStack'),
        ('eadp1024_ds_on', 'eadp1024', 'EADP K1024 全token：开启 DeepStack'),
    ]:
        score, old = report['scores'][arm], base_report['scores'][baseline]
        assert score['all_pointwise_equal'] and abs(score['official'] - score['independent']) < 1e-10
        assert old['all_pointwise_equal'] and abs(old['official'] - old['independent']) < 1e-10
        value, baseline_value = score['independent'], old['independent']
        delta = Decimal(str(value)) - Decimal(str(baseline_value))
        compared = report['per_arm_comparison'][arm]
        assert abs(float(delta) - compared['score_difference']) < 1e-12
        predictions = artifact(RUN / 'execution' / (arm + '.predictions.jsonl'))
        observations = artifact(RUN / 'execution' / (arm + '.observations.jsonl'))
        assert predictions['sha256'] == verification['prediction_sha256'][arm]
        assert observations['sha256'] == verification['observation_sha256'][arm]
        pointwise = artifact(RUN / 'score' / (arm + '.per_question.jsonl'))
        base_pointwise = artifact(BASE / 'score' / (baseline + '.per_question.jsonl'))
        assert pointwise['sha256'] == score['per_question_sha256']
        assert base_pointwise['sha256'] == old['per_question_sha256']
        for path in [predictions['path'], observations['path'], pointwise['path'], base_pointwise['path']]:
            with Path(path).open() as stream:
                assert sum(bool(line.strip()) for line in stream) == 5349, path
        rows.append(dict(model='qwen', task='DocVQA', method=label, arm=arm, n=5349,
            nominal_tokens=1024, paper_value=None, local_minus_paper=None,
            official_value=score['official'], independent_value=value, local_value=value,
            baseline_arm=baseline, baseline_value=baseline_value, toggle_minus_baseline=str(delta),
            unit='percentage_points', changed_answers=compared['changed_answers'],
            improved=compared['improved'], declined=compared['declined'],
            unchanged=5349-compared['improved']-compared['declined'],
            empty_answers=score['empty_answers'], native_NA_conversions=score['NA_conversions'],
            conditions='同路径跨运行对照；FA2/BF16；5349题；全部1024视觉tokens；仅首次prefill开关DeepStack与对应visual mask；固定16锚，非论文K256配方',
            raw_prediction=predictions, runtime_observations=observations,
            pointwise_scores=pointwise, baseline_pointwise_scores=base_pointwise))
    paired = artifact(RUN / 'score/paired_rows.jsonl')
    assert paired['sha256'] == report['paired_rows_sha256']
    return dict(definition='Full-token within-route DeepStack diagnostics; separate from paper pruning rows and the12-run ledger. Difference=toggle-minus-accepted-same-route-baseline.',
        rows=rows, plan=artifact(RUN / 'plan.json'), result_note=artifact(RUN / 'RESULT.md'),
        score_report=artifact(RUN / 'score/score_report.json'),
        generation_exit_proof=artifact(RUN / 'root_terminal_process_proof.json'),
        score_exit_proof=artifact(RUN / 'score_terminal_process_proof.json'),
        generation_verification=artifact(RUN / 'execution/generation_verification.json'),
        generation_release=artifact(RUN / 'execution/release.json'),
        generation_runtime=artifact(RUN / 'execution/runtime.json'),
        baseline_score_report=artifact(BASE / 'score/score_report.json'),
        baseline_generation_exit_proof=artifact(BASE / 'root_terminal_process_proof.json'),
        baseline_score_exit_proof=artifact(BASE / 'score_terminal_process_proof.json'),
        paired_rows=paired, builder=artifact(__file__), paper_argv_confirmed=False,
        part_of_eadp_pruning_ledger=False, full_denominators_preserved=True,
        conditional_cross_run_comparison=True, high_dimensional_arrays_saved=False,
        limits=[
            '旧完整基线未保存全部逐题内部张量与数值标志；仅固定16端点和变体跨运行锚定，属于有条件的同路径跨运行对照。',
            '新全量高维张量仅保存运行时摘要，不能事后独立重建。',
            'DeepStack及对应visual mask只在首次prefill联合开关；干预可自然改变后续KV与生成。',
            '两条原生API仍有其他区别；不能把两个差值完整分摊总路径差，也不能外推K256、替代论文剪枝值或认定paper argv。',
        ])


def main():
    data = build_data()
    (OUT / 'qwen_doc_full_ds_toggles_calibration.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    fields = ['model', 'task', 'method', 'arm', 'n', 'nominal_tokens', 'baseline_arm', 'baseline_value',
              'local_value', 'toggle_minus_baseline', 'unit', 'changed_answers', 'improved', 'declined', 'unchanged', 'conditions']
    with (OUT / 'qwen_doc_full_ds_toggles_calibration.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(data['rows'])
    print(json.dumps(dict(FULL_DeepStack_diagnostics=2, n_per_arm=5349, main_ledger_modified=False, rescored=False)))


if __name__ == '__main__':
    main()
