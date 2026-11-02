"""Export a completed K256 implementation diagnostic; never rescore or edit ledgers."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUN = ROOT / 'reproduction/20261028/qwen/doc_full_k256_ds_pair'
PLAN_SHA = '22f380977f660f1d6fdb7d0b0937ef6ff93cb2667f733874b321817b17f516d5'
LOCKED = {
    RUN / 'plan.json': PLAN_SHA,
    RUN / 'RESULT.md': 'cc80da73be11d3d5eebd093213726cd61d8cc1b3681dfd4278c030f6aeaca7e1',
    RUN / 'score/score_report.json': '263d6438d2fc6ed4982e98cbb66ec527209b12ec3b12c82fea75aba2bd9ca981',
    RUN / 'root_terminal_process_proof.json': '7908f769268b6987bb78666d6b3e9ef780ed0bb6ea2ebca66a32271220f5b79f',
    RUN / 'root_score_exit_proof.json': 'cabc28a998306b4ed9f16e2cd5760c7ff9c2ad3c7b1e0f029e94931467700098',
    RUN / 'execution/independent_verification.json': 'b8657133c11535fb73d68f6aaaa5c7c90c027eb00fe38f2046bb6fa0642efae9',
}
ARMS = ('ds_off', 'ds_on')


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
    assert report['plan_sha256'] == generation['plan_sha256'] == verification['plan_sha256'] == PLAN_SHA
    assert generation['root_owned_exec_handle'] == 86476 and generation['root_owned_exec_actual_exit_code'] == 0
    assert generation['all_original_processes_exited'] and generation['container_not_running']
    assert {item['role'] for item in generation['process_checks']} == {'controller', 'docker_client', 'host_init'}
    assert all(item['original_exited'] for item in generation['process_checks'])
    assert score_exit['score_root_pty_session'] == 20783 and score_exit['score_root_pty_actual_exit_code'] == 0
    assert score_exit['container_original_process_absent'] and score_exit['container_not_running']
    assert verification['success'] and verification['n_questions'] == 5349 and verification['n_generations'] == 10698
    assert verification['native_anchors'] == 32
    assert verification['processor_identity_checks'] == verification['selection_checks'] == verification['prefill_intervention_checks'] == verification['independent_token_redecodes'] == 10698
    assert verification['high_dimensional_arrays_independently_reconstructed_for_nonanchors'] is False
    assert verification['GT_read'] is False and verification['scoring_started'] is False
    assert report['CPU_only'] and report['model_loaded'] is False and report['CUDA_initialized'] is False


def build_data():
    for path, expected in LOCKED.items():
        assert artifact(path)['sha256'] == expected, str(path)
    assert '独立只读终审通过' in (RUN / 'RESULT.md').read_text()
    report = read(RUN / 'score/score_report.json')
    generation = read(RUN / 'root_terminal_process_proof.json')
    score_exit = read(RUN / 'root_score_exit_proof.json')
    verification = read(RUN / 'execution/independent_verification.json')
    complete_gates(report, generation, score_exit, verification)
    assert score_exit['generation_root_proof_sha256'] == LOCKED[RUN / 'root_terminal_process_proof.json']
    assert score_exit['score_report_sha256'] == LOCKED[RUN / 'score/score_report.json']
    assert generation['independent_verification_sha256'] == LOCKED[RUN / 'execution/independent_verification.json']
    state, release = read(RUN / 'state.json'), read(RUN / 'execution/release.json')
    assert state['status'] == 'complete' and state['native_exit_code'] == state['CPU_verifier_exit_code'] == 0
    assert state['original_container_exited'] and state['GPU_pid_absent'] and state['automatic_retry'] is False
    assert state['plan_sha256'] == release['plan_sha256'] == PLAN_SHA
    assert artifact(RUN / 'state.json')['sha256'] == generation['controller_state_sha256']
    assert artifact(RUN / 'execution/release.json')['sha256'] == generation['release_sha256'] == state['release_sha256']
    assert release['success'] and release['native_container_exited'] and release['GT_or_scoring_started'] is False
    assert release['independent_verification_sha256'] == generation['independent_verification_sha256']
    for filename, key in [('root_gpu_allocation.json', 'allocation_sha256'), ('root_launch_process_snapshot.json', 'launch_snapshot_sha256')]:
        assert artifact(RUN / filename)['sha256'] == generation[key]
    assert artifact(RUN / 'root_score_launch_snapshot.json')['sha256'] == score_exit['score_launch_sha256']
    for filename, key in [('native_complete.json', 'native_complete_sha256'), ('native_anchors.json', 'native_anchors_sha256'), ('runtime.json', 'runtime_sha256')]:
        assert artifact(RUN / 'execution' / filename)['sha256'] == verification[key]
    runtime = read(RUN / 'execution/runtime.json')
    assert runtime['model_instances'] == 1 and runtime['model_dtype'] == 'torch.bfloat16' and runtime['backend'] == 'flash_attention_2'
    recipe = runtime['actual_recipe']
    assert recipe['visual_token_num'] == 256 and recipe['alpha'] == .5 and recipe['beta'] == 2
    assert recipe['min_pixels'] == recipe['max_pixels'] == 1048576 and recipe['max_new_tokens'] == 2048
    history = read(OUT / 'comparison.json')
    paper = [row for row in history['rows'] if row['model'] == 'Qwen3-VL-8B' and row['task'] == 'DocVQA' and row['budget_nominal'] == 256]
    assert len(paper) == 1 and paper[0]['paper_value'] == 62.8
    paper = paper[0]
    rows = []
    for arm, label in [('ds_off', '公开 K256 路径：DS-off（本次重跑）'), ('ds_on', '修改实现 K256：DS-on')]:
        score = report['scores'][arm]
        assert score['all_pointwise_equal'] and abs(score['official'] - score['independent']) < 1e-10
        references = {}
        for field, suffix, name in [
            ('prediction_sha256', '.predictions.jsonl', 'raw_prediction'),
            ('observation_sha256', '.observations.jsonl', 'runtime_observations'),
            ('ordered_prediction_sha256', '.ordered.predictions.jsonl', 'ordered_prediction'),
            ('ordered_observation_sha256', '.ordered.observations.jsonl', 'ordered_observations')]:
            reference = artifact(RUN / 'execution' / (arm + suffix))
            assert reference['sha256'] == verification[field][arm]
            with Path(reference['path']).open() as stream:
                assert sum(bool(line.strip()) for line in stream) == 5349
            references[name] = reference
        for suffix, key, name in [('.per_question.jsonl', 'per_question_sha256', 'pointwise_scores'),
                                  ('.xlsx', 'native_input_sha256', 'native_score_input'),
                                  ('_results.xlsx', 'native_detail_sha256', 'native_score_detail')]:
            reference = artifact(RUN / 'score' / (arm + suffix))
            assert reference['sha256'] == score[key]
            references[name] = reference
        with Path(references['pointwise_scores']['path']).open() as stream:
            assert sum(bool(line.strip()) for line in stream) == 5349
        endings = verification['ending_counts'][arm]
        assert endings['EOS_present'] + endings['EOS_missing'] == 5349
        distance = Decimal(str(score['independent'])) - Decimal(str(paper['paper_value']))
        rows.append(dict(model='Qwen3-VL-8B', task='DocVQA', arm=arm, method=label,
            n=5349, nominal_tokens=256, paper_value=None, local_minus_paper=None,
            paper_reference_value=paper['paper_value'], diagnostic_minus_paper_reference=str(distance),
            official_value=score['official'], independent_value=score['independent'], local_value=score['independent'],
            author_implementation_modified=arm == 'ds_on', native_NA_conversions=score['NA_conversions'],
            EOS_present=endings['EOS_present'], EOS_missing=endings['EOS_missing'], at_max_tokens=endings['at_max_tokens'],
            unit='percentage_points', conditions='同实例配对；FA2/BF16；5349题；同一实际256空间选择；首次prefill联合开关DS与visual mask；legacy位置策略保持；on修改公开实现，非paper配方',
            **references))
    delta = Decimal(str(rows[1]['local_value'])) - Decimal(str(rows[0]['local_value']))
    assert abs(float(delta) - report['ds_on_minus_off']) < 1e-10
    paired = artifact(RUN / 'score/paired_rows.jsonl')
    assert paired['sha256'] == report['paired_rows_sha256']
    assert report['improved'] + report['declined'] <= 5349
    changed_ids = sum(not row['comparison']['output_ids_equal'] for row in verification['results'])
    assert len(verification['results']) == 5349
    assert sum(not row['comparison']['response_equal'] for row in verification['results']) == report['changed_answers']
    return dict(definition='Same-instance K256 DeepStack implementation diagnostic; separate from paper pruning rows, all main ledgers and FULL calibrations.',
        rows=rows, ds_on_minus_off=report['ds_on_minus_off'], ds_on_minus_off_decimal_from_scores=str(delta),
        delta_precision_note='Primary delta is the original score_report value; Decimal subtraction of printed scalar representations differs only at the floating-point tail.',
        changed_answers=report['changed_answers'], changed_output_ids=changed_ids,
        improved=report['improved'], declined=report['declined'], unchanged=5349-report['improved']-report['declined'],
        same_model_instance=True, legacy_position_policy=True, high_dimensional_arrays_saved_for_nonanchors=False,
        paper_argv_confirmed=False, part_of_eadp_pruning_ledger=False, full_denominators_preserved=True,
        paper_reference=dict(value=paper['paper_value'], model=paper['model'], task=paper['task'], nominal_tokens=256,
                             paper_evidence_id=paper['paper_evidence_id'], reference_index=artifact(OUT / 'comparison.json'),
                             source=artifact(paper['paper_source'])),
        plan=artifact(RUN / 'plan.json'), result_note=artifact(RUN / 'RESULT.md'),
        score_report=artifact(RUN / 'score/score_report.json'), generation_exit_proof=artifact(RUN / 'root_terminal_process_proof.json'),
        score_exit_proof=artifact(RUN / 'root_score_exit_proof.json'), generation_verification=artifact(RUN / 'execution/independent_verification.json'),
        generation_release=artifact(RUN / 'execution/release.json'), generation_runtime=artifact(RUN / 'execution/runtime.json'),
        generation_state=artifact(RUN / 'state.json'), generation_launch=artifact(RUN / 'root_launch_process_snapshot.json'),
        generation_allocation=artifact(RUN / 'root_gpu_allocation.json'), score_launch=artifact(RUN / 'root_score_launch_snapshot.json'),
        paired_rows=paired, builder=artifact(__file__), limits=[
            'DS-on修改作者公开实现；高于论文62.8仅为数值距离，不能认证论文实际配方、覆盖原复现主行或当作新剪枝方法。',
            '保持legacy位置策略，没有同时恢复原生mRoPE；仅在首次prefill联合开关DS及对应visual mask，后续KV和输出可自然变化。',
            '两臂同实例逐题各重跑实际选择；固定16的32份真实数组已锚定，其余5333对高维张量只保存运行时摘要，无法事后独立重建。',
            '旧完整K256未保存逐题内部状态；不能由本次off与旧值相同推断所有旧内部量相同。',
            '两臂各4条无EOS并达到2048上限，NA转换分别7/6，全部5349分母保留；观测开销不能当效率测量。'])


def main():
    data = build_data()
    (OUT / 'qwen_doc_k256_ds_diagnostic.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    fields = ['model', 'task', 'arm', 'method', 'n', 'nominal_tokens', 'paper_reference_value',
              'official_value', 'local_value', 'diagnostic_minus_paper_reference', 'author_implementation_modified',
              'native_NA_conversions', 'EOS_present', 'EOS_missing', 'at_max_tokens', 'conditions']
    with (OUT / 'qwen_doc_k256_ds_diagnostic.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(data['rows'])
    print(json.dumps(dict(K256_diagnostic_arms=2, n_per_arm=5349, ds_on_minus_off=data['ds_on_minus_off'],
                         main_ledger_modified=False, rescored=False)))


if __name__ == '__main__':
    main()
