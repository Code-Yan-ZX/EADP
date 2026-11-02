"""Export completed MMB guidance diagnostics; never run a model or scorer."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUN = ROOT / 'reproduction/20261028/mmbench/question_only_guidance_full_run02'
FAILED = RUN.parent / 'question_only_guidance_full'
PLAN_SHA = '09404c6685b0e148f3ea6373b3a79fd3842997ef14f5be2891d16d5a8784ea99'
LOCKED = {
    RUN / 'plan.json': PLAN_SHA,
    RUN / 'RESULT.md': '870bd58c208db8d4f3586ce5042f9e8863e12b4453257148e9fab8e26e73d3e2',
    RUN / 'execution/score_terminal.json': '0732cbe7e443032908433ddc76e58bfc249bb8497b394a4c7472c228d0536ba0',
    RUN / 'execution/generation_verification.json': '467a64d9489475a84eb5c213a52aa751522f4e4355a5d3fa648533c620daf3ca',
    RUN / 'root_exit_proof.json': 'bb06a4dd0eaa5d45d23f9185b78a5e0e1b04e716396ca5b3382d430bc28beadd',
    RUN / 'root_score_exit_proof.json': '43cded6664202dc265c4c59ac36fa7c6b8be3df4ac5e6d0ca13aa03d317a6ea8',
    FAILED / 'root_failure_exit_proof.json': '57f651458b794352e0c36f3f0772b321cb7c7d12c3c76092e5b7fedea11163b1',
    FAILED / 'plan.json': '523434132597cf923e6428094ddd8056ebe748797c58633fe9bdaa207abf6865',
}
ARMS = ('source', 'question_only')


def artifact(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def complete_gates(score, generation, score_exit, verification):
    assert score['success'] and score['plan_sha256'] == generation['plan_sha256'] == verification['plan_sha256'] == PLAN_SHA
    assert score['API_used'] is False and score['external_official_server_evaluated'] is False
    assert score['model_loaded'] is False and score['CUDA_initialized'] is False and score['GT_read'] is True
    assert generation['root_pty_session'] == 86075 and generation['root_pty_actual_exit_code'] == 0
    assert generation['root_pty_exit_observed_by'] == 'functions.write_stdin' and generation['verifier_container_not_running']
    assert len(generation['process_checks']) == 3
    assert {item['role'] for item in generation['process_checks']} == {'controller', 'worker', 'verifier_client'}
    assert all(item['original_process_absent'] for item in generation['process_checks'])
    assert score_exit['score_root_pty_session'] == 57729 and score_exit['score_root_pty_actual_exit_code'] == 0
    assert score_exit['score_root_exit_observed_by'] == 'functions.write_stdin'
    assert score_exit['score_container_inspect_not_running'] and score_exit['score_container_pid_absent_at_proof']
    assert verification['success'] and verification['n_rows'] == 4377 and verification['n_groups'] == 1176
    assert verification['n_generations'] == verification['processor_selection_and_prefill_checks'] == verification['independent_token_redecodes'] == 8754
    assert verification['n_anchor_rows'] == 60
    assert verification['GT_read'] is False and verification['scoring_started'] is False
    assert verification['high_dimensional_arrays_independently_reconstructed_for_nonanchors'] is False
    for arm in ARMS:
        assert score['runs'][arm]['n_rows'] == 4377 and score['runs'][arm]['n_groups'] == 1176


def build_data():
    for path, expected in LOCKED.items(): assert artifact(path)['sha256'] == expected, str(path)
    assert '独立只读终审通过' in (RUN / 'RESULT.md').read_text()
    plan, state = read(RUN / 'plan.json'), read(RUN / 'execution/state.json')
    score, verification = read(RUN / 'execution/score_terminal.json'), read(RUN / 'execution/generation_verification.json')
    generation, score_exit = read(RUN / 'root_exit_proof.json'), read(RUN / 'root_score_exit_proof.json')
    complete_gates(score, generation, score_exit, verification)
    assert score['root_exit_proof_sha256'] == score_exit['generation_root_proof_sha256'] == LOCKED[RUN / 'root_exit_proof.json']
    assert score_exit['score_report_sha256'] == LOCKED[RUN / 'execution/score_terminal.json']
    assert score['generation_verification_sha256'] == generation['verifier_sha256'] == LOCKED[RUN / 'execution/generation_verification.json']
    assert state['status'] == 'complete' and state['native_exit_code'] == state['verifier_exit_code'] == 0
    assert state['original_worker_exited'] and state['GPU_pid_absent'] and state['verifier_container_absent']
    assert state['plan_sha256'] == PLAN_SHA
    assert artifact(RUN / 'execution/state.json')['sha256'] == generation['state_sha256']
    proof_paths = [('root_gpu_allocation.json', 'allocation_sha256'), ('launch.claim.json', 'claim_sha256'),
                   ('execution/native_release.json', 'native_release_sha256'), ('execution/native_complete.json', 'native_complete_sha256'),
                   ('execution/worker_identity.json', 'worker_identity_sha256'), ('execution/verifier.cid', 'verifier_cid_sha256')]
    for name, key in proof_paths: assert artifact(RUN / name)['sha256'] == generation[key]
    release = read(RUN / 'execution/native_release.json')
    assert release['native_exit_code'] == 0 and release['original_worker_exited'] and release['GPU_pid_absent']
    assert release['plan_sha256'] == PLAN_SHA and release['allocation_sha256'] == state['allocation_sha256'] == generation['allocation_sha256']
    roles = {item['role']: item for item in generation['process_checks']}
    for role in ('controller', 'worker', 'verifier_client'): assert roles[role]['original'] == state[role]
    for name, key in [('native_complete.json', 'native_complete_sha256'), ('native_release.json', 'native_release_sha256'),
                      ('anchor_gate.json', 'anchor_gate_sha256'), ('runtime.json', 'runtime_sha256')]:
        assert artifact(RUN / 'execution' / name)['sha256'] == verification[key]
    anchors, runtime = read(RUN / 'execution/anchor_gate.json'), read(RUN / 'execution/runtime.json')
    assert anchors['passed'] and anchors['n_rows'] == 60 and anchors['n_generations'] == 120 and anchors['all_anchors_before_remaining']
    assert runtime['one_loaded_model'] and runtime['K'] == 128 and runtime['alpha'] == .5 and runtime['beta'] == 1
    assert runtime['model_dtype'] == 'torch.float16' and runtime['attention_implementation'] == 'sdpa'
    assert artifact(RUN / 'inputs/manifest.json')['sha256'] == plan['manifest_sha256']
    assert artifact(RUN / 'inputs/panel.tsv')['sha256'] == plan['panel_sha256']
    assert artifact(RUN / 'score_container.cid')['sha256'] == score_exit['score_container_cid_sha256']
    assert (RUN / 'score_container.cid').read_text().strip() == score_exit['score_container_id']
    failure = read(FAILED / 'root_failure_exit_proof.json')
    assert failure['root_pty_actual_exit_code'] == failure['native_exit_code'] == 1 and failure['no_scoring']
    assert plan['prior_failed_run_root_proof_sha256'] == LOCKED[FAILED / 'root_failure_exit_proof.json']
    assert artifact(FAILED / 'execution/state.json')['sha256'] == failure['state_sha256']
    assert artifact(FAILED / 'execution/worker.log')['sha256'] == failure['worker_log_sha256']
    paper = [row for row in read(OUT / 'comparison.json')['rows'] if row['model_key'] == 'v15' and row['task'] == 'MMB_EN' and row['budget_nominal'] == 128]
    assert len(paper) == 1 and paper[0]['paper_value'] == 62.7
    paper = paper[0]
    rows, groups = [], {}
    for arm, label in [('source', '完整提示CLIP指导（同实例重跑）'), ('question_only', '仅问题CLIP指导（启用原注释边界）')]:
        result = score['runs'][arm]
        assert len(result['rows']) == 4377 and len(result['groups']) == 1176
        assert result['public_correct_groups'] == result['local_correct_groups']
        assert result['public_exact_matching_percent'] == result['local_exact_letter_percent']
        assert all(item['public_hit'] == item['local_hit'] for item in result['groups'])
        assert sum(item['parser_disagrees'] for item in result['rows']) == result['parser_disagreements']
        groups[arm] = {item['group']: item['public_hit'] for item in result['groups']}
        assert len(groups[arm]) == 1176
        references = {}
        for suffix, field, key in [('_predictions.jsonl', 'prediction_sha256', 'raw_prediction'),
                                   ('_observations.jsonl', 'observation_sha256', 'runtime_observations')]:
            reference = artifact(RUN / 'execution' / (arm + suffix))
            assert reference['sha256'] == verification[field][arm]
            with Path(reference['path']).open() as stream: assert sum(bool(line.strip()) for line in stream) == 4377
            references[key] = reference
        ending = verification['ending_counts'][arm]
        assert ending['EOS_present'] + ending['EOS_missing'] == 4377
        value = result['public_exact_matching_percent']
        rows.append(dict(model='LLaVA-1.5-7B', task='MMB_EN', arm=arm, method=label, n_rows=4377, n_groups=1176,
            nominal_tokens=128, paper_value=None, local_minus_paper=None, paper_reference_value=paper['paper_value'],
            diagnostic_minus_paper_reference=str(Decimal(str(value))-Decimal(str(paper['paper_value']))),
            local_value=value, public_exact_matching_value=value, independent_local_letter_value=result['local_exact_letter_percent'],
            correct_groups=result['public_correct_groups'], unresolved_public_groups=result['unresolved_public_groups'],
            parser_disagreements=result['parser_disagreements'], comment_boundary_activated=arm == 'question_only',
            **ending, unit='percentage_points',
            conditions='同实例；旧EN4377行/1176循环组；K128/α.5/β1/q.2/三wait；LLM完整hint和选项保持；仅问题同时改变指导内容与分段，非论文已确认配方', **references))
    assert groups['source'].keys() == groups['question_only'].keys()
    improved = sum(groups['source'][group] == 0 and groups['question_only'][group] == 1 for group in groups['source'])
    declined = sum(groups['source'][group] == 1 and groups['question_only'][group] == 0 for group in groups['source'])
    results = verification['results']; assert len(results) == 4377
    changes = {key: sum(item['comparison'][key] for item in results) for key in ('response_changed','output_ids_changed','final_mask_changed','visual_input_changed')}
    helpers = []
    for item in score['helpers']:
        assert artifact(item['path'])['sha256'] == item['sha256']; helpers.append(item)
    return dict(definition='Same-instance MMB CLIP-guidance diagnostic; separate from paper pruning rows, 12 main runs and all previous calibrations.',
        rows=rows, question_only_minus_source=rows[1]['local_value']-rows[0]['local_value'],
        question_only_minus_source_decimal_from_scores=str(Decimal(str(rows[1]['local_value']))-Decimal(str(rows[0]['local_value']))),
        improved_groups=improved, declined_groups=declined, unchanged_groups=1176-improved-declined,
        changes=changes, group_mask_summary=verification['group_mask_summary'],
        same_model_instance=True, author_comment_active_in_release=False, paper_argv_confirmed=False,
        part_of_eadp_pruning_ledger=False, full_denominators_preserved=True, external_official_server_evaluated=False,
        high_dimensional_arrays_saved_for_nonanchors=False,
        score_container_start_ticks_recorded=score_exit['score_container_host_start_ticks_recorded'],
        score_container_full_PID_identity_reconstructable=False,
        paper_reference=dict(value=paper['paper_value'], model=paper['model'], task=paper['task'], nominal_tokens=128,
                             paper_evidence_id=paper['paper_evidence_id'], reference_index=artifact(OUT / 'comparison.json'), source=artifact(paper['paper_source'])),
        plan=artifact(RUN / 'plan.json'), result_note=artifact(RUN / 'RESULT.md'),
        score_report=artifact(RUN / 'execution/score_terminal.json'), generation_exit_proof=artifact(RUN / 'root_exit_proof.json'),
        score_exit_proof=artifact(RUN / 'root_score_exit_proof.json'), generation_verification=artifact(RUN / 'execution/generation_verification.json'),
        generation_state=artifact(RUN / 'execution/state.json'), generation_release=artifact(RUN / 'execution/native_release.json'),
        generation_runtime=artifact(RUN / 'execution/runtime.json'), generation_allocation=artifact(RUN / 'root_gpu_allocation.json'),
        input_manifest=artifact(RUN / 'inputs/manifest.json'), sanitized_TSV=artifact(RUN / 'inputs/panel.tsv'),
        anchor_gate=artifact(RUN / 'execution/anchor_gate.json'), score_helpers=helpers,
        previous_failed_attempt=dict(plan=artifact(FAILED / 'plan.json'), exit_proof=artifact(FAILED / 'root_failure_exit_proof.json'),
                                     state=artifact(FAILED / 'execution/state.json'), worker_log=artifact(FAILED / 'execution/worker.log'), scored=False),
        builder=artifact(__file__), limits=[
            '仅问题使用作者未启用的注释边界；去除选项、部分行的hint并改变分段，不能分离这些因素、认证论文argv或替换原复现主行。',
            '分数来自作者仓库公开exact-matching辅助函数加本地循环计分，并另列本地独立字母解析；未运行外部官方服务器或judge。',
            '固定60行120次真实数组已锚定，非锚高维张量仅存运行时摘要，无法事后独立重建；观测开销不能作效率测量。',
            '准备期物理读取含GT源TSV字节但仅解析输入列；生成/验收只读净化输入，完整终验和根退出门后才读取GT评分。',
            '首轮尺寸元组/列表检查失败及原始输出已保留，run02仅修该检查；失败不能冒充完整运行。',
            '评分根PTY实际exit0及容器ID/不运行已记录，但短时评分容器PID的start_ticks缺失，不能重建完整PID身份。'])


def main():
    data = build_data()
    (OUT / 'mmb_guidance_diagnostic.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    fields = ['model','task','arm','method','n_rows','n_groups','nominal_tokens','paper_reference_value',
              'public_exact_matching_value','independent_local_letter_value','diagnostic_minus_paper_reference',
              'correct_groups','unresolved_public_groups','parser_disagreements','EOS_present','EOS_missing','at_max_tokens','conditions']
    with (OUT / 'mmb_guidance_diagnostic.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(data['rows'])
    print(json.dumps(dict(MMB_diagnostic_arms=2, n_rows_per_arm=4377, groups_per_arm=1176,
                         question_only_minus_source=data['question_only_minus_source'], main_ledger_modified=False, rescored=False)))


if __name__ == '__main__': main()
