#!/usr/bin/env python3
"""Render a CPU-only snapshot of accepted historical and new complete results.

This consumes the independently checked ledger. It never reads predictions,
scores a running prefix, chooses a maximum, or edits the original comparison.
"""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RUNS = ROOT / 'reproduction/20261028'


def artifact(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def conditions(row):
    sync = '补三处CUDA同步' if row.get('three_wait_nodes') else '原公开代码'
    if row.get('attention_backend'):
        sync = row['attention_backend']
        if row['task'] == 'AI2D':
            sync += '；作者API失败后精确匹配回退，论文judge尚未复现'
    if row.get('input_hypothesis_only'):
        sync += '；仅问句CLIP指导假设；LLM完整OCR不变；paper argv未确认'
    if row.get('configuration_hypothesis_only'):
        sync += '；固定β单点对照；paper argv未确认'
    return f"α{row['alpha']:g}／β{row['beta']:g}；{sync}"


def main():
    history_path = OUT / 'comparison.json'
    ledger_path = RUNS / 'full_run_ledger.json'
    if (RUNS / 'full_run_ledger_v2.json').exists():
        ledger_path = RUNS / 'full_run_ledger_v2.json'
    if (RUNS / 'full_run_ledger_v3.json').exists():
        ledger_path = RUNS / 'full_run_ledger_v3.json'
    if (RUNS / 'full_run_ledger_v4.json').exists():
        ledger_path = RUNS / 'full_run_ledger_v4.json'
    history = json.loads(history_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    assert len(history['rows']) == 99
    accepted = sum(r['local_value'] is not None for r in history['rows'])
    assert accepted == 29
    for row in history['rows']:
        if row['local_value'] is not None:
            assert Decimal(str(row['local_value']))-Decimal(str(row['paper_value'])) == Decimal(row['local_minus_paper'])
    for row in ledger['full_runs']:
        assert Decimal(str(row['value']))-Decimal(str(row['paper'])) == Decimal(row['local_minus_paper'])
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    snapshot = dict(snapshot_beijing=now,
        definition='Difference=local-paper; historical and new complete runs retained separately; no max selection or missing-component average.',
        sources=[artifact(history_path), artifact(ledger_path), artifact(Path(__file__).resolve())],
        paper_cells=99, qualified_historical_pairs=accepted, new_complete_runs=len(ledger['full_runs']),
        historical_rows=history['rows'], new_full_runs=ledger['full_runs'],
        pending_full_runs=ledger['pending_full_runs'], verification_failed_full_runs=ledger['verification_failed_full_runs'],
        goal_complete=ledger['goal_complete'])
    calibration_path = OUT / 'full_path_calibrations.json'
    calibrations = json.loads(calibration_path.read_text())['rows'] if calibration_path.exists() else []
    if calibrations:
        snapshot['full_path_calibrations'] = calibrations
        snapshot['sources'].append(artifact(calibration_path))
    routes_path = OUT / 'qwen_doc_full_two_routes_calibration.json'
    routes = None
    if routes_path.exists():
        routes = json.loads(routes_path.read_text())
        assert routes['paper_argv_confirmed'] is False and routes['part_of_eadp_pruning_ledger'] is False
        assert len(routes['rows']) == 2 and [r['arm'] for r in routes['rows']] == ['plain','eadp1024']
        for key in ('score_report','generation_exit_proof','score_exit_proof','generation_verification'):
            reference = routes[key]
            assert artifact(Path(reference['path'])) == reference
        report = json.loads(Path(routes['score_report']['path']).read_text())
        generation_exit = json.loads(Path(routes['generation_exit_proof']['path']).read_text())
        score_exit = json.loads(Path(routes['score_exit_proof']['path']).read_text())
        assert report['success'] and report['n']==5349 and report['full_denominators_preserved']
        assert generation_exit['root_owned_exec_actual_exit_code']==0 and generation_exit['all_original_processes_exited']
        assert score_exit['root_owned_score_exec_actual_exit_code']==0
        assert score_exit['generation_root_proof_sha256']==routes['generation_exit_proof']['sha256']
        assert score_exit['score_report_sha256']==routes['score_report']['sha256']
        for row in routes['rows']:
            assert row['n']==5349 and row['paper_value']==94.5
            assert row['independent_value']==report['scores'][row['arm']]['independent']
            assert row['official_value']==report['scores'][row['arm']]['official']
            assert row['local_value']==row['independent_value']
            assert Decimal(str(row['local_value']))-Decimal(str(row['paper_value']))==Decimal(row['local_minus_paper'])
            assert artifact(Path(row['raw_prediction']['path']))==row['raw_prediction']
        delta=Decimal(str(routes['rows'][1]['local_value']))-Decimal(str(routes['rows'][0]['local_value']))
        assert delta==Decimal(routes['eadp1024_minus_plain_percentage_points'])
        assert abs(float(delta)-report['eadp1024_minus_plain'])<1e-12
        snapshot['qwen_doc_full_two_routes_calibration']=routes
        snapshot['sources'].append(artifact(routes_path))
    ds_path = OUT / 'qwen_doc_full_ds_toggles_calibration.json'
    ds_toggles = None
    if ds_path.exists():
        ds_toggles = json.loads(ds_path.read_text())
        assert ds_toggles['paper_argv_confirmed'] is False and ds_toggles['part_of_eadp_pruning_ledger'] is False
        assert ds_toggles['conditional_cross_run_comparison'] is True and ds_toggles['high_dimensional_arrays_saved'] is False
        assert len(ds_toggles['rows']) == 2 and [r['arm'] for r in ds_toggles['rows']] == ['plain_ds_off','eadp1024_ds_on']
        for key in ('plan','result_note','score_report','generation_exit_proof','score_exit_proof',
                    'generation_verification','generation_release','generation_runtime',
                    'baseline_score_report','baseline_generation_exit_proof','baseline_score_exit_proof','paired_rows','builder'):
            reference = ds_toggles[key]
            assert artifact(Path(reference['path'])) == reference
        report = json.loads(Path(ds_toggles['score_report']['path']).read_text())
        baseline = json.loads(Path(ds_toggles['baseline_score_report']['path']).read_text())
        generation_exit = json.loads(Path(ds_toggles['generation_exit_proof']['path']).read_text())
        score_exit = json.loads(Path(ds_toggles['score_exit_proof']['path']).read_text())
        assert report['success'] and report['n'] == 5349 and report['full_denominators_preserved']
        assert generation_exit['root_owned_exec_actual_exit_code'] == 0 and generation_exit['all_original_processes_exited']
        assert score_exit['root_owned_score_exec_actual_exit_code'] == 0 and score_exit['scorer_container_not_running']
        assert score_exit['generation_root_proof_sha256'] == ds_toggles['generation_exit_proof']['sha256']
        assert score_exit['score_report_sha256'] == ds_toggles['score_report']['sha256']
        for row in ds_toggles['rows']:
            assert row['n'] == 5349 and row['nominal_tokens'] == 1024
            assert row['paper_value'] is None and row['local_minus_paper'] is None
            assert row['local_value'] == row['independent_value'] == report['scores'][row['arm']]['independent']
            assert row['official_value'] == report['scores'][row['arm']]['official']
            assert row['baseline_value'] == baseline['scores'][row['baseline_arm']]['independent']
            delta = Decimal(str(row['local_value']))-Decimal(str(row['baseline_value']))
            assert delta == Decimal(row['toggle_minus_baseline'])
            assert abs(float(delta)-report['per_arm_comparison'][row['arm']]['score_difference']) < 1e-12
            for key in ('raw_prediction','runtime_observations','pointwise_scores','baseline_pointwise_scores'):
                assert artifact(Path(row[key]['path'])) == row[key]
        snapshot['qwen_doc_full_ds_toggles_calibration'] = ds_toggles
        snapshot['sources'].append(artifact(ds_path))
    k256_path = OUT / 'qwen_doc_k256_ds_diagnostic.json'
    k256_diagnostic = None
    if k256_path.exists():
        k256_diagnostic = json.loads(k256_path.read_text())
        assert k256_diagnostic['paper_argv_confirmed'] is False and k256_diagnostic['part_of_eadp_pruning_ledger'] is False
        assert k256_diagnostic['same_model_instance'] and k256_diagnostic['legacy_position_policy']
        assert k256_diagnostic['high_dimensional_arrays_saved_for_nonanchors'] is False
        assert len(k256_diagnostic['rows']) == 2 and [row['arm'] for row in k256_diagnostic['rows']] == ['ds_off', 'ds_on']
        for key in ('plan','result_note','score_report','generation_exit_proof','score_exit_proof',
                    'generation_verification','generation_release','generation_runtime','generation_state',
                    'generation_launch','generation_allocation','score_launch','paired_rows','builder'):
            reference = k256_diagnostic[key]
            assert artifact(Path(reference['path'])) == reference
        report = json.loads(Path(k256_diagnostic['score_report']['path']).read_text())
        generation_exit = json.loads(Path(k256_diagnostic['generation_exit_proof']['path']).read_text())
        score_exit = json.loads(Path(k256_diagnostic['score_exit_proof']['path']).read_text())
        assert report['success'] and report['n'] == 5349 and report['full_denominators_preserved']
        assert generation_exit['root_owned_exec_actual_exit_code'] == 0 and generation_exit['all_original_processes_exited']
        assert score_exit['score_root_pty_actual_exit_code'] == 0 and score_exit['container_not_running']
        assert score_exit['generation_root_proof_sha256'] == k256_diagnostic['generation_exit_proof']['sha256']
        assert score_exit['score_report_sha256'] == k256_diagnostic['score_report']['sha256']
        for key in ('reference_index','source'):
            reference = k256_diagnostic['paper_reference'][key]
            assert artifact(Path(reference['path'])) == reference
        assert k256_diagnostic['paper_reference']['value'] == 62.8
        for row in k256_diagnostic['rows']:
            assert row['n'] == 5349 and row['nominal_tokens'] == 256
            assert row['paper_value'] is None and row['local_minus_paper'] is None
            assert row['author_implementation_modified'] is (row['arm'] == 'ds_on')
            assert row['paper_reference_value'] == k256_diagnostic['paper_reference']['value']
            assert row['local_value'] == row['independent_value'] == report['scores'][row['arm']]['independent']
            assert row['official_value'] == report['scores'][row['arm']]['official']
            assert Decimal(str(row['local_value']))-Decimal(str(row['paper_reference_value'])) == Decimal(row['diagnostic_minus_paper_reference'])
            for key in ('raw_prediction','runtime_observations','ordered_prediction','ordered_observations',
                        'pointwise_scores','native_score_input','native_score_detail'):
                assert artifact(Path(row[key]['path'])) == row[key]
        assert k256_diagnostic['ds_on_minus_off'] == report['ds_on_minus_off']
        exact_delta = Decimal(str(k256_diagnostic['rows'][1]['local_value']))-Decimal(str(k256_diagnostic['rows'][0]['local_value']))
        assert exact_delta == Decimal(k256_diagnostic['ds_on_minus_off_decimal_from_scores'])
        assert abs(float(exact_delta)-report['ds_on_minus_off']) < 1e-10
        assert k256_diagnostic['changed_answers'] == report['changed_answers']
        snapshot['qwen_doc_k256_ds_implementation_diagnostic'] = k256_diagnostic
        snapshot['sources'].append(artifact(k256_path))
    mmb_path = OUT / 'mmb_guidance_diagnostic.json'
    mmb_diagnostic = None
    if mmb_path.exists():
        mmb_diagnostic = json.loads(mmb_path.read_text())
        assert mmb_diagnostic['same_model_instance'] and not mmb_diagnostic['author_comment_active_in_release']
        assert mmb_diagnostic['paper_argv_confirmed'] is False and mmb_diagnostic['part_of_eadp_pruning_ledger'] is False
        assert not mmb_diagnostic['external_official_server_evaluated'] and not mmb_diagnostic['high_dimensional_arrays_saved_for_nonanchors']
        assert mmb_diagnostic['score_container_start_ticks_recorded'] is False
        assert not mmb_diagnostic['score_container_full_PID_identity_reconstructable']
        assert [row['arm'] for row in mmb_diagnostic['rows']] == ['source','question_only']
        for key in ('plan','result_note','score_report','generation_exit_proof','score_exit_proof','generation_verification',
                    'generation_state','generation_release','generation_runtime','generation_allocation',
                    'input_manifest','sanitized_TSV','anchor_gate','builder'):
            reference = mmb_diagnostic[key]
            assert artifact(Path(reference['path'])) == reference
        for key in ('reference_index','source'):
            reference = mmb_diagnostic['paper_reference'][key]
            assert artifact(Path(reference['path'])) == reference
        for key in ('plan','exit_proof','state','worker_log'):
            reference = mmb_diagnostic['previous_failed_attempt'][key]
            assert artifact(Path(reference['path'])) == reference
        assert mmb_diagnostic['previous_failed_attempt']['scored'] is False
        report = json.loads(Path(mmb_diagnostic['score_report']['path']).read_text())
        generation_exit = json.loads(Path(mmb_diagnostic['generation_exit_proof']['path']).read_text())
        score_exit = json.loads(Path(mmb_diagnostic['score_exit_proof']['path']).read_text())
        assert report['success'] and report['external_official_server_evaluated'] is False
        assert generation_exit['root_pty_actual_exit_code'] == 0 and all(item['original_process_absent'] for item in generation_exit['process_checks'])
        assert score_exit['score_root_pty_actual_exit_code'] == 0 and score_exit['score_container_inspect_not_running']
        assert score_exit['score_report_sha256'] == mmb_diagnostic['score_report']['sha256']
        assert score_exit['generation_root_proof_sha256'] == report['root_exit_proof_sha256'] == mmb_diagnostic['generation_exit_proof']['sha256']
        for row in mmb_diagnostic['rows']:
            assert row['n_rows'] == 4377 and row['n_groups'] == 1176 and row['nominal_tokens'] == 128
            assert row['paper_value'] is None and row['local_minus_paper'] is None
            assert row['paper_reference_value'] == mmb_diagnostic['paper_reference']['value'] == 62.7
            assert row['local_value'] == row['public_exact_matching_value'] == report['runs'][row['arm']]['public_exact_matching_percent']
            assert row['independent_local_letter_value'] == report['runs'][row['arm']]['local_exact_letter_percent']
            assert Decimal(str(row['local_value']))-Decimal(str(row['paper_reference_value'])) == Decimal(row['diagnostic_minus_paper_reference'])
            for key in ('raw_prediction','runtime_observations'):
                assert artifact(Path(row[key]['path'])) == row[key]
        difference = mmb_diagnostic['rows'][1]['local_value']-mmb_diagnostic['rows'][0]['local_value']
        assert difference == mmb_diagnostic['question_only_minus_source']
        snapshot['mmb_clip_guidance_implementation_diagnostic'] = mmb_diagnostic
        snapshot['sources'].append(artifact(mmb_path))
    doc_path = RUNS / 'qwen/independent_audit/docvqa_preterminal_20261028.json'
    preliminary = None
    if doc_path.exists() and not any(r['run_id'] == 'Qwen_author_FA2_DocVQA_VAL' for r in ledger['full_runs']):
        preliminary = json.loads(doc_path.read_text())
        assert preliminary['n'] == 5349 and preliminary['prediction_validity']['coverage_complete']
        assert preliminary['formal_ledger_eligible'] is False
        snapshot['DocVQA_complete_but_whole_sequence_pending'] = dict(evidence=artifact(doc_path), report=preliminary)
    (OUT / 'current_comparison.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
    fields = ['scope','model','task','nominal_tokens','paper_value','local_value','local_minus_paper','unit','run_id','conditions','raw_prediction']
    with (OUT / 'current_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in history['rows']:
            writer.writerow(dict(scope='historical',model=row['model'],task=row['task'],nominal_tokens=row['budget_nominal'],
                paper_value=row['paper_value'],local_value=row['local_value'],local_minus_paper=row['local_minus_paper'],
                unit=row['delta_unit'],run_id=row['local_evidence_id'],conditions=json.dumps(row['protocol_caveat'],ensure_ascii=False),
                raw_prediction=json.dumps(row['raw_prediction'],ensure_ascii=False)))
        for row in ledger['full_runs']:
            writer.writerow(dict(scope='new_complete_run',model=row['model'],task=row['task'],nominal_tokens=row['token_budget'],
                paper_value=row['paper'],local_value=row['value'],local_minus_paper=row['local_minus_paper'],unit=row['unit'],
                run_id=row['run_id'],conditions=conditions(row),raw_prediction=row['predictions']))
    content=(OUT / 'comparison.html').read_text()
    content=content.replace('<h1>EADP V2：论文与本机复现对比</h1>',
        f'<h1>EADP V2：论文与本机复现对比</h1><p>当前快照：{html.escape(now)}。以下保留29组历史合格配对，末尾单列{len(ledger["full_runs"])}次新完整运行。原始comparison保持不变。</p>')
    extra=['<h2>新完成并通过完整终端核验的重跑</h2><table><thead><tr><th>模型／任务／条件</th><th>名义tokens</th><th>论文</th><th>本机</th><th>差值</th><th>题数</th></tr></thead><tbody>']
    for row in ledger['full_runs']:
        label=html.escape(f"{row['model']}／{row['task']}／{conditions(row)}")
        extra.append(f"<tr><td>{label}</td><td>{row['token_budget']}</td><td>{row['paper']:g}</td><td>{row['value']:.3f}</td><td>{Decimal(row['local_minus_paper']):+.3f}</td><td>{row['n']}</td></tr>")
    extra.append('</tbody></table><p>MMB新三次运行均为旧EN4377行/1176循环组，与历史V11分母不同；公开匹配helper已复核，外部服务器judge未执行。POPE原代码重复输入冲突231/2334，补同步后的两次为0/2334；所有运行均保留。作者主表完整argv尚未闭合。</p>')
    if any(row.get('attention_backend') for row in ledger['full_runs']):
        extra.append('<p>Qwen新完整运行采用作者公开FA2配方，DocVQA5349题、AI2D3088题；全部进程实际退出及整段终端核验通过。AI2D为原生API检查连接失败后源码精确匹配回退，与独立can_infer逐题一致；不声称论文API judge已复现。历史Doc生成时的确切包版本/backend未闭合，当前旧环境的安装状态不能替代当时记录。</p>')
    if preliminary:
        extra.append(f'<p>DocVQA完整5349题另已核验：原生ANLS {preliminary["native_ANLS_percent"]:.6f}，论文62.8，差{Decimal(preliminary["local_minus_paper_percentage_points"]):+.6f}。AI2D及整个原生序列终端门仍待完成，因此暂未加入上表正式台账；空答0、失败0。此项是完整Doc结果，未计算AI2D前缀分。</p>')
    if calibrations:
        extra.append('<h2>FULL基线对照</h2><p>单列用于排查基础路径与剪枝损失，不替代EADP剪枝结果；历史基线的运行条件见备注。</p><table><thead><tr><th>模型／任务</th><th>论文FULL</th><th>本机FULL</th><th>差值</th><th>条件</th></tr></thead><tbody>')
        for row in calibrations:
            extra.append(f'<tr><td>{html.escape(row["model"]+"／"+row["task"])}</td><td>{row["paper_value"]:g}</td><td>{row["local_value"]:.3f}</td><td>{Decimal(row["local_minus_paper"]):+.3f}</td><td>{html.escape(row["conditions"])}</td></tr>')
        if routes:
            for row in routes['rows']:
                label=html.escape(row['model']+'／'+row['task']+'／'+row['method'])
                extra.append(f'<tr><td>{label}</td><td>{row["paper_value"]:g}</td><td>{row["local_value"]:.3f}</td><td>{Decimal(row["local_minus_paper"]):+.3f}</td><td>{html.escape(row["conditions"])}</td></tr>')
        extra.append('</tbody></table><p><a href="full_path_calibrations.csv">原FULL基线CSV</a> · <a href="full_path_calibrations.json">原FULL来源与条件JSON</a></p>')
        if routes:
            extra.append(f'<p>Qwen DocVQA两条新完整路径为同模型、同FA2环境下的5349题配对校准：EADP K1024全token路径减原生多模态FULL为 {Decimal(routes["eadp1024_minus_plain_percentage_points"]):+.3f} 个百分点，{routes["changed_answers"]}题原回答不同。K1024是显式构造、作者未注册的诊断配置；两条路径同时存在多处差异，不能单独归因DeepStack或mRoPE，也不确认论文主表参数。旧Qwen FULL基线仍独立保留。</p>')
            extra.append('<p><a href="qwen_doc_full_two_routes_calibration.json">Qwen双FULL路径数据与来源</a></p>')
        if ds_toggles:
            extra.append('<h3>Qwen DocVQA：同路径 DeepStack 开关诊断</h3><p>各5349题、全部1024视觉tokens；仅首次prefill联合开关DeepStack与对应visual mask。下列差值为开关减同路径已完成基线，不是论文差值，也不加入12次剪枝主运行。</p><table><thead><tr><th>路径／开关</th><th>原基线ANLS</th><th>开关后ANLS</th><th>变化（点）</th><th>回答变化</th><th>条件</th></tr></thead><tbody>')
            for row in ds_toggles['rows']:
                extra.append(f'<tr><td>{html.escape(row["method"])}</td><td>{row["baseline_value"]:.6f}</td><td>{row["local_value"]:.6f}</td><td>{Decimal(row["toggle_minus_baseline"]):+.6f}</td><td>{row["changed_answers"]}／{row["n"]}</td><td>{html.escape(row["conditions"])}</td></tr>')
            extra.append('</tbody></table><p>旧全量基线未保存全部逐题内部张量与数值标志，只有固定16端点和变体跨运行锚定；这是有条件的同路径跨运行对照。新全量高维张量只留运行时摘要，不能事后独立重建；干预可自然改变后续KV与生成。两条API仍有其他区别，不能用这两个差值完整分摊总路径差，也不能外推K256或确认论文原配方。</p>')
            extra.append('<p><a href="qwen_doc_full_ds_toggles_calibration.csv">DeepStack诊断CSV</a> · <a href="qwen_doc_full_ds_toggles_calibration.json">DeepStack诊断数据、退出证明及来源SHA</a></p>')
    if k256_diagnostic:
        extra.append('<h2>Qwen DocVQA K256：同实例实现诊断</h2><p>各5349题、同一模型实例、同一实际256空间选择；仅首次prefill联合开关DeepStack与对应visual mask，legacy位置策略保持。此表独立于论文剪枝行、12次主运行和FULL校准；DS-on修改作者公开实现，不能称论文实际复现或新的剪枝方法。</p><table><thead><tr><th>路径／实现</th><th>论文数值参考</th><th>诊断ANLS</th><th>相对参考（仅数值）</th><th>题数</th><th>条件</th></tr></thead><tbody>')
        for row in k256_diagnostic['rows']:
            extra.append(f'<tr><td>{html.escape(row["method"])}</td><td>{row["paper_reference_value"]:g}</td><td>{row["local_value"]:.6f}</td><td>{Decimal(row["diagnostic_minus_paper_reference"]):+.6f}</td><td>{row["n"]}</td><td>{html.escape(row["conditions"])}</td></tr>')
        extra.append(f'</tbody></table><p>本次同实例DS-on减DS-off：{k256_diagnostic["ds_on_minus_off"]:+.6f} 个百分点；{k256_diagnostic["changed_answers"]}／5349题原回答改变，逐题ANLS提高{k256_diagnostic["improved"]}、降低{k256_diagnostic["declined"]}、不变{k256_diagnostic["unchanged"]}。原公开K256复现61.166042与论文62.8的主行保持；DS-on高于论文只表示数值距离，不确认paper argv。</p>')
        extra.append('<p>没有同时恢复原生mRoPE，后续KV与生成可随干预自然变化；不把分歧后的状态当同条件。固定16题的32份真实数组已锚定，其余5333对高维张量仅保存运行时摘要，不能事后独立重建。旧全量内部状态未逐题保存；本次off分数相同不证明所有旧内部量相同。两臂各4条无EOS且达到2048上限、NA转换分别7／6，均保留全部5349分母；观测开销不作为效率结果。</p>')
        extra.append('<p><a href="qwen_doc_k256_ds_diagnostic.csv">K256实现诊断CSV</a> · <a href="qwen_doc_k256_ds_diagnostic.json">K256实现诊断数据、评分与退出来源SHA</a></p>')
    if mmb_diagnostic:
        extra.append('<h2>MMB K128：同实例CLIP指导内容诊断</h2><p>各4377行、1176完整循环组；同一LLaVA-1.5-7B、K128／α.5／β1／q.2、三次wait，完整LLM提示保持。仅问题臂启用作者公开源码未启用的注释边界，去除选项、部分行的hint并改变分段，不是单一长度控制或已确认的论文原配方；此表不覆盖论文主行、12次主运行或已有校准。</p><table><thead><tr><th>指导内容</th><th>论文数值参考</th><th>公开exact＋本地循环组</th><th>独立字母＋循环组</th><th>相对参考（仅数值）</th><th>正确组</th></tr></thead><tbody>')
        for row in mmb_diagnostic['rows']:
            extra.append(f'<tr><td>{html.escape(row["method"])}</td><td>{row["paper_reference_value"]:g}</td><td>{row["public_exact_matching_value"]:.6f}</td><td>{row["independent_local_letter_value"]:.6f}</td><td>{Decimal(row["diagnostic_minus_paper_reference"]):+.6f}</td><td>{row["correct_groups"]}／{row["n_groups"]}</td></tr>')
        extra.append(f'</tbody></table><p>仅问题减完整指导：{mmb_diagnostic["question_only_minus_source"]:+.6f} 个百分点；{mmb_diagnostic["improved_groups"]}组改善、{mmb_diagnostic["declined_groups"]}组退步，净增加{mmb_diagnostic["improved_groups"]-mmb_diagnostic["declined_groups"]}组。{mmb_diagnostic["changes"]["response_changed"]}／4377回答及输出IDs改变，{mmb_diagnostic["changes"]["final_mask_changed"]}行最终mask及视觉输入改变；source有{mmb_diagnostic["group_mask_summary"]["source"]["varied"]}／1176组mask随轮换变化，仅问题臂全部1176组内mask恒定。8754输出均含EOS、未到1024上限，全部分母保留。</p>')
        extra.append('<p>上述公开exact为作者仓库辅助函数加本地循环计分；未运行外部官方服务器或judge，不能声称等同论文实际评分服务。固定60行的120次真实数组已锚定，非锚高维仅保存运行时摘要，无法事后独立重建。准备期读取过含GT原TSV字节但只解析输入列，生成和验收使用净化输入，完整终验及根退出后才读取GT评分。</p>')
        extra.append('<p>首轮尺寸元组／JSON列表检查失败及真实输出保留；run02仅修该检查。评分根PTY实际exit0和容器ID／不运行可证，但短时评分容器PID的start_ticks未记录，不能重建完整PID身份。此诊断不能确认论文argv，也不替代原公开路径结果。</p>')
        extra.append('<p><a href="mmb_guidance_diagnostic.csv">MMB指导诊断CSV</a> · <a href="mmb_guidance_diagnostic.json">MMB指导诊断来源、失败记录与退出证明SHA</a></p>')
    extra.append(f'<p>台账尚待完整核验：{len(ledger["pending_full_runs"])}项；核验失败：{len(ledger["verification_failed_full_runs"])}项。该快照不宣称整体复现成功，运行状态需重新检查真实进程。</p>')
    extra.append('<p><a href="current_comparison.csv">当前完整CSV</a> · <a href="current_comparison.json">当前来源与状态JSON</a></p>')
    (OUT / 'current_comparison.html').write_text(content.replace('</html>','\n'.join(extra)+'</html>'))
    with (OUT / 'current_comparison.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==99+len(ledger['full_runs'])
    assert sum(bool(row['local_value']) for row in rows)==accepted+len(ledger['full_runs'])
    print(json.dumps(dict(historical_pairs=accepted,new_complete_runs=len(ledger['full_runs']),pending=len(ledger['pending_full_runs']),goal_complete=ledger['goal_complete'])))


if __name__ == '__main__':
    main()
