"""CPU-only switch to the author-confirmed EADP VizWiz evaluator.

Use the prescribed predictions already selected in the final table. Never select
predictions or scoring formulas by their maximum. Preserve official LOO files,
frozen experiment sources, and an exact archive of the previous presentation.
"""
import ast
import contextlib
import copy
import csv
import datetime
import hashlib
import html
import importlib.util
import io
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/media/disk2/YZX/research/EADP_amp')
HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[1] / 'submission_tables_20261009'
GT = ROOT / 'LLaVA/playground/data/eval/vizwiz/val.json'
SCORER = ROOT / 'LLaVA/scripts/eval_vizwiz.py'
NORMALIZER = ROOT / 'LLaVA/llava/eval/m4c_evaluator.py'
SCORES = HERE / 'author_confirmed_vizwiz_scores_20261010.json'
POLICY = HERE / 'author_confirmed_vizwiz_policy_20261010.json'
STAMP = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
NOTE = ('VizWiz-val follows the EADP released evaluator: normalize predictions and all ten answers, '
        'then average min(matches/3, 1). The user reports author confirmation by email on 2026-10-10. '
        'Official leave-one-annotator-out scores are retained separately for audit. '
        'K128 (NeXT nominal640) uses the shared public-default alpha0/beta1 pair; '
        'other budgets retain their recorded recipes. Confirmation concerns the scorer, not all generation arguments.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

assert sha(GT) == 'f96a6a42da56fa067d023ec25a841df840d6c5989b1b0e7b0e6f0d269c617ab7'
assert sha(SCORER) == '6d92bfefff9fbea9cb49ea85b63ce6ec6d1fd57dba4eed08d26cceef86179b26'
assert sha(NORMALIZER) == '96c4dcf88e7a94cb48b40db713b02d5da2f18e92814d82f4b8e796e21b690da7'
before = read(TABLE / 'latest_tables.json')
if before.get('vizwiz_scoring_policy'):
    if '--resume' not in sys.argv:
        raise RuntimeError('Already applied; --resume regenerates from the exact archived original.')
    archive = Path(before['vizwiz_scoring_policy']['archive'])
    before = read(archive / 'latest_tables.json')
else:
    archive = HERE / ('author_confirmed_scoring_archive_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    archive.mkdir()
    archived = {}
    for path in sorted(TABLE.glob('latest_tables.*')) + sorted(TABLE.glob('table_*.png')) + [TABLE/'core_tables.txt', HERE.parent/'three_row_audit.json', HERE.parent/'three_row_audit.md']:
        if path.exists():
            (archive / path.name).write_bytes(path.read_bytes())
            archived[str(path)] = sha(path)
    save(archive / 'manifest.json', {'archived_beijing': STAMP, 'sources_sha256': archived})
policy = {'effective_beijing': STAMP, 'scope': 'All available local VizWiz-val cells, all models and budgets; same evaluator for EADP and AnchorZip.',
          'author_confirmation': {'reported_by': 'user', 'date': '2026-10-10', 'channel': 'email', 'email_independently_inspected': False,
                                  'user_statement': '包括其他模型的都换成这个高的评分，作者用的就是这个，我发邮件问了。'},
          'primary_metric': 'EADP released VizWiz accuracy (%)', 'formula': '100 * mean(min(1, count(normalize(pred)==normalize(answer))/3))',
          'normalization': 'EADP EvalAIAnswerProcessor on predictions and all ten reference answers',
          'auxiliary_metric': 'official VizWiz leave-one-annotator-out accuracy (%)',
          'prediction_selection': before['selection_policy'], 'reported_rows': 'Preserve paper-reported values; no fabricated rescoring without predictions.',
          'generation_arguments_confirmed': False, 'archive': str(archive), 'GPU_generation': False,
          'scorer_source': str(SCORER), 'scorer_sha256': sha(SCORER), 'normalizer_sha256': sha(NORMALIZER),
          'source_url': 'https://github.com/SJTU-DeepVisionLab/EADP/blob/e1a08801461c181871b9ff1cb0803b8e9966f083/LLaVA/scripts/eval_vizwiz.py'}
save(POLICY, policy)
normalizer = module(NORMALIZER, 'author_confirmed_cpu_normalizer').EvalAIAnswerProcessor()
gt = read(GT)
assert len(gt) == 4319 and all(len(x['answers']) == 10 for x in gt)
references = [[normalizer(a['answer']) for a in item['answers']] for item in gt]
fn = next(n for n in ast.parse(SCORER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
rows = []
for row in before['rows']:
    if row['kind'] != 'local' or 'VizWiz' not in row['cells'] or row['cells']['VizWiz']['value'] is None:
        continue
    cell = row['cells']['VizWiz']
    original_path = Path(cell['source'])
    original = read(original_path)
    assert before['sources_sha256'][str(original_path)] == sha(original_path)
    prediction = original.get('inputs', {}).get('predictions', {}).get('path')
    if prediction is None:
        assert row['model'] == 'v15' and row['budget'] in (32, 64)
        arm = {32: 'LRMAIN00625', 64: 'LRMAIN0125'}[row['budget']]
        prediction = str(ROOT / ('LLaVA/playground/data/eval/anchorzip_p3/vizwiz/' + arm + '.jsonl'))
    prediction = Path(prediction)
    if original.get('prediction_sha256'):
        assert original['prediction_sha256'] == sha(prediction)
    answers = [json.loads(line) for line in prediction.read_text().splitlines()]
    assert len(answers) == 4319 and len({a['question_id'] for a in answers}) == 4319
    assert {a['question_id'] for a in answers} == set(range(4319))
    assert all(isinstance(a['text'], str) for a in answers)
    by_id = {a['question_id']: a['text'] for a in answers}
    per_question = [{'question_id': i, 'acc': min(1., sum(normalizer(by_id[i]) == a for a in refs)/3.)} for i, refs in enumerate(references)]
    value = 100 * sum(x['acc'] for x in per_question) / 4319
    args = SimpleNamespace(annotation_file=str(GT), result_file=str(prediction), visual_token_num=None)
    ns = {'parse_args': lambda: args, 'json': json, 'EvalAIAnswerProcessor': type(normalizer)}
    exec(compile(tree, str(SCORER), 'exec'), ns)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        ns['main']()
    assert 'Evaluated 4319 samples.' in output.getvalue() and 'Missing predictions: 0' in output.getvalue()
    assert 'Accuracy: ' + format(value, '.2f') + '%' in output.getvalue()
    official = 100 * sum(x['acc'] for x in original['per_question']) / 4319
    assert abs(official - cell['value']) < 1e-9
    record = {'record_id': f"{row['model']}_K{row['budget']}_{row['method'].split(' (')[0]}",
              'model': row['model'], 'budget': row['budget'], 'method': row['method'], 'n': 4319,
              'prediction': str(prediction), 'prediction_sha256': sha(prediction), 'original_official_score': str(original_path),
              'original_official_score_sha256': sha(original_path), 'official_loo': official, 'value': value,
              'metric': policy['primary_metric'], 'released_main_stdout': output.getvalue(), 'per_question': per_question,
              'control_protocol_sha256': original.get('control_protocol_sha256'),
              'generation_recipe': 'alpha0/beta1, native loader, max128, wait' if row['budget'] == 128 else 'Original recorded recipe; no new generation',
              'primary_prediction_selection_changed': False}
    rows.append(record)
assert len(rows) == 8
historical = read(HERE.parent/'vizwiz_scoring_verification_20261009.json')
for old in historical['rows']:
    assert sha(old['prediction']) == old['prediction_sha256']
historical_note = 'Historical runs are retained for comparison only. NeXT K128 60.716215 (60.72) is the old alpha0.5/beta2 run, not the latest shared alpha0/beta1 pair.'
paired = {}
comparison = module(HERE/'pair_statistics.py', 'author_confirmed_paired_statistics')
for model in ('v15', 'next'):
    reports = []
    for method, arm in [('EADP', 'E128'), ('AnchorZip', 'AZ128')]:
        record = next(r for r in rows if r['model'] == model and r['budget'] == 128 and r['method'].startswith(method))
        score = {k: record[k] for k in ('n', 'value', 'per_question', 'prediction_sha256', 'control_protocol_sha256')}
        score.update(method=method, task='vizwiz')
        protocol = read(HERE / f'{model}_vizwiz_{arm}_streamwait.control.protocol.json')
        reports.append({'score': score, 'protocol': protocol})
    result = comparison.compare(reports)
    result['metric'] = policy['primary_metric']
    result['interpretation'] = 'Same shared-recipe local comparison using the author-confirmed scorer; generation argv identity to the paper remains unverified.'
    paired[f'followup_{model}_vizwiz_K128_release'] = result
save(SCORES, {'updated_beijing': STAMP, 'policy': str(POLICY), 'policy_sha256': sha(POLICY),
              'source_sha256': {str(p): sha(p) for p in (SCORER, NORMALIZER, GT, Path(__file__))},
              'rows': rows, 'paired_statistics': paired, 'historical_previous_runs': historical['rows'],
              'historical_note': historical_note, 'errors': []})
data = copy.deepcopy(before)
data['snapshot_beijing'] = STAMP
data['vizwiz_scoring_policy'] = policy
data['notes'] = [n for n in data['notes'] if not n.startswith('Local VizWiz primary scores use official')]
data['notes'].append(NOTE)
data['sources_sha256'].update({str(p): sha(p) for p in (SCORER, NORMALIZER, GT, SCORES, POLICY, Path(__file__))})
for row in data['rows']:
    for record in rows:
        if (row['model'], row['budget'], row['method']) == (record['model'], record['budget'], record['method']):
            cell = row['cells']['VizWiz']
            cell.update(value=record['value'], source=str(SCORES), score_record_id=record['record_id'],
                        metric=policy['primary_metric'], official_loo_auxiliary=record['official_loo'],
                        official_loo_source=record['original_official_score'], prediction_sha256=record['prediction_sha256'])
for old, new in zip(before['rows'], data['rows']):
    assert {k:v for k,v in old['cells'].items() if not (old['kind']=='local' and k=='VizWiz')} == {k:v for k,v in new['cells'].items() if not (new['kind']=='local' and k=='VizWiz')}
save(TABLE/'latest_tables.json', data)

# Reuse only the frozen formatting/export section; do not execute its score selection.
base = (TABLE/'build_tables.py').read_text()
assert sha(TABLE/'build_tables.py') == '1b718ec81ce246af17174682f413843005c28740d0d8174de52c0b3009fd92a8'
defs = base[base.index('def display('):base.index("snapshot={'snapshot_beijing'")]
exports = base[base.index("with (OUT/'latest_tables.csv')"):base.index('# Old values below')]
exports = exports.replace("tex=[r'\\documentclass", "notes += ' ' + NOTE\ntex=[r'\\documentclass", 1)
assert "notes += ' ' + NOTE" in exports
scope = dict(OUT=TABLE, STAMP=STAMP, rows=data['rows'], csv=csv, html=html, Decimal=Decimal,
             ROUND_HALF_UP=ROUND_HALF_UP, NOTE=NOTE,
             cell=lambda v,status='complete',source=None:dict(value=v,status=status,source=source),
             LL=['GQA','VizWiz','SQA_IMG','TextVQA','POPE','MME','MMB_EN','MMB_CN','MMVet','Avg7'],
             QW=['TextVQA','ChartQA','AI2D','OCRBench','HallusionBench','MME','MMB_EN','MMB_CN','DocVQA','InfoVQA','Avg10'])
exec(defs + '\n' + exports, scope)
render = (TABLE/'render_tables.py').read_text()
plan = read(HERE/'followup.plan.json')
assert sha(TABLE/'render_tables.py') == plan['source_and_input_sha256'][str(TABLE/'render_tables.py')]
anchor = "        fig.text(.025,.12/height,'\\n'.join(footer)"
assert render.count(anchor) == 1
insert = """        if model != 'qwen':
            footer = [
                '— Missing or unresolved protocol. * VizWiz validation; EADP released evaluator is used for all local VizWiz scores.',
                'VizWiz: mean min(matches/3, 1), EADP answer normalization; author confirmation reported on 2026-10-10. Official LOO retained for audit.',
                'K128 (NeXT nominal640): shared alpha0/beta1 public defaults; other budgets retain recorded recipes. Paper generation argv unverified.',
                'SQA: shipped CQM-A, 2017 image questions. POPE: mean F1 (2910/3000/3000). NeXT budgets are nominal.',
                'Avg7 = (GQA + SQA-IMG + TextVQA + POPE-F1 + MME/20 + MMB-EN + MMB-CN)/7; MME is perception raw.',
                'Local Avg7 stays blank for missing/unresolved MMB. Reported baselines are references, not local reproductions.'
            ]
"""
exec(compile(render.replace(anchor, insert+anchor), str(TABLE/'render_tables.py')+'[author-confirmed-VizWiz]', 'exec'), {'__name__':'__main__','__file__':str(TABLE/'render_tables.py')})

audit_path = HERE.parent/'three_row_audit.json'
audit = read(archive/'three_row_audit.json')
audit['official_loo_paired_statistics_auxiliary'] = copy.deepcopy(audit['paired_statistics'])
audit['paired_statistics'].update(paired)
audit['released_formula_primary'] = {'policy': str(POLICY), 'scores': str(SCORES), 'rows': [{k:v for k,v in r.items() if k!='per_question'} for r in rows]}
audit.pop('released_formula_auxiliary', None)
selected = {(r['model'],r['budget'],r['method']):r for r in data['rows']}
audit['rows'] = [copy.deepcopy(selected[r['model'],r['budget'],r['method']]) for r in audit['rows']]
for comparison_row in audit['comparisons'] + audit['priority_comparisons']:
    if comparison_row['task'] != 'VizWiz':
        continue
    for field, method in [('reported','EADP (reported)'),('reproduced','EADP (reproduced)'),('ours','AnchorZip (ours)')]:
        row = selected.get((comparison_row['model'], comparison_row['budget'], method))
        comparison_row[field] = copy.deepcopy(row['cells'].get('VizWiz')) if row else None
    e, a = comparison_row['reproduced'], comparison_row['ours']
    comparison_row['ours_minus_reproduced'] = a['value']-e['value'] if e and a else None
    comparison_row['scorer_author_confirmed_by_user'] = True
    if 'numerical_status' in comparison_row and e:
        delta = Decimal(str(e['value']))-Decimal(str(comparison_row['reported']['value']))
        comparison_row['reproduced_minus_reported_pp'] = str(delta)
        comparison_row['numerical_status'] = 'within_0.2pp_or_higher' if delta >= Decimal('-.2') else 'below_reported_over_0.2pp'
audit['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
audit['table_sha256'] = sha(TABLE/'latest_tables.json')
audit['table_pdf_sha256'] = sha(TABLE/'latest_tables.pdf')
audit['source_sha256'].update(data['sources_sha256'])
audit['vizwiz_scoring_policy'] = policy
audit['necessary_followup']['vizwiz'] = 'Completed shared public-recipe pairs; EADP released score now primary after user-reported author email confirmation; official LOO auxiliary.'
audit['unresolved'] = ['Authors actual task-specific generation argv unavailable; VizWiz released scorer confirmed by user-reported author email.' if x=='Authors actual task-specific argv/scorer unavailable.' else x for x in audit['unresolved']]
audit['presentation'] = {'numerical_change_scope':'Only 8 local VizWiz cells rescored from identical predictions; all other cells unchanged.',
                         'author_confirmed_primary_metric_visible_in_all_exports':True, 'frozen_build_and_render_sources_unchanged':True}
audit['archived_previous_audit'] = str(archive/'three_row_audit.json')
save(audit_path, audit)
md = ['# 完整批次与后续三行最终审计（作者确认的VizWiz评分）', '',
      f'更新：{STAMP}。原75臂/55组与后续5臂/3组已完成。本次仅CPU重评分，原预测、官方LOO文件和冻结实验源码保持不变。', '',
      '用户于2026-10-10告知已邮件确认作者采用EADP发布评分器。此新证据取代此前“作者scorer未知”的判断；邮件内容未由代理独立查看。所有本机VizWiz格统一采用该评分器，官方留一分数保留作辅审计。', '',
      '|模型|预算参数|任务|reported E|reproduced E|ours|ours−local E（pp）|', '|---|---:|---|---:|---:|---:|---:|']
for r in audit['priority_comparisons']:
    def number(c): return '缺失' if not c else f"{c['value']:.9f}"
    delta = '—' if r['ours_minus_reproduced'] is None else f"{r['ours_minus_reproduced']:+.9f}"
    md.append(f"|{r['model']}|{r['budget']}|{r['task']}|{number(r['reported'])}|{number(r['reproduced'])}|{number(r['ours'])}|{delta}|")
md += ['', 'NeXT参数K128/64/32对应名义640/320/160；VizWiz全部4319题。主分为EADP预测与十份GT归一化后的min(matches/3,1)平均。', '',
       '固定seed图像cluster bootstrap（2000次，seed20261009）；VizWiz区间按新主指标重新计算：']
for group, r in audit['paired_statistics'].items():
    lo, hi = r['bootstrap']['ci95_pp']
    md.append(f"- {group}：AZ−E {r['delta_pp']:+.9f}pp，95%CI [{lo:+.9f}, {hi:+.9f}]。")
md += ['', 'K128的双方仍是预定义alpha0/beta1/native loader/max128/wait配对，其他预算保留原recipe。评分器确认不等于全部paper生成参数确认。两模型EADP K128 VizWiz主分已在reported的0.2pp范围内；方法配对差与CI如实呈现。', '',
       '旧NeXT K128的60.72对应旧alpha0.5/beta2预测；最新同协议E/AZ主分为60.677626/60.476962。旧值单独留档，不按分数择优覆盖新配对。', '',
       '其他数据集所有数值和Avg保持原值，缺失低预算EADP格保持缺失。MMB-CN语言、NeXT跨crop assignment和旧Qwen来源限制继续披露。', '',
       '论文表注建议：VizWiz-val is evaluated with the released EADP evaluator, consistent with the protocol confirmed by its authors.', '',
       f'评分与逐题来源：{SCORES}。旧表归档：{archive}。']
(HERE.parent/'three_row_audit.md').write_text('\n'.join(md)+'\n')
with (TABLE/'vizwiz_author_confirmed.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.writer(stream)
    writer.writerow(['model','budget_nominal','method','official_loo_auxiliary','eadp_released_primary','prediction','prediction_sha256'])
    for r in rows:
        writer.writerow([r['model'],r['budget']*(5 if r['model']=='next' else 1),r['method'],r['official_loo'],r['value'],r['prediction'],r['prediction_sha256']])
print(json.dumps({'archive':str(archive), 'updated_cells':len(rows), 'other_numerical_cells_unchanged':True,
                  'primary_scores':[{k:r[k] for k in ('model','budget','method','value')} for r in rows],
                  'paired_statistics':paired}, ensure_ascii=False, indent=2))
