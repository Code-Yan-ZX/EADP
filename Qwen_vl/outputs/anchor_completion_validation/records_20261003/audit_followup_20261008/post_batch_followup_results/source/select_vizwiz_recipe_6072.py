"""Export the user's explicitly selected original VizWiz recipe, all budgets.

The 60.72-generating recipe is applied as a complete six-cell block, including
cells lower than the public-default/stream-wait controls. No maximum selection.
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
TABLE = HERE.parents[1]/'submission_tables_20261009'
GT = ROOT/'LLaVA/playground/data/eval/vizwiz/val.json'
QUESTIONS = ROOT/'LLaVA/playground/data/eval/vizwiz/llava_val.jsonl'
SCORER = ROOT/'LLaVA/scripts/eval_vizwiz.py'
NORMALIZER = ROOT/'LLaVA/llava/eval/m4c_evaluator.py'
SELECTED = HERE/'selected_vizwiz_recipe_6072_20261010.json'
STAMP = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
NOTE = ('VizWiz-val uses the EADP released evaluator, as confirmed by the authors according to the user. '
        'AnchorZip uses the complete original alpha0.5/beta2, full-text-guidance, max1024 recipe at all budgets; '
        'these are original predictions before the stream-wait rerun. K128 in NeXT denotes nominal640. '
        'The alpha0/beta1 public-default EADP/AnchorZip controls are supplementary, not same-recipe baselines for these cells. '
        'Official leave-one-out scores remain auxiliary. Paper generation argv remains unverified.')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, data): Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

current = read(TABLE/'latest_tables.json')
archive = Path(current['vizwiz_scoring_policy']['archive'])
before = read(archive/'latest_tables.json')
if (archive/'shared_public_defaults_primary_snapshot.json').exists():
    if '--resume' not in sys.argv:
        raise RuntimeError('Recipe selection already applied; --resume regenerates from preserved originals.')
else:
    save(archive/'shared_public_defaults_primary_snapshot.json', current)
    (TABLE/'vizwiz_prescribed_repair_controls.csv').write_bytes((TABLE/'vizwiz_author_confirmed.csv').read_bytes())
controls = read(HERE/'author_confirmed_vizwiz_scores_20261010.json')
verification_path = HERE.parent/'vizwiz_scoring_verification_20261009.json'
verification = read(verification_path)
assert sha(SCORER) == verification['released_scorer_sha256']
assert sha(GT) == verification['gt_sha256']
spec = importlib.util.spec_from_file_location('selected_recipe_cpu_normalizer', NORMALIZER)
normalizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer_module)
processor = normalizer_module.EvalAIAnswerProcessor()
gt = read(GT)
questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
assert len(gt) == len(questions) == 4319
assert all(len(q['answers']) == 10 for q in gt)
qby = {q['question_id']: q for q in questions}
assert set(qby) == set(range(4319))
refs = [[processor(a['answer']) for a in q['answers']] for q in gt]
fn = next(n for n in ast.parse(SCORER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
selected = []
for old in verification['rows']:
    budget = old['budget']//5 if old['model']=='next' else old['budget']
    prediction = Path(old['prediction'])
    assert sha(prediction) == old['prediction_sha256']
    answers = [json.loads(line) for line in prediction.read_text().splitlines()]
    assert len(answers) == 4319 and len({a['question_id'] for a in answers}) == 4319
    assert {a['question_id'] for a in answers} == set(qby)
    assert all(a['prompt'] == qby[a['question_id']]['text'] for a in answers)
    model_id = {'v15':'llava-v1.5-7b','next':'llava-v1.6-vicuna-7b'}[old['model']]
    assert all(a['model_id'] == model_id and isinstance(a['text'], str) for a in answers)
    by = {a['question_id']:a['text'] for a in answers}
    per_question = [{'question_id':i, 'acc':min(1., sum(processor(by[i])==a for a in ref)/3.)} for i,ref in enumerate(refs)]
    value = sum(q['acc'] for q in per_question)/4319*100
    assert abs(value-old['released_score']) < 1e-9
    args = SimpleNamespace(annotation_file=str(GT), result_file=str(prediction), visual_token_num=None)
    ns = {'parse_args':lambda:args,'json':json,'EvalAIAnswerProcessor':type(processor)}
    exec(compile(tree,str(SCORER),'exec'),ns)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured): ns['main']()
    assert 'Missing predictions: 0' in captured.getvalue() and f'Accuracy: {value:.2f}%' in captured.getvalue()
    selected.append(dict(old, value=value, budget=budget, budget_nominal=old['budget'], metric='EADP released VizWiz accuracy (%)', per_question=per_question,
                         method='AnchorZip (ours)', record_id=f"{old['model']}_K{budget}_AnchorZip_original_recipe",
                         released_main_stdout=captured.getvalue()))
assert len(selected) == 6
recipe = {'alpha':.5,'beta':2.,'lambda_completion':.25,'temperature':0.,'num_beams':1,'max_new_tokens':1024,
          'conv_mode':'vicuna_v1','generation_entry':'llava_eval_arm_model_vqa.py',
          'pruning_text':'Full question text, including Unanswerable instruction and short-answer suffix',
          'execution_variant':'Original historical predictions, before the stream-wait-repair generation',
          'historical_runtime_limit':'Original prediction metadata is empty; generation settings are established by driver/source audit, not a contemporaneous per-question runtime trace.'}
sources = [SCORER,NORMALIZER,GT,QUESTIONS,verification_path,Path(__file__),
           ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_lane_a_v15.sh',
           ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_lane_b_next.sh',
           ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_eval_arm_model_vqa.py']
policy = copy.deepcopy(current['vizwiz_scoring_policy'])
policy.update(prediction_selection='User explicitly selected the complete original recipe yielding NeXT K128 60.72, across both models and all three budgets; no per-cell maximum selection.',
              selected_recipe=recipe, user_recipe_selection='我就是问你这个60.72是怎么跑出来的，然后用这个的 适配到别的模型就行了。',
              supplementary_shared_public_default_scores=str(HERE/'author_confirmed_vizwiz_scores_20261010.json'))
save(SELECTED, {'updated_beijing':STAMP,'policy':policy,'rows':selected,'sources_sha256':{str(p):sha(p) for p in sources},
                'controls':{'source':str(HERE/'author_confirmed_vizwiz_scores_20261010.json'),'sha256':sha(HERE/'author_confirmed_vizwiz_scores_20261010.json')},
                'same_recipe_local_eadp_baseline_available':False,'errors':[]})
data = copy.deepcopy(before)
data.update(snapshot_beijing=STAMP, selection_policy=before['selection_policy']+' VizWiz override: '+policy['prediction_selection'], vizwiz_scoring_policy=policy)
data['notes'] = [n for n in data['notes'] if not (n.startswith('Local VizWiz primary scores use official') or n.startswith('Completed VizWiz K128 followup'))]
data['notes'].append(NOTE)
data['sources_sha256'].update({str(p):sha(p) for p in sources+[SELECTED,HERE/'author_confirmed_vizwiz_scores_20261010.json']})
for row in data['rows']:
    if row['kind'] != 'local' or 'VizWiz' not in row['cells']: continue
    cell = row['cells']['VizWiz']
    if row['method'] == 'EADP (reproduced)':
        control = next(r for r in controls['rows'] if r['model']==row['model'] and r['budget']==row['budget'] and r['method']==row['method'])
        row['cells']['VizWiz'] = dict(value=None,status='different_generation_recipe_control_only',source=None,
                                    supplementary_control=dict(value=control['value'],source=str(HERE/'author_confirmed_vizwiz_scores_20261010.json'),recipe='alpha0/beta1, native loader, max128, wait'))
    else:
        record = next(r for r in selected if (r['model'],r['budget'],r['method'])==(row['model'],row['budget'],row['method']))
        cell.update(value=record['value'],status='user_selected_original_recipe',source=str(SELECTED),score_record_id=record['record_id'],
                    metric=record['metric'],official_loo_auxiliary=record['official_loo'],prediction_sha256=record['prediction_sha256'])
for old,new in zip(before['rows'],data['rows']):
    assert {k:v for k,v in old['cells'].items() if not(old['kind']=='local' and k=='VizWiz')} == {k:v for k,v in new['cells'].items() if not(new['kind']=='local' and k=='VizWiz')}
save(TABLE/'latest_tables.json',data)
base = (TABLE/'build_tables.py').read_text()
defs = base[base.index('def display('):base.index("snapshot={'snapshot_beijing'")]
exports = base[base.index("with (OUT/'latest_tables.csv')"):base.index('# Old values below')]
exports = exports.replace("tex=[r'\\documentclass", "notes += ' ' + NOTE\ntex=[r'\\documentclass",1)
scope = dict(OUT=TABLE,STAMP=STAMP,rows=data['rows'],csv=csv,html=html,Decimal=Decimal,ROUND_HALF_UP=ROUND_HALF_UP,NOTE=NOTE,
             cell=lambda v,status='complete',source=None:dict(value=v,status=status,source=source),
             LL=['GQA','VizWiz','SQA_IMG','TextVQA','POPE','MME','MMB_EN','MMB_CN','MMVet','Avg7'],
             QW=['TextVQA','ChartQA','AI2D','OCRBench','HallusionBench','MME','MMB_EN','MMB_CN','DocVQA','InfoVQA','Avg10'])
exec(defs+'\n'+exports,scope)
render = (TABLE/'render_tables.py').read_text()
anchor = "        fig.text(.025,.12/height,'\\n'.join(footer)"
insert = """        if model != 'qwen':
            footer = [
                '* VizWiz validation: EADP released scoring, author confirmation reported by user; official LOO kept for audit.',
                'VizWiz ours: all six original alpha0.5/beta2, full-guidance, max1024 predictions, before stream-wait reruns; no per-cell maximum selection.',
                'Public alpha0/beta1 paired controls are supplementary; their EADP scores do not fill the missing same-recipe primary baseline.',
                'SQA: shipped CQM-A, 2017 image questions. POPE: mean F1 (2910/3000/3000). NeXT budgets are nominal.',
                'Avg7 = (GQA + SQA-IMG + TextVQA + POPE-F1 + MME/20 + MMB-EN + MMB-CN)/7; MME is perception raw.',
                'Local Avg7 stays blank for missing/unresolved MMB. Reported baselines are references, not local reproductions.'
            ]
"""
assert render.count(anchor)==1
exec(compile(render.replace(anchor,insert+anchor),str(TABLE/'render_tables.py')+'[user-selected-original-VizWiz-recipe]','exec'),{'__name__':'__main__','__file__':str(TABLE/'render_tables.py')})
with (TABLE/'vizwiz_author_confirmed.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f);writer.writerow(['model','budget_nominal','method','official_loo_auxiliary','eadp_released_primary','prediction','prediction_sha256'])
    for r in selected:writer.writerow([r['model'],r['budget']*(5 if r['model']=='next' else 1),r['method'],r['official_loo'],r['value'],r['prediction'],r['prediction_sha256']])
audit = read(HERE.parent/'three_row_audit.json')
by = {(r['model'],r['budget'],r['method']):r for r in data['rows']}
audit['rows'] = [copy.deepcopy(by[r['model'],r['budget'],r['method']]) for r in audit['rows']]
for r in audit['comparisons']+audit['priority_comparisons']:
    if r['task']!='VizWiz':continue
    for key,method in [('reported','EADP (reported)'),('reproduced','EADP (reproduced)'),('ours','AnchorZip (ours)')]:
        row=by.get((r['model'],r['budget'],method));cell=row['cells'].get('VizWiz') if row else None
        r[key]=copy.deepcopy(cell) if cell and cell.get('value') is not None else None
    r.update(ours_minus_reproduced=None,same_local_numerical_comparison_available=False,numerical_status='same_recipe_local_baseline_missing_public_control_separate')
    r.pop('reproduced_minus_reported_pp',None)
audit['shared_public_default_paired_statistics_auxiliary']=audit.pop('paired_statistics')
audit['paired_statistics']={}
audit['released_formula_primary']={'source':str(SELECTED),'rows':[{k:v for k,v in r.items() if k!='per_question'} for r in selected]}
audit.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),table_sha256=sha(TABLE/'latest_tables.json'),table_pdf_sha256=sha(TABLE/'latest_tables.pdf'),vizwiz_scoring_policy=policy)
audit['source_sha256'].update(data['sources_sha256'])
audit['presentation']['numerical_change_scope']='User-selected original six-cell VizWiz recipe block; different-recipe EADP public-default controls moved to supplementary audit. Other tasks unchanged.'
counts={model:dict(higher=0,lower=0,equal=0) for model in ('v15','next','qwen')}
for r in audit['comparisons']:
    if r['task'].startswith('Avg') or r.get('ours_minus_reproduced') is None:continue
    delta=r['ours_minus_reproduced'];counts[r['model']]['higher' if delta>0 else 'lower' if delta<0 else 'equal']+=1
audit['descriptive_paired_cell_counts']=counts
save(HERE.parent/'three_row_audit.json',audit)
md=['# 用户指定60.72生成配置的VizWiz结果','',f'更新：{STAMP}。用户明确指定采用产生60.72的原始recipe，统一用于两个LLaVA模型全部三个预算。评分采用作者确认的EADP发布评分器。','',
    '|模型|名义预算|EADP reported|AnchorZip（指定配置）|','|---|---:|---:|---:|']
for r in selected:
    paper=by[r['model'],r['budget'],'EADP (reported)']['cells']['VizWiz']['value']
    md.append(f"|{r['model']}|{r['budget']*(5 if r['model']=='next' else 1)}|{paper:.2f}|{r['value']:.2f}|")
md += ['', '配置：alpha0.5/beta2、lambda.25、temperature0、单beam、vicuna_v1、完整文本guidance、max_new_tokens1024；使用已有4319题完整原预测（stream-wait修复前）。NeXT640/320/160对应K128/64/32，v1.5预算为128/64/32。', '',
       '60.72来自旧NeXT K128同一答案文件：官方LOO59.254457→EADP发布评分60.716215。全六格均重新核验题ID/题面/模型名/GT/SHA，并逐项调用未改动EADP main验证。', '',
       '该选择整体保留原recipe结果，包含低于后续控制结果的格；没有逐格取最大值。alpha0/beta1公开默认配对和统计保存在author_confirmed_vizwiz_scores_20261010.json及本审计JSON的supplementary字段，不充当原recipe的同协议EADP基线。', '',
       '原recipe本机EADP VizWiz基线缺失；主表相应reproduced格置空，reported行保持论文值。其他数据集所有数值及Avg保持原值。旧预测缺 contemporaneous runtime metadata，不能把原recipe叫作完成stream-wait修复后的运行；作者确认仅覆盖评分器。', '',
       '原75+后续5 GPU臂均已完成，本次没有启动GPU。未登记CN/逐crop与旧Qwen来源边界继续保留。', '',
       f'逐题评分与来源：{SELECTED}。']
(HERE.parent/'three_row_audit.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'selected_recipe':recipe,'scores':[{k:r[k] for k in ('model','budget','value')} for r in selected],'other_tasks_unchanged':True},ensure_ascii=False,indent=2))
