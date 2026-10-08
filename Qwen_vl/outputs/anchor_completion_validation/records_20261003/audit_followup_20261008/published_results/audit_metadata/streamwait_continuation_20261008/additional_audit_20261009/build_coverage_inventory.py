"""Read-only CPU coverage inventory; writes only beside this script."""
import ast
import csv
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path('/media/disk2/YZX/research/EADP_amp')
OUT=ROOT/'Qwen_vl/outputs/audit_followup_20261008'
HERE=Path(__file__).parent
P3=ROOT/'LLaVA/playground/data/eval/anchorzip_p3'
SCRIPTS=ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot'
QROOT=ROOT/'Qwen_vl/outputs'
RECOVERY=Path('/media/disk2/YZX/research/audit_base_gap_20261003/archive_recovery/20261003_r2')
ARTIFACTS={}
COUNTS=dict(TextVQA=5000,SQA_IMG=2017,MME=2374,GQA=12578,POPE=8910,VizWiz=4319,
            MMB_EN=4876,MMB_CN=4876,VQAv2=447793,MMVet=218)
TASKS=dict(textvqa='TextVQA',sqa='SQA_IMG',mme='MME',gqa='GQA',pope='POPE',vizwiz='VizWiz',
           mmben='MMB_EN',mmbcn='MMB_CN',vqav2='VQAv2',mmvet='MMVet')
QDS=dict(TextVQA='TextVQA_VAL',ChartQA='ChartQA_TEST',AI2D='AI2D_TEST',OCRBench='OCRBench',
         HallusionBench='HallusionBench',MME='MME',MMB_EN='MMBench_DEV_EN_V11',MMB_CN='MMBench_DEV_CN_V11',
         DocVQA='DocVQA_VAL',InfoVQA='InfoVQA_VAL',MMStar='MMStar',RealWorldQA='RealWorldQA',POPE='POPE')
QN=dict(TextVQA=5000,ChartQA=2500,AI2D=3088,OCRBench=1000,HallusionBench=951,MME=2374,
        MMB_EN=4876,MMB_CN=4876,DocVQA=5349,InfoVQA=2801,MMStar=1500,RealWorldQA=765,POPE=9000)
QM=dict(TextVQA='official VQA accuracy (%)',ChartQA='official relaxed accuracy (%)',AI2D='official exact-match accuracy (%)',
        OCRBench='OCRBench raw score (maximum1000)',HallusionBench='HallusionBench fAcc (%)',MME='MME perception+cognition total raw points',
        MMB_EN='MMBench circular accuracy (%)',MMB_CN='MMBench circular accuracy (%)',DocVQA='official ANLS (%)',
        InfoVQA='official ANLS (%)',MMStar='official accuracy (%)',RealWorldQA='official accuracy (%)',POPE='native official POPE F1 (%)')

def read(path):return json.loads(Path(path).read_text())
def artifact(path,role='evidence'):
    path=Path(path).absolute();key=str(path)
    if not path.is_file():return dict(path=key,exists=False,role=role)
    if key not in ARTIFACTS:
        before=path.stat();h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        after=path.stat()
        ARTIFACTS[key]=dict(path=key,exists=True,sha256=h.hexdigest(),bytes=after.st_size,
                            stable_during_read=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    return dict(ARTIFACTS[key],role=role)

def scan(path):
    path=Path(path)
    if not path.is_file():return dict(exists=False,n=0,invalid_lines=0)
    before=path.stat();bad=0
    if path.suffix=='.jsonl':
        rows=[]
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:bad+=1
        keys=[(str(row.get('question_id',row.get('id'))),row.get('prompt',row.get('text'))) for row in rows]
        result=dict(n=len(rows),unique_composite_keys=len(set(keys)),
                    unique_question_ids=len({str(row.get('question_id',row.get('id'))) for row in rows}),invalid_lines=bad,
                    failure_sentinel=sum(str(row.get('text','')).strip().upper()=='FAILED' for row in rows))
    elif path.suffix=='.xlsx':
        import openpyxl
        book=openpyxl.load_workbook(path,read_only=True,data_only=True);sheet=book.active
        it=sheet.iter_rows(values_only=True);columns=next(it);rows=list(it);book.close()
        result=dict(n=len(rows),columns=list(columns),invalid_lines=0)
    elif path.suffix=='.tsv':
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream,delimiter='\t'))
        result=dict(n=len(rows),unique_index=len({row.get('index') for row in rows}),columns=list(rows[0]) if rows else [],invalid_lines=0)
    else:
        value=read(path);rows=value.get('records',{})
        result=dict(n=len(rows),unique_record_keys=len(rows),invalid_lines=0,metadata=value.get('meta',{}))
    after=path.stat()
    result.update(exists=True,stable_during_read=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                  prediction=artifact(path,'raw_prediction'))
    return result

paper=read(OUT/'paper_reference.json')
latest=read(OUT/'latest_method_comparison.json')
summary=read(OUT/'rerun_batch/repaired_results_summary.json')
inventory=read(OUT/'streamwait_continuation_20261008/legacy_affected_inventory.json')
by_old={job['jobid']:job for job in inventory['jobs']}
cells={}
for model,pipeline,tasks,budgets,table in [
    ('v15','llava',list(COUNTS),['FULL',128,64,32],1),('next','llava',list(COUNTS),['FULL',128,64,32],2),
    ('qwen','official_legacy',list(QDS)[:10],['FULL',512,256,128,64,32],4),
    ('qwen','native_restored',list(QDS),['FULL',512,256,128,64,32],4)]:
    for task in tasks:
        for budget in budgets:
            table_data=next(t for t in paper['tables'] if t['table']==table)
            pr=next((r for r in table_data['rows'] if r['method']==('FULL' if budget=='FULL' else 'EADP')
                     and (budget=='FULL' or r['budget_per_crop_equivalent']==budget)),None)
            cells[(model,pipeline,task,budget)]=dict(model=model,pipeline=pipeline,benchmark=task,budget=budget,
                local_method_required='FULL' if budget=='FULL' else 'EADP',expected_prediction_n=(QN if model=='qwen' else COUNTS)[task],
                metric_denominator_n=1292 if task in ('MMB_EN','MMB_CN') else None,
                paper_score=pr['values'].get(task) if pr else None,paper_source_url=pr['source_url'] if pr else None,
                paper_table_reference_only=(pipeline=='native_restored'),completed_local=[],unscored_or_partial_local=[],
                registered_local_reruns=[],anchorzip_evidence=[],caveats=[])

def add_local(cell,path,stage,score_path=None,score=None,metric=None,expected=None,metadata=None,inputs=(),entry=None):
    raw=scan(path);n=expected or cell['expected_prediction_n']
    evidence=dict(stage=stage,raw=raw,expected_n=n,score=score,metric=metric,
                  official_score_artifact=artifact(score_path,'official_score') if score_path else None,
                  question_and_source_evidence=[artifact(p,'question_or_source') for p in inputs],
                  native_entry=artifact(entry,'native_entry') if entry else None,parameters=metadata or {})
    complete=raw.get('exists') and raw['n']==n and not raw.get('invalid_lines') and not raw.get('failure_sentinel') and raw.get('stable_during_read')
    evidence['complete_raw']=bool(complete)
    evidence['full_score_backed']=bool(complete and score is not None and score_path and Path(score_path).is_file())
    (cell['completed_local'] if evidence['full_score_backed'] else cell['unscored_or_partial_local']).append(evidence)
    return evidence

# Latest complete repaired TextVQA / CQM-A SQA. Read actual full predictions and
# check their summary SHA, rather than turning a queue count into coverage.
lane_jobs={}
for name in ['lane1_v15_plan.json','lane2_next_plan.json']:
    plan_path=OUT/'rerun_batch'/name;plan=read(plan_path)
    for job in plan['jobs']:lane_jobs[job['output']]=(job,plan_path)
for row in summary['rows']:
    if row['method'] not in ('EADP','FULL'):continue
    key=(row['model'],'llava',TASKS[row['task']],'FULL' if row['method']=='FULL' else row['budget_parameter'])
    cell=cells[key];output=Path(row['output']);job,plan_path=lane_jobs.get(row['output'],({},OUT/'rerun_batch/rerun_schedule_manifest.json'))
    protocol_path=Path(str(output)+'.protocol.json');protocol=read(protocol_path) if protocol_path.is_file() else {}
    score_path=(output.with_suffix('.result.json') if row['task']=='sqa' else Path(job.get('score_log',str(output)+'.score.txt')))
    if not score_path.is_file():score_path=OUT/'rerun_batch/repaired_results_summary.json'
    ev=add_local(cell,output,'repaired_input_or_official_entry',score_path,row.get('accuracy') if row['status']=='complete' else None,
                 'ScienceQA IMG accuracy (%)' if row['task']=='sqa' else 'official TextVQA accuracy (%)',
                 metadata=protocol,inputs=[job['question_file']] if job.get('question_file') else [],
                 entry=job.get('generate_command',[None,None])[1] if job else None)
    ev['summary_declared_complete']=row['status']=='complete'
    ev['summary_prediction_sha_matches']=not row.get('prediction_sha256') or ev['raw'].get('prediction',{}).get('sha256')==row['prediction_sha256']
    ev['protocol_artifact']=artifact(protocol_path,'generation_protocol')
    ev['frozen_plan']=artifact(plan_path,'frozen_plan')
    ev['official_score_sources']=[artifact(p,'GT_or_official_evaluator') for p in row.get('score_sources',{})]
    if not ev['summary_prediction_sha_matches']:raise ValueError('Complete summary source changed')
    if row['model']=='next':cell['caveats'].append('Original complete run remains exposed to AnyRes stream dependency; wait-only rerun registered separately.')

# Old local E/FULL: inventory recovers original recipe; original external env and
# process argv are not retrospectively known. VizWiz/MMB have no v15 local E.
registry={(r['table'],r['task'],r['method'],r['budget_total_nominal']):r for r in paper['local_source_registry']}
for model,prefix,table in [('v15','',1),('next','next_',2)]:
    for task,short in [(value,key) for key,value in TASKS.items()]:
        for budget,arm in [('FULL','FULL'),(128,'EGATHER')]:
            cell=cells[(model,'llava',task,budget)]
            if task in ('TextVQA','SQA_IMG'):continue # latest evidence supersedes these raw formats below
            path=P3/(prefix+short)/(arm+'.jsonl')
            if not path.is_file():continue
            recipe=by_old.get(f'{model}_{short}_{arm}',{})
            nominal=(576 if model=='v15' else 2880) if budget=='FULL' else (128 if model=='v15' else 640)
            reg=registry.get((table,task,'FULL' if budget=='FULL' else 'EADP',nominal))
            sc=reg['source'] if reg else None
            qfile=recipe.get('native_question_file')
            if not qfile and task=='VQAv2':qfile=str(ROOT/'LLaVA/playground/data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl')
            if not qfile and task=='MMVet':qfile=str(ROOT/'LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl')
            ev=add_local(cell,path,'historical_legacy_entry',sc,reg['value'] if reg else None,reg.get('metric') if reg else None,
                      metadata=recipe.get('parameters',{}),inputs=[qfile] if qfile else [],entry=recipe.get('native_entry'))
            if recipe:
                ev['official_scoring_source']=artifact(recipe['score_entry'],'official_scoring_source')
                ev['official_GT_evidence']=[dict(artifact(item['path'],'official_GT'),
                    matches_inventory_sha256=artifact(item['path']).get('sha256')==item.get('sha256')) for item in recipe.get('score_data',[])]
                ev['historical_recipe_sources']=[dict(artifact(item['path'],'historical_recipe_source'),
                    matches_inventory_sha256=artifact(item['path']).get('sha256')==item.get('sha256')) for item in recipe.get('source_evidence',[])]
            if recipe:cell['caveats'].append('Historical recipe recovered; external env/actual process argv not recorded. Wait repair pending.')
            if task=='MMVet':cell['caveats'].append('Complete 218 predictions require corresponding official GPT evaluation; unrelated bundled model grades are not this run.')
            if task=='VQAv2':cell['caveats'].append('Historical driver uses test2015 (447793 questions), not a completed official submission; partial/corrupted output does not reproduce the paper.')

# Preserve old CQM-I evidence alongside the completed CQM-A repaired protocol.
for model,prefix,table in [('v15','',1),('next','next_',2)]:
    for budget,arm in [('FULL','FULL'),(128,'EGATHER')]:
        path=P3/(prefix+'sqa')/(arm+'_CQMI_vicuna.jsonl');result=path.with_name(arm+'_CQMI_vicuna_result.json')
        if not path.is_file() or not result.is_file():continue
        data=read(result);add_local(cells[(model,'llava','SQA_IMG',budget)],path,'historical_CQM_I',result,
            None,'ScienceQA IMG (historical CQM-I; not promoted from All score)',expected=4241,
            inputs=[ROOT/'LLaVA/playground/data/eval/scienceqa/problems.json'])

# Fixed full POPE E32 control, actual complete predictions and exact official score.
directory=OUT/'rerun_fast';path=directory/'pope_v15_EADP32_streamwait.jsonl';sc=directory/'pope_v15_EADP32_streamwait.official.score.json'
if path.is_file() and sc.is_file():
    data=read(sc);fair=read(directory/'pope_v15_EADP32_streamwait.fair_comparison.json')
    value=fair['scores']['EADP32']['summary']['average_f1'] if 'summary' in fair['scores']['EADP32'] else None
    if value is None:value=sum(200*v['TP']/(2*v['TP']+v['FP']+v['FN']) for v in fair['scores']['EADP32']['categories'].values())/3
    add_local(cells[('v15','llava','POPE',32)],path,'completed_wait_only_control',sc,value,'POPE mean category F1 (%)',
        metadata=read(directory/'pope_v15_EADP32_streamwait.protocol.json'),
        inputs=[ROOT/'LLaVA/playground/data/eval/pope/llava_pope_test.jsonl'],entry=directory/'run_pope_streamwait_control.py')

# Actual registration is separate from completed EADP. Files can be partial while
# the supervisor is waiting/running; never assign their accuracy.
for stem,kind in [('next_gap_diagnosis_20261008/stability/next_text_full_streamwait.controller','k32'),
                  ('streamwait_continuation_20261008/streamwait_repair_continuation','phase1'),
                  ('streamwait_continuation_20261008/legacy_streamwait_continuation','phase2')]:
    sp=OUT/(stem+'.state.json');pp=OUT/(stem+'.plan.json')
    if not sp.is_file() or not pp.is_file():continue
    state=read(sp);plan=read(pp)
    if not state.get('started_utc'):continue
    for job_id,job in plan['jobs'].items():
        if kind=='k32':
            method='AnchorZip' if job_id=='AZ_beta2' else 'EADP';model='next';task='TextVQA';budget=32
        else:method=job.get('method', 'AnchorZip' if '_AZ' in job_id else 'FULL' if job_id.endswith('FULL') else 'EADP');model=job.get('model','next');task=TASKS[job.get('task','textvqa')];budget='FULL' if method=='FULL' else job.get('budget',job.get('parameters',{}).get('visual_token_num'))
        cell=cells[(model,'llava',task,budget)];paths=job['output_paths'];raw=scan(paths['prediction'])
        record=dict(method=method,job_id=job_id,attempt=job['attempt'],registered_started_utc=state['started_utc'],
             supervisor_status=state['status'],job_status=state.get('jobs',{}).get(job_id,{}).get('status'),raw=raw,
             finished_exists=Path(paths['finished']).is_file(),score_exists=Path(paths['score']).is_file(),
             frozen_plan=artifact(pp,'frozen_plan'),control_protocol=artifact(job['control_protocol'],'registered_control_protocol'),
             expected_n=5000 if kind=='k32' else job.get('n',COUNTS[task]),parameters=job['parameters'])
        if method in ('EADP','FULL'):cell['registered_local_reruns'].append(record)
        else:cell['anchorzip_evidence'].append(dict(record,stage='registered_AZ_rerun'))

# Qwen legacy E256: 3 archived independent official-wrapper runs + 7 full new
# legacy runs. Native BASE is a different engine and is not a legacy substitute.
p1=read(QROOT/'stage1_roundtrip_pilot/legacy_full/p1_unified_20261008.json')
for task,ds in list(QDS.items())[:10]:
    cell=cells[('qwen','official_legacy',task,256)]
    if task in ('TextVQA','DocVQA','OCRBench'):
        path=RECOVERY/'old_predictions'/('old_'+ds+'.xlsx');sc=RECOVERY/'old_rescored.json'
        value=p1['avg10']['eg_columns'][task];ev=add_local(cell,path,'archived_official_EADP_recovered',sc,value,
          QM[task],metadata=dict(K=256,alpha=.5,beta=2.,resolution=1024,deepstack=False,position='1d'),
          inputs=[RECOVERY/'archive_manifest.json',RECOVERY/'provenance_old_run.md',RECOVERY/'rescore_old_predictions.py'])
        ev['archive_manifest_source_hash_verified']=artifact(path)['sha256'] in [v['sha256'] for v in read(RECOVERY/'archive_manifest.json')['sources'].values()]
    else:
        d=QROOT/'stage1_roundtrip_pilot/legacy_full/acc/L_E_GATHER';path=d/(ds+'.json');sc=d/(ds+'_score.json')
        ev=add_local(cell,path,'full_official_legacy_EADP',sc,p1['avg10']['eg_columns'][task],
          QM[task],
          inputs=[d/(ds+'_pred.tsv'),SCRIPTS/'egather_p1_driver.sh',SCRIPTS/'egather_p1_analyze.py',SCRIPTS/'p1_p4_recheck.py'],entry=SCRIPTS/'egather_legacy.py')
        ev['parameters']=ev['raw'].get('metadata',{})
        bank=QROOT/'anchor_completion_validation'/('bank_full_'+ds+'.json.gz')
        if bank.is_file():
            ev['selection_bank']=artifact(bank,'EADP_selection_bank')
            ev['recorded_bank_hash_matches']=ev['selection_bank']['sha256']==ev['parameters'].get('bank_sha256')
            if not ev['recorded_bank_hash_matches']:
                cell['caveats'].append('Complete raw and official score exist, but recorded compressed selection-bank SHA differs from current bank; generation-time bank identity is not closed. Resume writes meta only when absent. A gzip-byte mismatch alone does not prove changed token indices or an incorrect score.')
    cell['caveats'].extend(['Legacy = DeepStack off / 1d positions; native restored results cannot replace this cell.',
      'Paper authors actual complete runtime argv unverified; local full results do not establish all paper configurations reproduced.'])
    dataset=Path('/media/disk2/YZX/LMUData')/(ds+'.tsv')
    cell['current_dataset_reference']=dict(path=str(dataset),exists=dataset.is_file(),
        bytes=dataset.stat().st_size if dataset.is_file() else None,
        note='Current dataset path is a reference; a current file is not automatically proof of generation-time bytes. Prediction XLSX/TSV carries full question/answer rows and verified SHA.')

for pipeline,arm in [('official_legacy','FULL_HALLB'),('native_restored','FULL_HALLB_NATIVE')]:
    d=QROOT/'stage1_roundtrip_pilot/legacy_full/acc'/arm;sc=d/'HallusionBench_score.json'
    score=read(sc)['official']['fAcc']['0']
    ev=add_local(cells[('qwen',pipeline,'HallusionBench','FULL')],d/'HallusionBench.json','keep_all_engine_control',sc,score,
                  'HallusionBench fAcc (%)',inputs=[d/'HallusionBench_pred.tsv'])
    ev['parameters']=ev['raw']['metadata'];cells[('qwen',pipeline,'HallusionBench','FULL')]['caveats'].append(
       'Pipeline chosen by actual deepstack/position metadata, not the official_legacy string carried by both files.')

for panel in ('main','nonreg'):
    d=QROOT/'anchor_completion_validation/acc'/panel/'BASE/K256'
    for task,ds in QDS.items():
        path=d/(ds+'.json');sc=d/(ds+'_score.json')
        if not path.is_file() or not sc.is_file():continue
        data=read(sc);value=data.get('official',{}).get('Overall')
        if task=='OCRBench':value=data['official']['Final Score']
        elif task=='MMB_EN':value=data['official'].get('Overall')
        if isinstance(value,str):value=ast.literal_eval(value)[0]
        if isinstance(value,dict):value=value.get('0',value.get(0))
        if value is not None and value<=1 and task!='OCRBench':value*=100
        ev=add_local(cells[('qwen','native_restored',task,256)],path,'native_BASE_EADP_independent',sc,value,
          QM[task]+'; native auxiliary engine',inputs=[d/(ds+'_pred.tsv')],
          entry=ROOT/'Qwen_vl/scripts/anchor_completion_validation/acu_accuracy.py')
        ev['parameters']=ev['raw'].get('metadata',{})
        bank=QROOT/'anchor_completion_validation'/('bank_full_'+ds+'.json.gz')
        if bank.is_file():
            ev['selection_bank']=artifact(bank,'EADP_selection_bank')
            ev['recorded_bank_hash_matches']=ev['selection_bank']['sha256']==ev['parameters'].get('bank_sha256')
        if task=='POPE':cells[('qwen','native_restored',task,256)]['caveats'].append(
          'Raw/shard/TSV has5127 unique questions, whereas headline official category metrics may expand shared questions; full9000 native completeness not certified here.')

# AnchorZip scores are useful evidence that a task ran, but never create an EADP
# baseline. Resolve its underlying raw file, not only the summary score number.
for row in latest['rows']:
    if row['task']=='Avg' or row['status']!='complete':continue
    model={'LLaVA-1.5-7B':'v15','LLaVA-NeXT-7B':'next','Qwen3-VL-8B':'qwen'}[row['model']]
    pipeline='llava' if model!='qwen' else 'official_legacy' if row['pipeline']=='official_legacy' else 'native_restored'
    key=(model,pipeline,row['task'],row['budget_parameter'])
    if key not in cells:continue
    path=Path(row['score_source_path'])
    raw=path if path.suffix=='.jsonl' else path.with_name(path.name.replace('_score.json','.json'))
    cells[key]['anchorzip_evidence'].append(dict(stage='historical_full_AZ',score=row['actual_score'],metric=row['metric'],
        raw=scan(raw),score_artifact=artifact(path,'AnchorZip_score'),
        source_sha_matches_latest=artifact(path).get('sha256')==row['score_source_sha256']))
    if model=='qwen' and row['pipeline']=='native_restored_auxiliary':cells[key]['caveats'].append('Native pipeline is auxiliary; do not mix into legacy Avg10.')

# Complete or partial AZ raw files on server also exist for externally graded
# MMVet and interrupted VQAv2. Keep generation separate from score reproduction.
for model,prefix in [('v15',''),('next','next_')]:
    for short,task in TASKS.items():
        for budget,arm in [(128,'LRMAIN025'),(64,'LRMAIN0125'),(32,'LRMAIN00625')]:
            cell=cells[(model,'llava',task,budget)]
            if any(item.get('stage')=='historical_full_AZ' for item in cell['anchorzip_evidence']):continue
            path=P3/(prefix+short)/(arm+'.jsonl')
            if path.is_file():cell['anchorzip_evidence'].append(dict(stage='AZ_raw_no_certified_official_score',raw=scan(path),score=None))

rows=[]
for cell in cells.values():
    completed=cell['completed_local'];pending=cell['registered_local_reruns'];partial=cell['unscored_or_partial_local']
    cell['has_full_scored_local_baseline']=bool(completed)
    cell['registered_EADP_or_FULL_count']=len(pending)
    cell['wait_repaired_complete']=any(ev['stage']=='completed_wait_only_control' for ev in completed)
    cell['status']=('complete_local_with_wait_repair_pending' if pending and completed else 'complete_local' if completed else
                    'registered_pending_no_complete_local' if pending else 'raw_partial_or_unscored' if partial else
                    'AZ_only_no_local_EADP' if cell['anchorzip_evidence'] else 'not_reproduced')
    cell['paper_configuration_identity']='unverified' if cell['pipeline']!='native_restored' else 'different_engine_auxiliary'
    cell['caveats']=sorted(set(cell['caveats']))
    rows.append(cell)
counts={}
for model,pipeline in {(cell['model'],cell['pipeline']) for cell in rows}:
    selected=[cell for cell in rows if (cell['model'],cell['pipeline'])==(model,pipeline)]
    counts[model+'/'+pipeline]=dict(n_cells=len(selected),statuses=dict(Counter(cell['status'] for cell in selected)),
        complete_scored_cells=sum(cell['has_full_scored_local_baseline'] for cell in selected),
        completed_EADP_only=sum(cell['has_full_scored_local_baseline'] and cell['budget']!='FULL' for cell in selected),
        paper_main_budget_cells=sum(cell['paper_score'] is not None for cell in selected),
        local_complete_among_paper_main_budget_cells=sum(cell['paper_score'] is not None and cell['has_full_scored_local_baseline'] for cell in selected))
report=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,
  GPU_or_queue_or_source_modified=False,scope='Local actual full EADP/FULL reproduction coverage; paper and AnchorZip never substitute local baselines',
  source_artifacts=[artifact(OUT/name,'audit_input') for name in ['paper_reference.json','latest_method_comparison.json','rerun_batch/repaired_results_summary.json']]+[artifact(ROOT/'docs/project_handoff.md','research_context')],
  build_source=artifact(__file__,'CPU_inventory_builder'),
  protocol='One model/pipeline/benchmark/budget cell; old, repaired input and registered stream-wait generations kept distinct within cell.',
  rows=rows,summary=counts,
  averages=dict(llava_v15_paper_Avg9='not reproduced: VQAv2/MMVet and other local E cells missing',
                llava_next_paper_Avg9='not reproduced: six legacy benchmarks have no complete local EADP baseline',
                qwen_legacy_K256_Avg10=dict(local=p1['avg10']['eg'],source=artifact(QROOT/'stage1_roundtrip_pilot/legacy_full/p1_unified_20261008.json'),
                    coverage='ten complete legacy EADP256 benchmark predictions and scores; other budgets/FULL not fully reproduced'),
                qwen_native='independent auxiliary engine; does not constitute legacy paper Avg10 reproduction'),
  limitations=['This is a source-and-full-count inventory, consuming already verified official scores; no new GPU or parameter search.',
    'Complete scored coverage does not certify generation-time bank/source identity: Qwen legacy HallusionBench256 has a recorded/current compressed bank SHA mismatch, separately exposed in its evidence.',
    'Declared official score provenance is preserved; historical process argv/environment absent is not retrospectively reconstructed as fact.',
    'Qwen legacy and native are separate; native tables are auxiliary, not proof of paper legacy reproduction.',
    'Original28 completed and55 publication groups registered does not mean full model/benchmark/budget coverage.',
    'Current partial rerun source hashes are a read-only snapshot; no partial accuracy is calculated.',
    'No full LLaVA paper Avg9 computed from missing VQAv2/MMVet; paper configuration identity remains unverified.'])
report['artifact_registry']=list(ARTIFACTS.values())
(HERE/'coverage_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
columns=['model','pipeline','benchmark','budget','local_method_required','status','has_full_scored_local_baseline',
         'registered_EADP_or_FULL_count','paper_score','paper_configuration_identity','expected_prediction_n','metric_denominator_n',
         'completed_local','unscored_or_partial_local','registered_local_reruns','anchorzip_evidence','caveats']
with (HERE/'coverage_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in rows:writer.writerow({key:json.dumps(row[key],ensure_ascii=False) if isinstance(row[key],(list,dict)) else row[key] for key in columns})
print(json.dumps(counts,ensure_ascii=False,indent=2))
print('cells',len(rows),'unique_files',len(ARTIFACTS))
