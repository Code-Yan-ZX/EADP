"""Score only independently accepted complete M3 DEV oracle results."""
import argparse,hashlib,importlib.util,io,json,math,os,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXEC=ROOT/'m3_oracle_exec';E1=ROOT/'e1_analysis'
sys.path[:0]=[str(HERE),str(EXEC),str(E1)]
# Validate all local modules before Python executes their top-level code.
_PINNED_MODULES = {'paired': '4070e5343d40d26fe1135468a50c5e30da138cde0f1b8f0533e46d25ff6d31e3', 'score_gate': 'dde97124f3107aef20db62ee4693d1721a750f2507bbf8f8869a706a81e56aec', 'analysis_gate': 'eee62f7a1c89c5a137bb22fa3d218a3fef45ace8c9294312d60bb3ae2085096d', 'e1_admission': 'ac967e1b7742e2fc5868c1295ee273c7f354563448bab2a493aa3692c08d64e5', 'e1_baseline': '834677389c8fd1dc40f5512f4f3345cb76faf60c0d586ab0bf6032de80eda39d', 'batch_gate': '4093833e34a1e47773b2f9bcaa5739b33a99454578153c9d8a33ba5322e9a4fd'}
_module_paths={'paired':HERE/'paired.py','score_gate':HERE/'score_gate.py','analysis_gate':HERE/'analysis_gate.py','e1_admission':EXEC/'e1_admission.py','e1_baseline':EXEC/'e1_baseline.py','batch_gate':ROOT/'m3_oracle_batch2/batch_gate.py'}
for _name,_path in _module_paths.items():
    if _name in sys.modules or hashlib.sha256(_path.read_bytes()).hexdigest()!=_PINNED_MODULES[_name]:raise ValueError('preloaded or changed scoring dependency: '+_name)
import paired,score_gate,e1_admission,e1_baseline,analysis_gate
for _module,_path in ((paired,HERE/'paired.py'),(score_gate,HERE/'score_gate.py'),(analysis_gate,HERE/'analysis_gate.py'),(e1_admission,EXEC/'e1_admission.py'),(e1_baseline,EXEC/'e1_baseline.py')):
    if Path(getattr(_module,'__file__','/')).resolve()!=_path.resolve():raise ValueError('scoring module source mismatch')
PLAN_SHA256='948e70d0a1c40060d57bda4550af25104ecd8273b063c3a5421f95e4193ec94a'

def lines(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]

def author_modules():
    # Pin modules even if caller has polluted generic module names.
    for name in ('completion','metrics','paired_stats'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=(E1/(name+'.py')).resolve():
            raise ValueError('conflicting E1 scorer dependency: '+name)
    spec=importlib.util.spec_from_file_location('m3_accepted_e1_scorer',E1/'score.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.metrics

def native_predictions(rows):
    import pandas as pd
    out={}
    for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
        for arm in paired.ARMS:
            selected=[r for r in rows if r['task']==task];stream=io.BytesIO()
            pd.DataFrame({'index':list(range(len(selected))),'prediction':[r['arms'][arm]['answer'] for r in selected]}).to_excel(stream,index=False,engine='xlsxwriter')
            stream.seek(0);table=pd.read_excel(stream)
            if len(table)!=len(selected) or list(table['index'])!=list(range(len(selected))):raise ValueError('author spreadsheet denominator changed')
            for row,value in zip(selected,table['prediction']):out[(task,str(row['qid']),arm)]=str(value)
    return out

def main():
    p=argparse.ArgumentParser()
    for name in ('run','proof','verify-started','verify-terminal','run-started','run-terminal','review','review-started','review-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();
    if a.out.resolve()!=(HERE/'score_001').resolve():raise ValueError('fixed second-batch output required')
    plan=json.loads((HERE/'plan.json').read_text())
    if (score_gate.batch_gate.sha(HERE/'plan.json')!=PLAN_SHA256 or
        score_gate.batch_gate.sha(ROOT/'m3_oracle/protocol_001.json')!=plan['oracle_protocol_sha256'] or
        score_gate.batch_gate.sha(EXEC/'protocol_amendment_001.json')!=plan['oracle_amendment_sha256'] or
        score_gate.batch_gate.sha(ROOT/'m3_oracle_batch2/execution_plan_002.json')!=plan['batch2_execution_sha256'] or
        score_gate.batch_gate.sha(HERE/'plan_001_before_timing_clarification.json')!=plan['prior_plan_sha256'] or
        score_gate.batch_gate.sha(ROOT/'m3_oracle_analysis/plan.json')!=plan['first_batch_plan_sha256'] or
        plan.get('questions')!=128 or plan.get('new_generations')!=768 or plan.get('arms')!=list(paired.ARMS) or
        plan.get('GT_opened') is not False or plan.get('predictions_opened') is not False or plan.get('scoring_performed') is not False):
        raise ValueError('analysis plan invalid')
    # No prediction JSON or dataset reference is parsed before all gates pass.
    analysis_approval=analysis_gate.validate()
    accepted=score_gate.gate(a)
    run=a.run.resolve();sample_path=ROOT/'m3_oracle_batch2/sample_002.json';sample=json.loads(sample_path.read_text())
    if score_gate.batch_gate.sha(sample_path)!=plan['sample_sha256']:raise ValueError('sample changed')
    events=lines(run/'panel/panel_events.jsonl');starts=[r for r in events if r['event']=='question_started']
    if len(starts)!=128 or len(sample['examples'])!=128:raise ValueError('sample inventory invalid')
    e1=e1_admission.validate();proof_path=ROOT/'e1/e1full001_completion.json';proof_bytes=proof_path.read_bytes();proof=json.loads(proof_bytes)
    journal=Path(proof['files']['journal']['path']).read_bytes();inputs=Path(proof['files']['inputs']['path']).read_bytes();rows=[]
    for expected,start in zip(sample['examples'],starts):
        key=(expected['task'],str(expected['qid']),expected['group_id'])
        if (start['task'],str(start['qid']),start['group_id'])!=key:raise ValueError('question order differs')
        raw,row,_=e1_baseline.select(proof_bytes,journal,inputs,expected,e1);base=json.loads(raw)
        path=run/'panel'/start['journal'];events_q=lines(path);generated=[x for x in events_q if x['event']=='generation_completed']
        if len(generated)!=6:raise ValueError('missing generated arm')
        arms={'EADP':{'answer':base['answer'],'eos':base['eos'],'truncated':base['truncated']}}
        for item in generated:
            name=item['source']+'-r'+str(item['r']);out=item['output']
            if name in arms:raise ValueError('duplicate generated arm')
            arms[name]={'answer':out['answer'],'eos':out['eos'],'truncated':out['truncated']}
        if set(arms)!=set(paired.ARMS):raise ValueError('seven-arm inventory mismatch')
        rows.append({'task':key[0],'qid':key[1],'group_id':key[2],'arms':arms})
    if Counter(r['task'] for r in rows)!={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32}:
        raise ValueError('frozen denominator differs')
    score_module,metrics=author_modules();source_path=Path(proof['files']['dataset_summary']['path']);source= json.loads(source_path.read_text())
    if score_gate.batch_gate.sha(source_path)!=proof['files']['dataset_summary']['sha256']:raise ValueError('dataset source proof changed')
    a.out.mkdir(exist_ok=False);source_files=[HERE/'plan.json',HERE/'score_oracle.py',HERE/'score_gate.py',HERE/'analysis_gate.py',HERE/'paired.py',E1/'score.py',E1/'metrics.py',E1/'completion.py']
    source_files += [metrics.VQA,metrics.MISC,metrics.MATCH]
    before={str(x):score_gate.batch_gate.sha(x) for x in source_files}
    frozen={'stage':'oracle_batch2_dev_scoring','acceptance_sha256':score_gate.batch_gate.sha(a.proof),'review_sha256':score_gate.batch_gate.sha(a.review),
            'plan_sha256':before[str(HERE/'plan.json')],'source_sha256':before,'questions':128,'new_generations':768,
            'analysis_approval':analysis_approval,'scope':'exploratory answer-dependent oracle; no CONFIRM/FINAL, no deployable performance claim'}
    with (a.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    # Original reference sources are opened only after completion and independent review.
    truths=score_module.read_truths(source['sources'],{(r['task'],str(r['qid'])) for r in rows})
    author=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(rows)
    for row in rows:
        key=(row['task'],str(row['qid']))
        for arm in paired.ARMS:
            item=row['arms'][arm];value=metrics.score(key[0],item['answer'],truths[key],author,constants,scoring_prediction=native[(key[0],key[1],arm)])
            if not math.isfinite(value['official']) or abs(value['official']-value['independent'])>1e-12:raise ValueError('author/independent score mismatch')
            item['score']=value['official'];item['independent_score']=value['independent']
    reports={}
    for label in ('ALL','DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
        subset=rows if label=='ALL' else [r for r in rows if r['task']==label]
        pair=[{'group_id':r['group_id'],'scores':{a:r['arms'][a]['score'] for a in paired.ARMS},'answers':{a:r['arms'][a]['answer'] for a in paired.ARMS}} for r in subset]
        reports[label]=paired.analyze(pair,replicates=plan['bootstrap']['replicates'],seed=plan['bootstrap']['seed'])
        reports[label]['terminal']={arm:{'eos':sum(r['arms'][arm]['eos'] for r in subset),'truncated':sum(r['arms'][arm]['truncated'] for r in subset),
                                           'empty':sum(not r['arms'][arm]['answer'].strip() for r in subset)} for arm in paired.ARMS}
    if any(score_gate.batch_gate.sha(p)!=h for p,h in before.items()):raise ValueError('scoring source changed')
    if any(score_gate.batch_gate.sha(path)!=h for path,h in accepted['result_files_sha256'].items()):raise ValueError('oracle record changed during scoring')
    if any(score_gate.batch_gate.sha(desc['path'])!=desc['sha256'] for desc in source['sources'].values()):raise ValueError('reference source changed during scoring')
    scored=a.out/'scores.jsonl'
    with scored.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    output={'status':'completed','n_questions':128,'n_new_generations':768,'reports':reports,'scores_sha256':score_gate.batch_gate.sha(scored),
            'plan_sha256':before[str(HERE/'plan.json')],'scope':'second exploratory look, 64/32/32 task weighted answer-dependent DEV oracle; separately reported, no pooled estimate or deployment claim; confidence intervals unadjusted'}
    with (a.out/'analysis.json').open('x') as f:json.dump(output,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0
if __name__=='__main__':sys.exit(main())
