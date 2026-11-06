"""Score only independently accepted complete M3 DEV oracle results."""
import argparse,importlib.util,io,json,math,os,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXEC=ROOT/'m3_oracle_exec';E1=ROOT/'e1_analysis'
sys.path[:0]=[str(HERE),str(EXEC),str(E1)]
import paired,score_gate,e1_admission,e1_baseline,analysis_gate
PLAN_SHA256='b69018cc38ae5f9dbf5e98dc2808d48a664b4d51d6435e47085850156feaf5a3'

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
    a=p.parse_args();plan=json.loads((HERE/'plan.json').read_text())
    if (score_gate.runtime_gate.sha(HERE/'plan.json')!=PLAN_SHA256 or
        score_gate.runtime_gate.sha(ROOT/'m3_oracle/protocol_001.json')!=plan['oracle_protocol_sha256'] or
        score_gate.runtime_gate.sha(EXEC/'protocol_amendment_001.json')!=plan['oracle_amendment_sha256'] or
        plan.get('questions')!=128 or plan.get('new_generations')!=768 or plan.get('arms')!=list(paired.ARMS) or
        plan.get('GT_opened') is not False or plan.get('predictions_opened') is not False or plan.get('scoring_performed') is not False):
        raise ValueError('analysis plan invalid')
    # No prediction JSON or dataset reference is parsed before all gates pass.
    analysis_approval=analysis_gate.validate()
    accepted=score_gate.gate(a)
    run=a.run.resolve();sample_path=ROOT/'m3_oracle/sample_001.json';sample=json.loads(sample_path.read_text())
    if score_gate.runtime_gate.sha(sample_path)!=plan['sample_sha256']:raise ValueError('sample changed')
    events=score_gate.verify.lines(run/'panel/panel_events.jsonl');starts=[r for r in events if r['event']=='question_started']
    if len(starts)!=128 or len(sample['examples'])!=128:raise ValueError('sample inventory invalid')
    e1=e1_admission.validate();proof_path=ROOT/'e1/e1full001_completion.json';proof_bytes=proof_path.read_bytes();proof=json.loads(proof_bytes)
    journal=Path(proof['files']['journal']['path']).read_bytes();inputs=Path(proof['files']['inputs']['path']).read_bytes();rows=[]
    for expected,start in zip(sample['examples'],starts):
        key=(expected['task'],str(expected['qid']),expected['group_id'])
        if (start['task'],str(start['qid']),start['group_id'])!=key:raise ValueError('question order differs')
        raw,row,_=e1_baseline.select(proof_bytes,journal,inputs,expected,e1);base=json.loads(raw)
        path=run/'panel'/start['journal'];events_q=score_gate.verify.lines(path);generated=[x for x in events_q if x['event']=='generation_completed']
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
    if score_gate.runtime_gate.sha(source_path)!=proof['files']['dataset_summary']['sha256']:raise ValueError('dataset source proof changed')
    a.out.mkdir(exist_ok=False);source_files=[HERE/'plan.json',HERE/'score_oracle.py',HERE/'score_gate.py',HERE/'analysis_gate.py',HERE/'paired.py',E1/'score.py',E1/'metrics.py',E1/'completion.py']
    source_files += [metrics.VQA,metrics.MISC,metrics.MATCH]
    before={str(x):score_gate.runtime_gate.sha(x) for x in source_files}
    frozen={'stage':'oracle_dev_scoring','acceptance_sha256':score_gate.runtime_gate.sha(a.proof),'review_sha256':score_gate.runtime_gate.sha(a.review),
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
    if any(score_gate.runtime_gate.sha(p)!=h for p,h in before.items()):raise ValueError('scoring source changed')
    if any(score_gate.runtime_gate.sha(path)!=h for path,h in accepted['result_files_sha256'].items()):raise ValueError('oracle record changed during scoring')
    if any(score_gate.runtime_gate.sha(desc['path'])!=desc['sha256'] for desc in source['sources'].values()):raise ValueError('reference source changed during scoring')
    scored=a.out/'scores.jsonl'
    with scored.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    output={'status':'completed','n_questions':128,'n_new_generations':768,'reports':reports,'scores_sha256':score_gate.runtime_gate.sha(scored),
            'plan_sha256':before[str(HERE/'plan.json')],'scope':'answer-dependent DEV oracle; not a deployable method or universal upper bound; confidence intervals unadjusted for multiple comparisons'}
    with (a.out/'analysis.json').open('x') as f:json.dump(output,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0
if __name__=='__main__':sys.exit(main())
