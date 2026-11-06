"""Score an accepted full E1 DEV run. Never opens answers before acceptance."""
import argparse
import ast
import csv
import io
import importlib.metadata
import json
import math
from pathlib import Path
import string
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import completion, metrics, paired_stats

def native_xlsx_predictions(natural):
    """Reproduce author xlsxwriter→pandas prediction-column round trip."""
    import pandas as pd
    result={}
    for task in completion.TASKS:
        for arm in paired_stats.ARMS:
            subset=[r for r in natural if r['task']==task and r['arm']==arm]
            data=pd.DataFrame({'index':list(range(len(subset))), 'prediction':[r['answer'] for r in subset]})
            stream=io.BytesIO()
            data.to_excel(stream,index=False,engine='xlsxwriter')
            stream.seek(0)
            read=pd.read_excel(stream)
            if len(read)!=len(subset) or list(read['index'])!=list(range(len(subset))):
                raise ValueError('xlsx round-trip changed denominator/order')
            for row,value in zip(subset,read['prediction']):
                result[(task,str(row['qid']),arm)]=str(value)
    if len(result)!=len(natural):raise ValueError('xlsx prediction inventory mismatch')
    return result

def read_truths(sources,needed):
    csv.field_size_limit(100_000_000)
    result={}
    for task,desc in sources.items():
        path=Path(desc['path'])
        if completion.sha(path)!=desc['sha256']:raise ValueError('source dataset changed: '+task)
        wanted={qid for t,qid in needed if t==task}
        # Read source bytes for identity and locate DEV qids before decoding or
        # CSV-parsing any answer. These frozen TSVs place index at one edge.
        # A selected row spanning physical lines fails closed at the inventory
        # or row-width check rather than admitting an unchecked answer.
        with path.open('rb') as f:
            header=next(csv.reader([f.readline().decode('utf8')],delimiter='\t'))
            index_col=header.index('index')
            if index_col not in (0,len(header)-1):
                raise ValueError('source qid must be first or last field')
            wanted_bytes={qid.encode('utf8') for qid in wanted}
            for raw in f:
                key=(raw.split(b'\t',1)[0] if index_col==0 else raw.rstrip(b'\r\n').rsplit(b'\t',1)[-1])
                if len(key)>=2 and key[:1]==b'"' and key[-1:]==b'"': key=key[1:-1]
                if key not in wanted_bytes:continue
                fields=next(csv.reader([raw.decode('utf8')],delimiter='\t'))
                if len(fields)!=len(header):raise ValueError('selected TSV row not complete')
                row=dict(zip(header,fields)); qid=str(row['index'])
                if qid not in wanted:raise ValueError('selected qid mismatch')
                if (task,qid) in result:raise ValueError('duplicate source qid')
                if task=='AI2D_TEST':
                    choices={k:row[k] for k in string.ascii_uppercase if k in row and row[k] not in metrics.NATIVE_NA}
                    result[task,qid]={'answer':row['answer'],'choices':choices}
                else:
                    value=row['answer']
                    try:parsed=ast.literal_eval(value)
                    except (ValueError,SyntaxError):parsed=value
                    answers=[str(x) for x in parsed] if isinstance(parsed,list) else [str(parsed)]
                    if not answers:raise ValueError('empty answer list')
                    result[task,qid]={'answers':answers}
        if len([k for k in result if k[0]==task])!=len(wanted):raise ValueError('missing reference question: '+task)
    if set(result)!=needed:raise ValueError('reference inventory differs')
    return result

def recheck_acceptance(proof_path):
    proof=json.loads(proof_path.read_text())
    if (proof.get('status')!='passed' or proof.get('stage')!='full' or proof.get('natural_questions')!=384 or
        proof.get('natural_records')!=1536 or proof.get('tokenizer_decode_checked')!=1536 or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or
        proof.get('verifier_sha256')!=completion.sha(HERE/'completion.py')):
        raise ValueError('acceptance proof is incomplete or inconsistent')
    if set(proof['files'])!=completion.PROOF_FILES:raise ValueError('acceptance file inventory mismatch')
    for item in proof['files'].values():
        if completion.sha(item['path'])!=item['sha256']:raise ValueError('accepted artifact changed: '+item['path'])
    files={k:Path(v['path']) for k,v in proof['files'].items()}
    terminal=json.loads(files['terminal'].read_text());exited=json.loads(files['container_exit'].read_text())
    completion.validate_terminal(terminal,exited,json.loads(files['launch'].read_text()),files['journal'].parent)
    panel=completion.read_jsonl(files['panel']);journal=completion.read_jsonl(files['journal'])
    natural=completion.validate_records(panel,journal,completion.read_jsonl(files['quartets']))
    if len(natural)!=1536:raise ValueError('full natural set changed')
    return proof,files,panel,natural

def validate_acceptance_supervisor(proof_path,started_path,terminal_path):
    started=json.loads(started_path.read_text());terminal=json.loads(terminal_path.read_text())
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or
        terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid') or
        not isinstance(started.get('worker_pid'),int) or started['worker_pid']<=0):
        raise ValueError('acceptance verifier did not exit cleanly under supervisor')
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    scripts=[(cwd/arg).resolve() for arg in command if Path(arg).name=='completion.py']
    if scripts!=[(HERE/'completion.py').resolve()] or command.count('--out')!=1:
        raise ValueError('supervisor did not invoke the acceptance verifier')
    index=command.index('--out')
    if index+1>=len(command) or (cwd/command[index+1]).resolve()!=proof_path.resolve():
        raise ValueError('supervisor did not produce this acceptance proof')

def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--acceptance-started',type=Path,required=True);p.add_argument('--acceptance-terminal',type=Path,required=True)
    p.add_argument('--dataset-summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    # This entire block is run before accessing dataset answers.
    validate_acceptance_supervisor(a.acceptance,a.acceptance_started,a.acceptance_terminal)
    proof,files,panel,natural=recheck_acceptance(a.acceptance)
    if completion.sha(a.dataset_summary)!=completion.sha(files['dataset_summary']):raise ValueError('dataset summary differs from accepted run')
    meta=json.loads(a.dataset_summary.read_text())
    if meta['outputs_sha256']['panel_manifest.jsonl']!=completion.sha(files['panel']):raise ValueError('dataset summary not bound to accepted panel')
    if set(meta['sources'])!=set(completion.TASKS):raise ValueError('unexpected source tasks')
    needed={(r['task'],str(r['qid'])) for r in panel if r['split']=='DEV'}
    if len(needed)!=384:raise ValueError('incomplete DEV panel')
    a.out.mkdir(parents=True,exist_ok=False)
    source_code={str(path):completion.sha(path) for path in [metrics.VQA,metrics.MISC,metrics.MATCH,HERE/'metrics.py',HERE/'paired_stats.py',Path(__file__)]}
    frozen={'acceptance_sha256':completion.sha(a.acceptance),'acceptance_started_sha256':completion.sha(a.acceptance_started),
            'acceptance_terminal_sha256':completion.sha(a.acceptance_terminal),'dataset_summary_sha256':completion.sha(a.dataset_summary),
            'inputs':proof['files'],'code_sha256':source_code,'native_na_policy':'Exact author xlsxwriter/pandas prediction-column round trip by task and arm; raw answers retained; literal-score diagnostic separately.',
            'runtime_versions':{name:importlib.metadata.version(name) for name in ('numpy','pandas','xlsxwriter','openpyxl')},
            'ai2d':'author can_infer deterministic exact_matching, unresolved mapped Z; no external judge',
            'text_independent':'separately coded punctuation and closed-form leave-one-out consensus; constants extracted from author source',
            'bootstrap':{'unit':'image group','replicates':10000,'seed':20261104,'method':'percentile'},
            'scope':'384 DEV questions only; no CONFIRM/FINAL predictions read'}
    (a.out/'frozen.json').write_text(json.dumps(frozen,indent=2,ensure_ascii=False)+'\n')
    # Ground truth is opened only after full completeness, decoding, exit and hash gates.
    truths=read_truths(meta['sources'],needed)
    author=metrics.author_functions();constants=metrics._author_constants()
    native_predictions=native_xlsx_predictions(natural)
    by_q={}
    for r in natural:
        key=(r['task'],str(r['qid'])); scored=metrics.score(r['task'],r['answer'],truths[key],author,constants,
                                                       scoring_prediction=native_predictions[(key[0],key[1],r['arm'])])
        if not math.isfinite(scored['official']) or abs(scored['official']-scored['independent'])>1e-12:
            raise ValueError(f'official/independent discrepancy: {key} {r["arm"]}')
        dst=by_q.setdefault(key,{'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'arms':{}})
        dst['arms'][r['arm']]={'raw_answer':r['answer'],'score':scored['official'],'independent_score':scored['independent'],
                                'literal_answer_diagnostic':scored['literal_answer_diagnostic'],
                                'normalized_prediction':scored['normalized_prediction'],'eos':r['eos'],'truncated':r['truncated'],
                                **({'official_option':scored['official_option'],'independent_option':scored['independent_option']} if r['task']=='AI2D_TEST' else {})}
    if set(by_q)!=needed or any(set(x['arms'])!=set(paired_stats.ARMS) for x in by_q.values()):raise ValueError('score quartet incomplete')
    scored=[by_q[(r['task'],str(r['qid']))] for r in panel if r['split']=='DEV']
    per_task={}
    for task in completion.TASKS:
        selected=[r for r in scored if r['task']==task]
        statsrows=[{'group_id':r['group_id'],'scores':{a:r['arms'][a]['score'] for a in paired_stats.ARMS},
                    'answers':{a:r['arms'][a]['raw_answer'] for a in paired_stats.ARMS}} for r in selected]
        per_task[task]=paired_stats.analyze(statsrows)
        per_task[task]['terminal']={a:{'empty':sum(not r['arms'][a]['raw_answer'].strip() for r in selected),
                                      'truncated':sum(r['arms'][a]['truncated'] for r in selected),
                                      'eos':sum(r['arms'][a]['eos'] for r in selected)} for a in paired_stats.ARMS}
    for path,digest in source_code.items():
        if completion.sha(path)!=digest:raise ValueError('metric source changed during scoring')
    for task,desc in meta['sources'].items():
        if completion.sha(desc['path'])!=desc['sha256']:raise ValueError('dataset changed during scoring: '+task)
    for item in proof['files'].values():
        if completion.sha(item['path'])!=item['sha256']:raise ValueError('accepted artifact changed during scoring')
    out=a.out/'scores.jsonl'
    with out.open('x',encoding='utf8') as f:
        for row in scored:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    summary={'status':'completed','tasks':per_task,'n_questions':len(scored),'scores_sha256':completion.sha(out),
             'acceptance_sha256':completion.sha(a.acceptance),'limits':['DEV only, exploratory; 95% intervals are per contrast without multiplicity adjustment.',
             'AI2D reports deterministic local exact-matching fallback, not an API judge.',
             'Bootstrap resamples image groups with all their questions; benchmark datasets were historically exposed.']}
    (a.out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'completed','n_questions':len(scored)}))
if __name__=='__main__':main()
