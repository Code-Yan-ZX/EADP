"""Exploratory E1 DEV strata from native metadata; no held-out answers/predictions."""
import argparse, ast, collections, csv, datetime, hashlib, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
sys.path.insert(0,str(HERE.parent))
import paired_stats

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def labels(path,task,wanted,column):
    """Only decode/parse TSV records with a DEV qid at an edge field."""
    csv.field_size_limit(100000000); wanted_bytes={q.encode() for q in wanted}; found={}
    with Path(path).open('rb') as f:
        header=next(csv.reader([f.readline().decode()],delimiter='\t')); index=header.index('index')
        if index not in (0,len(header)-1): raise ValueError('qid not at edge')
        for raw in f:
            q=(raw.split(b'\t',1)[0] if index==0 else raw.rstrip(b'\r\n').rsplit(b'\t',1)[-1])
            if len(q)>=2 and q[:1]==b'"' and q[-1:]==b'"':q=q[1:-1]
            if q not in wanted_bytes:continue
            fields=next(csv.reader([raw.decode()],delimiter='\t'))
            if len(fields)!=len(header):raise ValueError('incomplete selected TSV record')
            row=dict(zip(header,fields)); qid=str(row['index'])
            if qid not in wanted or qid in found:raise ValueError('wrong/duplicate DEV qid')
            value=row[column]
            if task=='DocVQA_VAL':
                value=ast.literal_eval(value)
                if not isinstance(value,list) or any(not isinstance(v,str) or not v for v in value) or len(set(value))!=len(value):
                    raise ValueError('invalid DocVQA native question_types')
            else:
                if not isinstance(value,str) or not value:raise ValueError('empty AI2D category')
                value=[value]
            found[qid]=value
    if set(found)!=wanted:raise ValueError('missing DEV native type')
    return found

def analyze(rows):
    result=paired_stats.analyze(rows,replicates=10000,seed=20261104)
    # Small groups remain visible but do not get unstable bootstrap intervals.
    if len({r['group_id'] for r in rows})<10:
        for value in result['contrasts'].values():value['image_cluster_percentile_95_ci_pp']=None
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    e=BASE/'e1';scored=e/'e1full001_score'; audit=BASE/'e1_independent_review/e1full001_review.json'
    files={'scores':scored/'scores.jsonl','analysis':scored/'analysis.json','provenance':scored/'provenance.json',
           'score_frozen':scored/'frozen.json','score_terminal':e/'monitorscoree1full001/terminal.json',
           'completion':e/'e1full001_completion.json','independent_review':audit,
           'independent_review_terminal':BASE/'e1_independent_review/monitor_e1full001/terminal.json',
           'dataset_summary':BASE/'data_manifest/manifest_summary.json','code':Path(__file__),'stats_code':HERE.parent/'paired_stats.py'}
    review=json.loads(audit.read_text());term=json.loads(files['score_terminal'].read_text());rt=json.loads(files['independent_review_terminal'].read_text())
    if review.get('status')!='passed' or term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or rt.get('reason')!='completed' or rt.get('worker_returncode')!=0 or rt.get('remaining_process_group')!=[]:raise ValueError('upstream exit/review failed')
    if review.get('n_expected')!=384 or review.get('n_verified')!=384 or review.get('completion_proof_sha256')!=sha(files['completion']):raise ValueError('independent review does not bind this full run')
    before={k:sha(v) for k,v in files.items()}
    prov=json.loads(files['provenance'].read_text()); summary=json.loads(files['analysis'].read_text())
    if prov.get('acceptance_sha256')!=sha(files['completion']):raise ValueError('score provenance does not bind accepted run')
    for key in ('scores','analysis'):
        if prov[key+'_sha256']!=sha(files[key]):raise ValueError('scored artifact changed')
    if prov['score_terminal_sha256']!=sha(files['score_terminal']) or summary['status']!='completed' or summary['n_questions']!=384:raise ValueError('scoring not complete')
    source=json.loads(files['dataset_summary'].read_text())['sources']
    for task in ('DocVQA_VAL','AI2D_TEST'):
        if sha(source[task]['path'])!=source[task]['sha256']:raise ValueError('source metadata TSV changed')
    rows=[json.loads(x) for x in files['scores'].read_text().splitlines()]
    counts=collections.Counter(r['task'] for r in rows)
    if counts!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} or len({(r['task'],str(r['qid'])) for r in rows})!=384 or len({r['group_id'] for r in rows})!=384:
        raise ValueError('DEV question/group inventory invalid')
    raw={task:{str(r['qid']) for r in rows if r['task']==task} for task in counts}; strata={}
    for task,col in [('DocVQA_VAL','question_types'),('AI2D_TEST','category')]:
        mapping=labels(source[task]['path'],task,raw[task],col); grouped=collections.defaultdict(list)
        for r in rows:
            if r['task']!=task:continue
            arms=r['arms'];item={'group_id':r['group_id'],'scores':{a:arms[a]['score'] for a in paired_stats.ARMS},'answers':{a:arms[a]['raw_answer'] for a in paired_stats.ARMS}}
            for label in mapping[str(r['qid'])]:grouped[label].append(item)
        strata[task]={key:analyze(value) for key,value in sorted(grouped.items())}
    if any(sha(path)!=before[k] for k,path in files.items()):raise ValueError('upstream changed during analysis')
    for task in ('DocVQA_VAL','AI2D_TEST'):
        if sha(source[task]['path'])!=source[task]['sha256']:raise ValueError('source TSV changed during analysis')
    report={'status':'completed','scope':'exploratory DEV only; no held-out answer or prediction parsed','question_type_provenance':{'DocVQA_VAL':'native question_types, multi-label','AI2D_TEST':'native category','TextVQA_VAL':'source TSV has no question-type column'},
            'counts':dict(counts),'overall':summary['tasks'],'strata':strata,'method':{'bootstrap':'10000 image-group draws, seed 20261104; percentile intervals suppressed for fewer than 10 groups',
            'contrast':'paired score difference in percentage points; answer_changed compares exact raw output strings',
            'limitations':['Post-hoc exploratory subgroup analyses; intervals unadjusted for multiplicity.', 'DocVQA labels overlap; category counts must not be summed.', 'Small subgroups do not establish absence of an effect.', 'AI2D is deterministic local exact matching, no external judge.']},
            'files_sha256':{k:sha(v) for k,v in files.items()},'native_source_sha256':{t:source[t]['sha256'] for t in ('DocVQA_VAL','AI2D_TEST')},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'diagnostics.json').open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'completed','groups':{k:len(v) for k,v in strata.items()},'n_questions':len(rows)}))
if __name__=='__main__':main()
