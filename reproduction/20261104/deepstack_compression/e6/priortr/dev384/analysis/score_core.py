"""Pure DEV384 scoring and grouped statistics; no reference loading or CLI."""
from collections import Counter, defaultdict
import io
import numpy as np
import pandas as pd

TASKS = ('DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST')
ARMS = ('stock_native', 'author_default', 'priortr_k256')
REFERENCE = 'EADP_P3_D1_reused_E1'
CONTRASTS = (('author_default','stock_native'), ('priortr_k256','stock_native'),
             ('author_default',REFERENCE), ('priortr_k256',REFERENCE))

def inventory(panel, records, per_task=128):
    n = len(panel)
    if n != per_task * len(TASKS) or Counter(r['task'] for r in panel) != {t: per_task for t in TASKS}:
        raise ValueError('panel denominator')
    if len({(r['task'],str(r['qid'])) for r in panel}) != n or len({r['group_id'] for r in panel}) != n:
        raise ValueError('duplicated question or image group')
    if len(records) != n * len(ARMS):
        raise ValueError('records denominator')
    result = {}
    for i, row in enumerate(panel):
        key = (row['task'],str(row['qid']))
        result[key] = {}
        for j, arm in enumerate(ARMS):
            record = records[j*n+i]
            if record.get('arm') != arm or record.get('forced') is not False or any(str(record.get(k)) != str(row.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):
                raise ValueError('arm order or input identity')
            result[key][arm] = record
    return result

def native_predictions(records, per_task=128):
    """Reproduce local author XLSX/pandas prediction conversion per task/arm."""
    output = {}
    for task in TASKS:
        for arm in ARMS:
            subset = [r for r in records if r['task'] == task and r['arm'] == arm]
            if len(subset) != per_task:
                raise ValueError('xlsx denominator')
            buf = io.BytesIO()
            pd.DataFrame({'index':list(range(len(subset))), 'prediction':[r['answer'] for r in subset]}).to_excel(buf,index=False,engine='xlsxwriter')
            buf.seek(0); restored = pd.read_excel(buf)
            if len(restored) != len(subset) or list(restored['index']) != list(range(len(subset))):
                raise ValueError('xlsx row mismatch')
            for row,value in zip(subset,restored['prediction']):
                output[(task,str(row['qid']),arm)] = str(value)
    if len(output) != len(records):
        raise ValueError('duplicate prediction')
    return output

def score_rows(panel, grouped, truths, native, e1_scores, scorer):
    output=[]
    for row in panel:
        key=(row['task'],str(row['qid'])); prior=e1_scores[key]
        if prior['group_id'] != row['group_id']:
            raise ValueError('E1 group mismatch')
        ref=prior['arms']['P3-D1']
        if abs(float(ref['score'])-float(ref['independent_score']))>1e-12:
            raise ValueError('E1 score discrepancy')
        arms={REFERENCE:{'score':float(ref['score']),'independent_score':float(ref['independent_score']),'eos':ref['eos'],'truncated':ref['truncated']}}
        for arm in ARMS:
            record=grouped[key][arm]
            values=scorer(key[0],record['answer'],truths[key],native[(key[0],key[1],arm)])
            a,b=float(values['official']),float(values['independent'])
            if not np.isfinite([a,b]).all() or not 0<=a<=1 or not 0<=b<=1 or abs(a-b)>1e-12:
                raise ValueError('author and independent scores differ')
            arms[arm]={'score':a,'independent_score':b,'eos':record['eos'],'truncated':record['truncated']}
        output.append({'task':key[0],'qid':key[1],'group_id':row['group_id'],'arms':arms})
    return output

def analyze(rows, replicates=10000, seed=20261104):
    if not rows or not isinstance(replicates,int) or replicates<1:
        raise ValueError('empty data or invalid replicates')
    configs=(*ARMS,REFERENCE);groups=defaultdict(list)
    for row in rows:
        if not isinstance(row.get('group_id'),str) or not row['group_id'] or set(row.get('arms',{})) != set(configs):
            raise ValueError('missing group or arm')
        groups[row['group_id']].append(row)
    values=np.asarray([[float(r['arms'][a]['score']) for a in configs] for r in rows])
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('invalid scores')
    ids=sorted(groups)
    sums=np.asarray([[sum(float(r['arms'][a]['score']) for r in groups[g]) for a in configs] for g in ids]);sizes=np.asarray([len(groups[g]) for g in ids])
    draw=np.random.default_rng(seed).integers(0,len(ids),size=(replicates,len(ids)))
    boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
    result={'n_questions':len(rows),'n_image_groups':len(ids),'means_pct':{a:float(100*values[:,i].mean()) for i,a in enumerate(configs)},'contrasts':{}}
    for a,b in CONTRASTS:
        i,j=configs.index(a),configs.index(b);diff=values[:,i]-values[:,j]
        result['contrasts'][a+' - '+b]={'difference_pp':float(100*diff.mean()),'percentile_95_ci_pp':[float(x) for x in np.quantile(100*(boot[:,i]-boot[:,j]),[.025,.975])], 'positive_questions':int((diff>1e-12).sum()),'negative_questions':int((diff< -1e-12).sum()),'zero_questions':int((abs(diff)<=1e-12).sum())}
    return result
