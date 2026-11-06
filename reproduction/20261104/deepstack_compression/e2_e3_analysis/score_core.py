"""E2/E3 in-memory scoring helpers; no IO, admission, or GT loading.

These helpers must not be called on real records until full formal and
independent no-GT acceptance have passed. No executable entrypoint exists.
"""
from collections import Counter
import io
import math
import pandas as pd
from paired import CONFIGS

TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')

def config(record):
    key=record['family']+'/'+record['arm']
    if key not in CONFIGS:raise ValueError('unexpected physical arm')
    return key

def inventory(panel,records,*,per_task=128):
    if len(panel)!=3*per_task or Counter(r['task'] for r in panel)!={t:per_task for t in TASKS}:
        raise ValueError('panel denominator')
    if len({(r['task'],str(r['qid'])) for r in panel})!=len(panel) or len({r['group_id'] for r in panel})!=len(panel):
        raise ValueError('duplicate question or group')
    if len(records)!=len(panel)*len(CONFIGS):raise ValueError('record denominator')
    grouped={}
    for i,row in enumerate(panel):
        key=(row['task'],str(row['qid']));grouped[key]={}
        for arm,r in zip(CONFIGS,records[i*len(CONFIGS):(i+1)*len(CONFIGS)]):
            if (config(r)!=arm or r.get('forced') is not False or any(str(r.get(k))!=str(row.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256'))):
                raise ValueError('record identity/order mismatch')
            grouped[key][arm]=r
    return grouped

def native_predictions(records,*,per_task=128):
    """Mirror E1's author XLSX writer/pandas reader for each task/config."""
    result={}
    for task in TASKS:
        for arm in CONFIGS:
            sub=[r for r in records if r['task']==task and config(r)==arm]
            if len(sub)!=per_task:raise ValueError('xlsx denominator')
            buf=io.BytesIO()
            pd.DataFrame({'index':list(range(len(sub))),'prediction':[r['answer'] for r in sub]}).to_excel(buf,index=False,engine='xlsxwriter')
            buf.seek(0);roundtrip=pd.read_excel(buf)
            if len(roundtrip)!=len(sub) or list(roundtrip['index'])!=list(range(len(sub))):raise ValueError('xlsx rows changed')
            for r,x in zip(sub,roundtrip['prediction']):result[(task,str(r['qid']),arm)]=str(x)
    if len(result)!=len(records):raise ValueError('prediction keys not unique')
    return result

def score_rows(panel,grouped,truth,native,score):
    """Require both real author and independent score supplied by caller."""
    out=[]
    for row in panel:
        key=(row['task'],str(row['qid']))
        if key not in truth or key not in grouped:raise ValueError('missing truth or group')
        result={'task':key[0],'qid':key[1],'group_id':row['group_id'],'arms':{}}
        for arm in CONFIGS:
            r=grouped[key][arm];value=score(key[0],r['answer'],truth[key],native[(key[0],key[1],arm)])
            a,b=float(value['official']),float(value['independent'])
            if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=1 or not 0<=b<=1 or abs(a-b)>1e-12:
                raise ValueError('author and independent score differ')
            result['arms'][arm]={'answer':r['answer'],'score':a,'independent_score':b,
                                 'eos':r['eos'],'truncated':r['truncated']}
        out.append(result)
    return out
