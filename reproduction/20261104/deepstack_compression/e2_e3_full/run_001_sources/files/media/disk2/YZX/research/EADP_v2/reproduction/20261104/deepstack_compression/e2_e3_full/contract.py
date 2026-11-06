"""CPU-only full DEV design contract; never loads labels or model outputs."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
E2=('000','100','010','001','110','101','011','111')
E3=('PERM','MEAN')
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')

def bind_dev_inputs(inputs,panel):
    """Join verified input records to the separately verified DEV panel.

    The caller must first validate both file hashes and the E1 proof/exit.
    E1 input records do not themselves contain a split field.
    """
    dev=[r for r in panel if r.get('split')=='DEV'];bykey={(r['task'],str(r['qid'])):r for r in dev}
    if len(dev)!=384 or len(bykey)!=384 or len(inputs)!=384:raise ValueError('DEV input/panel denominator')
    seen=set();result=[]
    for row in inputs:
        key=(row['task'],str(row['qid']));ref=bykey.get(key)
        if ref is None or key in seen or row.get('group_id')!=ref.get('group_id') or row.get('split','DEV')!='DEV':
            raise ValueError('input does not match frozen DEV panel')
        seen.add(key);result.append(dict(row,split='DEV'))
    if seen!=set(bykey):raise ValueError('incomplete DEV input join')
    inventory(result)
    return result

def inventory(rows):
    if len(rows)!=384 or any(x.get('split')!='DEV' for x in rows):raise ValueError('requires exactly 384 DEV input records')
    ids=[(x['task'],str(x['qid'])) for x in rows];groups=[x['group_id'] for x in rows]
    if len(set(ids))!=384 or len(set(groups))!=384 or any(sum(x['task']==t for x in rows)!=128 for t in TASKS):raise ValueError('DEV task or group inventory mismatch')
    return [(r['task'],str(r['qid']),r['group_id'],family,arm)
            for r in rows for family,arm in ([('E2',x) for x in E2]+[('E3',x) for x in E3])]

def validate_records(rows,records):
    expected=inventory(rows)
    actual=[(r.get('task'),str(r.get('qid')),r.get('group_id'),r.get('family'),r.get('arm')) for r in records]
    if actual!=expected or any(r.get('forced') is not False for r in records):raise ValueError('incomplete, duplicated or non-natural panel')
    if len(records)!=3840:raise ValueError('physical denominator mismatch')
    return {'questions':384,'physical_generations':3840,'E2_records':3072,'E3_new_records':768,
            'E3_endpoint_aliases':768,'GT_opened':False,'scoring_performed':False}

def record_digest(record):
    return hashlib.sha256(json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def e3_views(rows,records):
    """Endpoint aliases require same-run same-question immutable record hashes."""
    validate_records(rows,records);views=[]
    for i,row in enumerate(rows):
        block=records[10*i:10*i+10];byarm={(r['family'],r['arm']):r for r in block}
        entries={}
        for mode,key in (('OFF',('E2','000')),('MATCH',('E2','111')),('PERM',('E3','PERM')),('MEAN',('E3','MEAN'))):
            record=byarm[key];digest=record_digest(record)
            entries[mode]={'source_family':key[0],'source_arm':key[1],'record_sha256':digest}
        views.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'arms':entries})
    return views

def locked():
    raise ValueError('full E2/E3 design is locked pending real smoke, independent review and full container preflight')
