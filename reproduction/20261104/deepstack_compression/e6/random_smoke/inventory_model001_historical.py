"""Three-task random K256 smoke inventory, without model or launch authority.

The caller must first establish all upstream admissions, source identities,
model/recipe and exclusive GPU ownership. This module cannot unlock a run.
"""
import json
import os
from pathlib import Path

SEEDS=(20261104,20261105,20261106)

def emit(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n')
        f.flush();os.fsync(f.fileno())

def validate_inventory(rows, references, frozen, anchors):
    if len(rows)!=3 or len(anchors)!=3 or len({r['task'] for r in rows})!=3:raise ValueError('three distinct tasks required')
    keys=set()
    for row,anchor in zip(rows,anchors):
        key=(row['task'],str(row['qid']))
        if key in keys or row.get('split')!='DEV':raise ValueError('duplicate/non-DEV anchor')
        keys.add(key)
        if any(str(row.get(k))!=str(anchor.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('anchor mismatch')
        refs=references.get(key,[])
        if len(refs)!=1 or refs[0].get('arm')!='P3-D1' or refs[0].get('forced') is not False:raise ValueError('ambiguous E1 reference')
        for seed in SEEDS:
            choice=frozen.get((key[0],key[1],seed))
            if not isinstance(choice,dict) or any(str(choice.get(k))!=str(row.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('frozen seed binding')
            ids=choice.get('spatial_ids')
            if not isinstance(ids,list) or len(ids)!=256 or ids!=sorted(set(ids)) or any(type(x)!=int or x<0 or x>=1024 for x in ids):raise ValueError('frozen indices')
    return True

def execute(rows,references,frozen,anchors,out,run_one):
    """Run all fixed arms, writing exposure before post-generation checks.

    run_one(row, reference, seed, forced, on_generated) must call on_generated
    immediately after generate returns. Each completed record is fsynced.
    No restart or omission of failed arms is performed.
    """
    validate_inventory(rows,references,frozen,anchors)
    out=Path(out)
    if not out.is_dir() or any((out/n).exists() for n in ('generated.jsonl','records.jsonl','journal.jsonl','worker_complete.json')):raise ValueError('output must be new')
    n=0
    for row in rows:
      key=(row['task'],str(row['qid']));reference=references[key][0]
      for seed in SEEDS:
       for forced in (False,True):
        arm={'task':key[0],'qid':key[1],'group_id':row['group_id'],'seed':seed,'forced':forced}
        emit(out/'journal.jsonl',dict(event='started',**arm))
        observed=[]
        def generated(raw):
            if observed:raise ValueError('multiple exposure records')
            # Preserve the returned payload before even checking its identity:
            # a bad prefix or metadata is still a real generation exposure.
            emit(out/'generated.jsonl',{'scheduled':arm,'raw':raw});observed.append(True)
            if not isinstance(raw,dict) or raw.get('task')!=key[0] or str(raw.get('qid'))!=key[1] or raw.get('group_id')!=row['group_id'] or raw.get('seed')!=seed or raw.get('forced') is not forced:raise ValueError('exposure identity mismatch')
            if not isinstance(raw.get('output_ids'),list) or not raw['output_ids']:raise ValueError('empty exposure')
        try:
            rec=run_one(row,reference,seed,forced,generated)
            if len(observed)!=1 or not isinstance(rec,dict):raise ValueError('missing exposure/record')
            if any(rec.get(k)!=v for k,v in arm.items()):raise ValueError('record identity mismatch')
            if rec.get('spatial_ids')!=frozen[(key[0],key[1],seed)]['spatial_ids']:raise ValueError('selection differs from frozen manifest')
            emit(out/'records.jsonl',rec);n+=1;emit(out/'journal.jsonl',dict(event='recorded',**arm))
        except BaseException as exc:
            emit(out/'journal.jsonl',dict(event='failed',error_type=type(exc).__name__,error=str(exc),**arm));raise
    if n!=18:raise ValueError('incomplete inventory')
    complete={'status':'completed','arms':n,'natural':9,'forced33':9,'GT_opened':False,'scoring_performed':False,'timing_eligible':False}
    data=json.dumps(complete,indent=2).encode()+b'\n'
    with (out/'worker_complete.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return complete
