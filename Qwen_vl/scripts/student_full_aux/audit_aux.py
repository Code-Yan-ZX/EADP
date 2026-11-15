#!/usr/bin/env python3
"""Post-freeze independent prompt reconstruction and truncation sensitivity.

Does not change the frozen experiment or GPU workload. Sensitivity refuses to
read any predictions until all 5349 questions x 7 arms are successfully present.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]
OUT=QWEN/'outputs/student_full'
FREEZE=OUT/'run_freeze.json'
MANIFEST=OUT/'docvqa_val_manifest.json'
GATE=OUT/'dev_gate.json'
DATABASE=OUT/'predictions.sqlite3'
PROMPT=QWEN/'VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
SCORE=QWEN/'scripts/student_full/score_full.py'
SUFFIX='\nPlease answer concisely with short words or phrases when possible.'
ARMS=('full','b2','b1','rres512','magnitude_v__seed_11','magnitude_v__seed_23','magnitude_v__seed_37')
COMPARISONS=tuple((r,s) for s in ARMS[4:] for r in ARMS[:4])+(('b1','b2'),)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def get_freeze():
    freeze=json.loads(FREEZE.read_text())
    if freeze['config']['arms']!=list(ARMS) or freeze['count']!=5349:
        raise ValueError('unexpected frozen experiment design')
    if freeze['manifest_sha256']!=sha(MANIFEST):raise ValueError('manifest hash mismatch')
    for path in (PROMPT,SCORE):
        if freeze['source_sha256'].get(str(path.resolve()))!=sha(path):
            raise ValueError(f'frozen source mismatch: {path}')
    gate=json.loads(GATE.read_text())
    if not gate.get('pass_all') or gate.get('run_freeze_sha256')!=sha(FREEZE):
        raise ValueError('missing gate for freeze')
    return freeze


def prompt_rows(records, mixin):
    rows=[]
    for rec in records:
        q=rec['question']
        image_path=str(OUT/'images'/f"{rec['image_sha256']}{Path(rec['image_path']).suffix.lower()}")
        mixin.set_dump_image(lambda row,path=image_path:path)
        # Exercise the exact frozen Qwen _build_vqa_prompt implementation on
        # each question; no image decoding, model, prediction or label is used.
        msg=mixin._build_vqa_prompt({'question':q},'DocVQA_VAL')
        expected=[{'type':'image','value':image_path},{'type':'text','value':q+SUFFIX}]
        if msg != expected:raise ValueError('frozen prompt implementation changed')
        rows.append({'source_row':int(rec['source_row']),'index':str(rec['index']),
                     'parent_id':str(rec['parent_id']), 'source_image_sha256':rec['image_sha256'],
                     'logical_message':msg})
    return rows


def write_once(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def make_prompts(output):
    f=get_freeze(); payload=json.loads(MANIFEST.read_text()); records=payload['records']
    if not payload['complete'] or len(records)!=5349 or sorted(int(x['source_row']) for x in records)!=list(range(5349)):
        raise ValueError('unexpected manifest rows')
    mod=load_module('qwen_frozen_prompt',PROMPT)
    mixin=mod.Qwen3VLPromptMixin(use_custom_prompt=True)
    rows=prompt_rows(records,mixin)
    # Confirm the exact common.build_message dispatch and DocVQA classification
    # via a lightweight mixin; this does not load a VLM or run a GPU operation.
    sys.path.insert(0,str(QWEN/'VLMEvalKit'))
    from vlmeval.dataset import DATASET_TYPE
    if DATASET_TYPE('DocVQA_VAL',default=None)!='VQA' or not mixin.use_custom_prompt('DocVQA_VAL'):
        raise ValueError('benchmark does not dispatch through Qwen VQA prompt')
    for rec in (records[0],records[-1]):
        one=prompt_rows([rec],mixin)[0]
        if mixin.build_prompt({'question':rec['question']},dataset='DocVQA_VAL') != one['logical_message']:
            raise ValueError('Qwen prompt dispatch differs')
    value={'complete':True,'kind':'reconstructed logical message before tokenizer/chat template',
      'note':'Paths are deterministic expected isolated images; some may be created later. This is not a stored byte-for-byte model input_ids record.',
      'count':len(rows),'run_freeze_sha256':sha(FREEZE),'manifest_sha256':sha(MANIFEST),
      'prompt_source_sha256':sha(PROMPT),'generator_sha256':sha(__file__),'suffix':SUFFIX,'records':rows}
    write_once(output,value)
    print('prompt sidecar complete',len(rows),str(output),'sha256',sha(output))


def check_complete(con):
    # Strictly metadata only: never fetch predictions on an incomplete run.
    meta=dict(con.execute('SELECT key,value FROM meta'))
    if meta.get('run_freeze_sha256') != sha(FREEZE):raise ValueError('DB belongs to another freeze')
    counts=con.execute('SELECT arm,status,count(*) FROM results GROUP BY arm,status').fetchall()
    got={(a,s):n for a,s,n in counts}
    if set(got)!={(a,'success') for a in ARMS} or any(got[(a,'success')]!=5349 for a in ARMS):
        raise RuntimeError('full results incomplete; sensitivity refuses to read predictions')
    distinct=con.execute('SELECT count(DISTINCT row_index) FROM results').fetchone()[0]
    bounds=con.execute('SELECT min(row_index),max(row_index) FROM results').fetchone()
    if distinct!=5349 or bounds!=(0,5348):raise RuntimeError('incomplete row coverage')


def sensitivity(records,dbrows,score):
    # dbrows are fetched only after complete metadata validation.
    lookup={r['row_index']:r for r in records}
    data={a:{} for a in ARMS}
    for i,arm,status,pred,trunc,n_vis,error in dbrows:
        if i not in lookup or arm not in data or i in data[arm] or status!='success' or not isinstance(pred,str):
            raise ValueError('bad result row')
        if trunc not in (0,1) or n_vis!=(1024 if arm=='full' else 256) or error is not None:
            raise ValueError('bad generation metadata')
        data[arm][i]=(score.anls(lookup[i]['answer'],pred),bool(trunc))
    if any(len(data[a])!=5349 for a in ARMS):raise ValueError('incomplete arm coverage')
    pairs={}
    for ref,target in COMPARISONS:
        valid=[i for i in sorted(lookup) if not data[ref][i][1] and not data[target][i][1]]
        excluded_ref_only=sum(data[ref][i][1] and not data[target][i][1] for i in lookup)
        excluded_target_only=sum(not data[ref][i][1] and data[target][i][1] for i in lookup)
        excluded_both=sum(data[ref][i][1] and data[target][i][1] for i in lookup)
        if valid:
            a=[data[ref][i][0] for i in valid];b=[data[target][i][0] for i in valid]
            groups=[lookup[i]['parent_id'] for i in valid]
            stats=score.paired_bootstrap(a,b,groups)
            stats.update(reference_anls=float(np.mean(a)),target_anls=float(np.mean(b)))
        else:stats=None
        pairs[f'{target} - {ref}']={'primary_denominator':len(lookup),'retained':len(valid),
             'excluded_ref_only':excluded_ref_only,'excluded_target_only':excluded_target_only,
             'excluded_both':excluded_both,'paired':stats}
    return pairs


def make_sensitivity(output):
    f=get_freeze()
    con=sqlite3.connect('file:'+str(DATABASE.resolve())+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON');con.execute('BEGIN')
    try:
        check_complete(con)
        # Only after the check, load gold and prediction data for final scoring.
        score=load_module('frozen_student_full_score',SCORE)
        records=score.load_manifest(MANIFEST,Path(f['original_tsv_path']))
        dbrows=con.execute('SELECT row_index,arm,status,prediction,truncated,n_vis,error FROM results ORDER BY row_index,arm').fetchall()
        pairs=sensitivity(records,dbrows,score)
        snapshot=hashlib.sha256()
        for row in dbrows:snapshot.update((json.dumps(row,ensure_ascii=False)+'\n').encode())
    finally:con.rollback();con.close()
    result={'complete':True,'kind':'secondary sensitivity: both paired predictions untruncated',
      'note':'Different pairs may have different denominators; primary ANLS retains all 5349 questions.',
      'scale':'0 to 1; multiply by 100 for percentage points',
      'run_freeze_sha256':sha(FREEZE),'manifest_sha256':sha(MANIFEST),'ordered_db_rows_sha256':snapshot.hexdigest(),
      'generator_sha256':sha(__file__),'score_source_sha256':sha(SCORE),
      'bootstrap':{'unit':'paired source parent, question-weighted','draws':10000,'seed':20261109,'interval':'95% percentile'},
      'comparisons':pairs}
    write_once(output,result);print('complete sensitivity saved',output,'sha256',sha(output))


def self_check():
    mod=load_module('qwen_frozen_prompt_test',PROMPT); mixin=mod.Qwen3VLPromptMixin()
    fake=[{'source_row':0,'index':'x','parent_id':'p','image_sha256':'1'*64,
           'image_path':'a.jpg','question':'What?'}]
    rows=prompt_rows(fake,mixin)
    assert rows[0]['logical_message'][1]['value']=='What?'+SUFFIX
    assert rows[0]['logical_message'][0]['value'].endswith('1'*64+'.jpg')
    # Lightweight fake DB proves no prediction read occurs when incomplete.
    con=sqlite3.connect(':memory:');con.execute('CREATE TABLE meta(key TEXT,value TEXT)')
    con.execute('CREATE TABLE results(row_index INTEGER,arm TEXT,status TEXT,prediction TEXT)')
    con.execute('INSERT INTO meta VALUES (?,?)',('run_freeze_sha256',sha(FREEZE)))
    con.execute("INSERT INTO results VALUES(0,'full','success','DO_NOT_READ')")
    before=[];con.set_trace_callback(before.append)
    try:check_complete(con);raise AssertionError('partial DB incorrectly accepted')
    except RuntimeError:pass
    assert not any('prediction' in x.lower() for x in before)
    # Synthetic pair filtering, including both/one-sided truncation and reused
    # parent IDs; use a dummy scorer and correct full denominator 5349.
    class S:
        @staticmethod
        def anls(ans,pred):return float(pred)
        @staticmethod
        def paired_bootstrap(a,b,g):return {'n_questions':len(a),'delta':float(np.mean(np.array(b)-np.array(a))), 'n_parents':len(set(g))}
    recs=[{'row_index':i,'parent_id':'p'+str(i//3),'answer':'unused'} for i in range(5349)]
    rs=[]
    for i in range(5349):
        for arm in ARMS:
            t=int((arm=='full' and i in (0,2)) or (arm=='b2' and i in (1,2)))
            rs.append((i,arm,'success','0.5',t,1024 if arm=='full' else 256,None))
    pairs=sensitivity(recs,rs,S)
    d=pairs['magnitude_v__seed_11 - full'];assert d['retained']==5347 and d['excluded_ref_only']==2
    d=pairs['b2 - b1'];assert d['retained']==5347 and d['excluded_target_only']==2
    assert all(x['paired']['delta']==0 for x in pairs.values())
    print('self-check passed: frozen prompt, strict completion-before-prediction, paired truncation denominator')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prompt','sensitivity','self-check'))
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.command=='prompt':make_prompts(a.output or OUT/'prompt_sidecar.json')
    elif a.command=='sensitivity':make_sensitivity(a.output or OUT/'truncation_sensitivity.json')
    else:self_check()

if __name__=='__main__':main()
