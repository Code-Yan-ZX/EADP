#!/usr/bin/env python3
"""Wait for frozen full evaluation, then produce independent local reports.

No network or messages. Until the whole run is complete, SQL reads only
status/coverage/provenance metadata, never predictions or accuracy. Outputs go
to a new directory outside the frozen experiment's data and code directories.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback

HERE=Path(__file__).resolve().parent
QWEN=HERE.parents[1]
OUT=QWEN/'outputs/student_full'
AUXOUT=QWEN/'outputs/student_full_aux'
FREEZE=OUT/'run_freeze.json'
MANIFEST=OUT/'docvqa_val_manifest.json'
GATE=OUT/'dev_gate.json'
DB=OUT/'predictions.sqlite3'
SCORE=QWEN/'scripts/student_full/score_full.py'
AUX=HERE/'audit_aux.py'
ARMS=('full','b2','b1','rres512','magnitude_v__seed_11','magnitude_v__seed_23','magnitude_v__seed_37')
COUNT=5349


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def integrity():
    f=json.loads(FREEZE.read_text())
    if f['count']!=COUNT or f['config']['arms']!=list(ARMS):
        raise ValueError('frozen plan differs from expected full run')
    if f['manifest_sha256']!=sha(MANIFEST) or f['original_tsv_sha256']!=sha(f['original_tsv_path']):
        raise ValueError('frozen manifest or source dataset hash differs')
    for path,digest in f['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('frozen source hash differs: '+path)
    for arm,ck in f['checkpoints'].items():
        if sha(ck['path'])!=ck['sha256']:raise ValueError('student checkpoint hash differs: '+arm)
    for name in ('prior_eval_freeze','train_freeze','dev_manifest'):
        if sha(f[name])!=f[name+'_sha256']:raise ValueError('frozen input hash differs: '+name)
    gate=json.loads(GATE.read_text())
    if not gate.get('pass_all') or gate.get('run_freeze_sha256')!=sha(FREEZE):
        raise ValueError('development gate missing or belongs to another freeze')
    return f


def assess(con,freeze_hash,arms=ARMS,count=COUNT):
    """Return waiting/ready or raise on explicit error; never select predictions."""
    meta=dict(con.execute('SELECT key,value FROM meta'))
    if not meta:
        return 'waiting'  # writer can create tables before committing its meta row
    if meta.get('run_freeze_sha256')!=freeze_hash:
        raise ValueError('database freeze hash mismatch')
    groups=con.execute('SELECT arm,status,count(*),count(DISTINCT row_index),min(row_index),max(row_index) FROM results GROUP BY arm,status').fetchall()
    by_arm={}
    for arm,status,n,distinct,lo,hi in groups:
        if arm not in arms:raise ValueError('unexpected result arm: '+str(arm))
        if status!='success':raise RuntimeError(f'explicit current result error: {arm} status={status}')
        if n>count or distinct!=n or lo<0 or hi>=count:
            raise ValueError('invalid result row coverage: '+arm)
        by_arm[arm]=n
    sessions=con.execute('SELECT started_unix,finished_unix FROM sessions ORDER BY started_unix DESC LIMIT 1').fetchall()
    if sessions:
        start,finished=sessions[0]
        error=con.execute('SELECT row_index,arm,error FROM error_events WHERE created_unix>=? ORDER BY created_unix DESC LIMIT 1',(start,)).fetchone()
        if error:raise RuntimeError(f'explicit error event: row={error[0]} arm={error[1]} detail={error[2]}')
    if len(by_arm)<len(arms) or any(by_arm.get(a,0)!=count for a in arms):return 'waiting'
    if not sessions or sessions[0][1] is None:return 'waiting'  # writer finalizes its session after last row
    return 'ready'


def open_readonly(path):
    con=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True,timeout=0.2)
    con.execute('PRAGMA query_only=ON');con.execute('BEGIN')
    return con


def read_ready_status(path,freeze_hash):
    if not Path(path).exists():return 'waiting'
    try:
        con=open_readonly(path)
        try:return assess(con,freeze_hash)
        finally:con.rollback();con.close()
    except sqlite3.OperationalError as e:
        # A newly-created DB may not yet have tables; writer locks are transient.
        msg=str(e).lower()
        if any(x in msg for x in ('no such table','locked','busy','unable to open')):return 'waiting'
        raise


def snapshot_digest(path,freeze_hash):
    """Read predictions only after all 37443 rows are confirmed complete."""
    con=open_readonly(path)
    try:
        if assess(con,freeze_hash)!='ready':raise RuntimeError('database is no longer complete')
        h=hashlib.sha256();n=0
        for row in con.execute('SELECT row_index,arm,status,prediction,truncated,n_vis,error FROM results ORDER BY row_index,arm'):
            h.update((json.dumps(row,ensure_ascii=False)+'\n').encode());n+=1
        if n!=COUNT*len(ARMS):raise ValueError('unexpected snapshot size')
        return h.hexdigest()
    finally:con.rollback();con.close()


def write_json(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def watch_files(frozen):
    paths={str(FREEZE),str(MANIFEST),str(GATE),str(AUX),str(Path(__file__).resolve()),
           frozen['original_tsv_path'],frozen['prior_eval_freeze'],frozen['train_freeze'],frozen['dev_manifest']}
    paths.update(frozen['source_sha256'])
    paths.update(c['path'] for c in frozen['checkpoints'].values())
    return sorted(paths)


def file_stamps(paths):
    return {p:(os.stat(p).st_ino,os.stat(p).st_size,os.stat(p).st_mtime_ns) for p in paths}


def invoke(cmd,log):
    with log.open('w') as f:
        proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
    if proc.returncode!=0:raise RuntimeError(f'command exited {proc.returncode}; see {log}')


def produce(run_dir,freeze_hash):
    integrity()
    before=snapshot_digest(DB,freeze_hash)
    analysis_path=run_dir/'analysis.json'; sensitivity_path=run_dir/'truncation_sensitivity.json'
    invoke([sys.executable,str(SCORE),'--output',str(analysis_path)],run_dir/'score.log')
    after_score=snapshot_digest(DB,freeze_hash)
    if after_score!=before:raise RuntimeError('result snapshot changed during primary scoring')
    invoke([sys.executable,str(AUX),'sensitivity','--output',str(sensitivity_path)],run_dir/'sensitivity.log')
    after_sensitivity=snapshot_digest(DB,freeze_hash)
    if after_sensitivity!=before:raise RuntimeError('result snapshot changed during sensitivity scoring')
    analysis=json.loads(analysis_path.read_text());sens=json.loads(sensitivity_path.read_text())
    if not analysis.get('complete') or analysis['expected_results']!=COUNT*len(ARMS):raise ValueError('primary analysis incomplete')
    if set(analysis['arms'])!=set(ARMS) or any(analysis['arms'][a]['success']!=COUNT for a in ARMS):raise ValueError('primary coverage incomplete')
    if not sens.get('complete') or sens['run_freeze_sha256']!=freeze_hash or sens['ordered_db_rows_sha256']!=before:
        raise ValueError('sensitivity result does not match frozen database snapshot')
    if len(sens['comparisons'])!=13 or len(analysis['paired_changes'])!=13:
        raise ValueError('comparison inventory mismatch')
    lines=['# DocVQA VAL full-run local report','',
           'All 5,349 questions were included in each primary arm. ANLS and differences below are on a 0–100 scale.', '',
           '| Method | ANLS | Truncated |','| --- | ---: | ---: |']
    for arm in ARMS:
        a=analysis['arms'][arm];lines.append(f"| {arm} | {100*a['anls']:.3f} | {a['truncated']} |")
    lines += ['', 'Paired differences use the same question rows, source-parent bootstrap, 10,000 draws.', '',
              '| Comparison | Difference (points) | 95% interval |','| --- | ---: | ---: |']
    for name,pair in analysis['paired_changes'].items():
        lo,hi=pair['ci95'];lines.append(f"| {name} | {100*pair['delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines += ['', 'Truncation sensitivity is secondary: it removes questions for which either arm truncated; denominators can differ between pairs. The primary result retains every question.',
              '',f'Frozen run SHA256: `{freeze_hash}`', f'Ordered result snapshot SHA256: `{before}`',
              'Detailed per-question primary scores and all sensitivity denominators are in the adjacent JSON files.','']
    report=run_dir/'report.md';report.write_text('\n'.join(lines))
    return {'analysis_path':str(analysis_path),'analysis_sha256':sha(analysis_path),
            'sensitivity_path':str(sensitivity_path),'sensitivity_sha256':sha(sensitivity_path),
            'report_path':str(report),'report_sha256':sha(report),'ordered_db_rows_sha256':before}


def self_check():
    def db():
        c=sqlite3.connect(':memory:')
        c.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE results(row_index INTEGER,arm TEXT,status TEXT,prediction TEXT,PRIMARY KEY(row_index,arm)); CREATE TABLE sessions(started_unix REAL,finished_unix REAL); CREATE TABLE error_events(row_index INTEGER,arm TEXT,created_unix REAL,error TEXT);')
        c.execute("INSERT INTO meta VALUES('run_freeze_sha256','test')")
        return c
    arms=('a','b');c=db();trace=[];c.set_trace_callback(trace.append)
    assert assess(c,'test',arms,2)=='waiting'
    assert not any('prediction' in x.lower() for x in trace)
    c.execute('DELETE FROM meta');assert assess(c,'test',arms,2)=='waiting'
    c.execute("INSERT INTO meta VALUES('run_freeze_sha256','test')")
    c.execute('INSERT INTO sessions VALUES(1,NULL)')
    for a in arms:
        for i in range(2):c.execute('INSERT INTO results VALUES(?,?,?,?)',(i,a,'success','do not read'))
    assert assess(c,'test',arms,2)=='waiting'
    c.execute('UPDATE sessions SET finished_unix=2');assert assess(c,'test',arms,2)=='ready'
    try:assess(c,'wrong',arms,2);raise AssertionError('freeze mismatch accepted')
    except ValueError:pass
    c.execute("UPDATE results SET status='error' WHERE row_index=0 AND arm='a'")
    try:assess(c,'test',arms,2);raise AssertionError('error status accepted')
    except RuntimeError:pass
    c.execute("UPDATE results SET status='success'")
    c.execute("INSERT INTO error_events VALUES(0,'__prepare__',1.5,'bad')")
    try:assess(c,'test',arms,2);raise AssertionError('prepare failure accepted')
    except RuntimeError:pass
    c.execute('INSERT INTO sessions VALUES(3,4)');assert assess(c,'test',arms,2)=='ready'
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'not_yet.db';assert read_ready_status(p,'x')=='waiting'
        sqlite3.connect(p).close();assert read_ready_status(p,'x')=='waiting'
    assert not any('prediction' in x.lower() for x in trace)
    print('self-check passed: metadata-only wait, full coverage, writer completion, explicit errors, freeze binding, missing table')


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');p.add_argument('--interval',type=float,default=30.)
    args=p.parse_args()
    if args.self_check:self_check();return
    if args.interval<1:raise ValueError('interval must be >=1 second')
    AUXOUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run_dir=Path(tempfile.mkdtemp(prefix='postprocess_'+stamp+'_',dir=AUXOUT))
    status_path=run_dir/'status.json'
    start=time.time()
    try:
        frozen=integrity();fhash=sha(FREEZE)
        stable_aux=sha(AUX);stable_self=sha(__file__);stable_gate=sha(GATE)
        watched=watch_files(frozen);stamps=file_stamps(watched)
        print('waiting for complete evaluation; output:',run_dir,flush=True)
        while True:
            current=file_stamps(watched)
            if current!=stamps:
                integrity()
                if sha(FREEZE)!=fhash or sha(AUX)!=stable_aux or sha(__file__)!=stable_self or sha(GATE)!=stable_gate:
                    raise ValueError('freeze, gate, postprocessor, or sensitivity source changed while waiting')
                stamps=current
            if read_ready_status(DB,fhash)=='ready':break
            time.sleep(args.interval)
        outputs=produce(run_dir,fhash)
        state={'status':'complete','freeze_sha256':fhash,'elapsed_seconds':time.time()-start,
               'generator_sha256':sha(__file__),'outputs':outputs}
        write_json(status_path,state);print('complete',status_path,flush=True)
    except BaseException as e:
        write_json(status_path,{'status':'failed','error':str(e),'traceback':traceback.format_exc(),
                                'elapsed_seconds':time.time()-start,'generator_sha256':sha(__file__)})
        raise

if __name__=='__main__':main()
