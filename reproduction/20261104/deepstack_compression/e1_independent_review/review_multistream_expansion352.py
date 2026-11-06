"""Independently rebuild the post-hoc metadata cohort; never read features/scores."""
import argparse, hashlib, json, os, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_multistream_expansion352';I=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('supervision')
 return s,t
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);a=a.parse_args();out=a.out.resolve()
 if out!=(I/'multistream_expansion352_review.json').resolve():raise ValueError('output')
 fs,ft=terminal(D/'monitor_freeze001',D/'freeze_plan.py');ss,st=terminal(D/'monitor_selection001',D/'select_metadata.py')
 if ft['utc']>=ss['created_utc']:raise ValueError('selection preceded rule freeze')
 plan=read(D/'plan_freeze001.json');sel=read(D/'selection001.json');sources=plan['source_sha256']
 if plan.get('status')!='frozen' or any(sha(k)!=v for k,v in sources.items()) or sel.get('status')!='frozen' or sel.get('plan_freeze_sha256')!=sha(D/'plan_freeze001.json') or sel.get('selection_source_sha256')!=sha(D/'selection.py') or sel.get('runner_sha256')!=sha(D/'select_metadata.py') or sel.get('errors')!=[]:raise ValueError('source chain')
 for k in ('GT_opened','predictions_opened','feature_values_opened','score_values_opened','GPU_used'):
  if sel.get(k) is not False:raise ValueError('selection disclosure')
 fields=('task','qid','group_id','image_sha256','message_sha256');clean=lambda x:{k:str(x[k]) for k in fields};key=lambda r:(r['task'],r['qid'])
 inputs=B/'e1/input_preparation_002/inputs.jsonl';old_path=B/'e5/selector_real32/sample_manifest_001.json'
 rows=[clean(json.loads(x)) for x in inputs.read_text().splitlines() if x.strip()];old=[clean(x) for x in read(old_path)['pilot']];tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');mapping={key(r):r for r in rows}
 if len(rows)!=384 or len(mapping)!=384 or Counter(r['task'] for r in rows)!={t:128 for t in tasks} or len(old)!=32 or len({key(r) for r in old})!=32 or any(mapping.get(key(x))!=x for x in old) or any(len({r['group_id'] for r in rows if r['task']==t})!=128 for t in tasks):raise ValueError('input/old identity')
 prior={key(x) for x in old};order=lambda r:(hashlib.sha256(('multistream-expansion352-pilot-v1|'+r['task']+'|'+r['group_id']+'|'+r['qid']).encode()).hexdigest(),r['qid'])
 remain=[r for t in tasks for r in sorted((x for x in rows if x['task']==t and key(x) not in prior),key=order)];pilot=[x for t in tasks for x in [r for r in remain if r['task']==t][:3]]
 counts=dict(Counter(x['task'] for x in remain));pilot_counts=dict(Counter(x['task'] for x in pilot))
 if counts!={'DocVQA_VAL':117,'TextVQA_VAL':117,'AI2D_TEST':118} or pilot_counts!={x:3 for x in tasks} or remain!=sel['remaining'] or pilot!=sel['pilot']:raise ValueError('cohort or hash order')
 group_tasks=defaultdict(set)
 for r in rows:group_tasks[r['group_id']].add(r['task'])
 shared=sum(len(ts)>1 for ts in group_tasks.values());old_groups={x['group_id'] for x in old};remaining_overlap=sum(x['group_id'] in old_groups for x in remain)
 if shared!=sel['cross_task_shared_group_count'] or remaining_overlap!=sel['remaining_rows_in_pilot_groups']:raise ValueError('group disclosures')
 test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_selection.py','-v'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
 if test.returncode or any(sha(k)!=v for k,v in sources.items()):raise ValueError('test or source drift')
 v={'status':'passed','independent_review':True,'scope':'metadata-only retrospective cohort and pilot','counts':counts,'pilot_counts':pilot_counts,'cross_task_shared_groups':shared,'remaining_in_old_groups':remaining_overlap,'remaining_distinct_groups':len({r['group_id'] for r in remain}),'plan_sha256':sha(D/'plan_freeze001.json'),'selection_sha256':sha(D/'selection001.json'),'freeze_started_sha256':sha(D/'monitor_freeze001/started.json'),'freeze_terminal_sha256':sha(D/'monitor_freeze001/terminal.json'),'selection_started_sha256':sha(D/'monitor_selection001/started.json'),'selection_terminal_sha256':sha(D/'monitor_selection001/terminal.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'feature_values_opened':False,'score_values_opened':False,'training_performed':False,'GPU_used':False,'errors':[],'limitations':['post-hoc same DEV; not held-out or confirmatory','selection excludes old questions, not cross-task groups; this manifest currently has zero cross-task group overlap','selection runner and norm feature code were not part of the pre-selection frozen source set; independent identity reconstruction performed and future extraction requires a separately frozen full execution chain','future model capture has real computation and extraction cost; no cost or effectiveness measured here']}
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
