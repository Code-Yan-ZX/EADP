"""Freeze proposed 32-image/10-warm-per-task schedule without predictions or GT."""
import hashlib,json,os
from pathlib import Path
from pilot_contract import TASKS,build
from gate import sha
D=Path(__file__).resolve().parent;B=D.parent;R=B/'e1_independent_review';F=B/'e4_c_latency_full';E=B/'e1'
def read(p):return json.loads(Path(p).read_text())
def terminal(d):
 s=read(d/'started.json');t=read(d/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('unclean '+str(d))
formal=F/'full_001_verification.json';review=R/'e4c_latency_full001_review.json';f=read(formal);r=read(review);frozen=F/'full_001/frozen.json';inputs=E/'input_preparation_002/inputs.jsonl';e1=read(E/'e1full001/frozen.json')
for mon in (F/'monitor_full001',F/'monitor_verify_full001',R/'monitor_e4c_latency_full001'):terminal(mon)
if f['status']!='passed' or f['n_jobs']!=5940 or f['GT_opened'] is not False or f['scoring_performed'] is not False or f['evidence_sha256'].get(str(frozen.resolve()))!=sha(frozen) or r['status']!='passed' or r['independent_review'] is not True or r['formal_proof_sha256']!=sha(formal) or r['n_jobs']!=5940 or r['errors']!=[] or sha(inputs)!=e1['inputs_sha256']:raise ValueError('cohort source not accepted')
x=read(frozen)['schedule'];cohort=x['cohort'];items={}
for line in inputs.open():
 row=json.loads(line);key=(row['task'],row['group_id'])
 if key in items:raise ValueError('duplicate frozen input group')
 if hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=row['message_sha256']:raise ValueError('prompt hash mismatch')
 items[key]={k:row[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')}
inv,jobs=build(cohort,items);needed={(t,g) for t in TASKS for phase in ('measured','warmup') for g in inv[t][phase]};selected=[items[k] for k in sorted(needed)]
sources=[D/'pilot_contract.py',D/'gate.py',D/'test_contract.py',Path(__file__),B/'initial_plan.json',formal,review,frozen,inputs,E/'e1full001/frozen.json']
for m in (F/'monitor_full001',F/'monitor_verify_full001',R/'monitor_e4c_latency_full001'):
 sources.extend((m/'started.json',m/'terminal.json'))
plan={'stage':'proposed post-hoc 32-image four-budget pilot; CPU-only','status':'prepared_unreviewed','selection':'hash-rank from previously accepted 64-image DEV latency cohort by task; warmup from its disjoint ten-image cohort','input_identities':selected,'cohort':inv,'jobs':jobs,'measured_jobs':1280,'warmup_jobs':240,'GT_opened':False,'predictions_opened':False,'GPU_authorized':False,'source_sha256':{str(p.resolve()):sha(p) for p in sources},'budget_source_sha256':sha(B/'initial_plan.json'),'limits':['selection and pilot proposed after previous DEV and timing observations','no real model timing worker or actual no-GPU container preflight exists','fixed32 path and clean end-to-end boundary require implementation and independent identity review','future full-panel results must not automatically reuse pilot measurements']}
out=D/'pilot_plan001.json'
with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('prepared',len(selected),'identities',len(jobs),'jobs',sha(out))
