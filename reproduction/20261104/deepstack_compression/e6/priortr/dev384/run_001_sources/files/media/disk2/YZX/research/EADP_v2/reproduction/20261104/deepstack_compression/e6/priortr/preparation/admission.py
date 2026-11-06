"""PriorTR DEV-only deterministic small-sample admission, stdlib CPU only."""
import hashlib,json
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parents[3]; E1=BASE/'e1'; DATA=BASE/'data_manifest'
COUNTS={'DocVQA_VAL':3,'TextVQA_VAL':3,'AI2D_TEST':2}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def jsonl(path):
    raw=Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('truncated journal')
    return [json.loads(x) for x in raw.splitlines()]
def select(inputs,panel,old):
    dev={(r['task'],str(r['qid'])):r for r in panel if r.get('split')=='DEV'}
    if len(dev)!=384 or len(inputs)!=384 or Counter(r['task'] for r in dev.values())!={t:128 for t in COUNTS}:raise ValueError('DEV inventory')
    if {(r['task'],str(r['qid'])) for r in inputs}!=set(dev):raise ValueError('input inventory')
    rows=[]
    for task,n in COUNTS.items():
        c=[r for r in inputs if r['task']==task and r['group_id'] not in old]
        c.sort(key=lambda r:(hashlib.sha256(('20261104|e6-priortr|'+task+'|'+r['group_id']).encode()).hexdigest(),r['group_id'],str(r['qid'])))
        seen=set()
        for r in c:
            if r['group_id'] in seen:continue
            key=(task,str(r['qid']))
            if r['group_id']!=dev[key]['group_id']:raise ValueError('wrong image group')
            expected=hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            if r['message_sha256']!=expected or r['image_sha256']!=sha(r['image_path']):raise ValueError('input/image mismatch')
            seen.add(r['group_id']);rows.append(r)
            if len(seen)==n:break
        if len(seen)!=n:raise ValueError('insufficient groups')
    if len({r['group_id'] for r in rows})!=8:raise ValueError('overlapping image groups')
    return rows

def admit():
    proof_path=E1/'e1full001_completion.json';proof=json.loads(proof_path.read_text());review_path=BASE/'e1_independent_review/e1full001_review.json';review=json.loads(review_path.read_text())
    if proof.get('status')!='passed' or proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or proof.get('natural_questions')!=384 or proof.get('natural_records')!=1536:raise ValueError('E1 completion gate')
    if review.get('status')!='passed' or review.get('n_verified')!=384 or review.get('completion_proof_sha256')!=sha(proof_path):raise ValueError('independent review gate')
    for path in (E1/'monitorverifye1full001/terminal.json',BASE/'e1_independent_review/monitor_e1full001/terminal.json'):
        t=json.loads(path.read_text())
        if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('E1 verifier did not exit cleanly')
    for entry in proof['files'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('E1 accepted artifact changed')
    summary=DATA/'anchor_exclusion/summary.json';verify=DATA/'anchor_exclusion/verification.json';s=json.loads(summary.read_text());v=json.loads(verify.read_text())
    if v.get('status')!='pass' or v.get('summary_sha256')!=sha(summary) or v.get('verifier_sha256')!=sha(DATA/'anchor_exclusion/verify.py') or s.get('outputs_sha256',{}).get('anchor_groups.jsonl')!=sha(DATA/'anchor_exclusion/anchor_groups.jsonl'):raise ValueError('anchor exclusion gate')
    old={r['group_id'] for r in jsonl(DATA/'anchor_exclusion/anchor_groups.jsonl')}
    if len(old)!=16:raise ValueError('anchor inventory')
    inputs=jsonl(proof['files']['inputs']['path']);panel=jsonl(proof['files']['panel']['path'])
    return proof,select(inputs,panel,old)
