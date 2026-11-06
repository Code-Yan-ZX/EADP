"""Fixed E4-A smoke inventory, CPU only, no prediction or label access."""
import hashlib
from identity_worker import DATA, read_jsonl, select_panel

COUNTS={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}

def select_smoke(proof):
    # Reuse E1 input/image/panel and exclusion-proof checks first.
    select_panel(proof)
    old={r['group_id'] for r in read_jsonl(DATA/'anchor_exclusion/anchor_groups.jsonl')}
    rows=read_jsonl(proof['files']['inputs']['path']); selected=[]
    for task,n in COUNTS.items():
        candidates=[r for r in rows if r['task']==task and r['group_id'] not in old]
        candidates.sort(key=lambda r:(hashlib.sha256(('20261104|e4a|'+task+'|'+r['group_id']).encode()).hexdigest(),r['group_id'],str(r['qid'])))
        seen=set()
        for row in candidates:
            if row['group_id'] in seen:continue
            seen.add(row['group_id']);selected.append(row)
            if len(seen)==n:break
        if len(seen)!=n:raise ValueError('insufficient unique DEV groups for smoke')
    if len(selected)!=32 or len({r['group_id'] for r in selected})!=32:
        raise ValueError('smoke requires 32 distinct DEV groups')
    return selected
