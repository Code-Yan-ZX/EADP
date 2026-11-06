"""Extract 384 E1 identity records only after upstream proof admission."""
import hashlib
import json
from pathlib import Path

FIELDS = ('task','qid','group_id','message_sha256','arm','forced','processor',
          'visual_sha256','deepstack_full_sha256','original_position_ids',
          'full_rope_delta','ordered_ids','generation_scalars','eos_ids')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def load(path, receipt, rows):
    if sha(path) != receipt.journal_sha: raise ValueError('accepted journal changed')
    requested = {(r['task'],str(r['qid'])):r for r in rows}
    if len(rows) != 384 or len(requested) != 384 or any(r.get('split') != 'DEV' for r in rows):
        raise ValueError('expected all unique DEV inputs')
    found = {k:[] for k in requested}
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line); key = (row.get('task'),str(row.get('qid')))
            if key in found and row.get('arm') == 'P3-D1' and row.get('forced') is False:
                if any(row.get(k) != requested[key].get(k) for k in ('task','group_id','message_sha256')) or any(k not in row for k in FIELDS):
                    raise ValueError('E1 identity mismatch')
                found[key].append({k:row[k] for k in FIELDS})
    if sha(path) != receipt.journal_sha or any(len(v) != 1 for v in found.values()):
        raise ValueError('ambiguous or changed E1 records')
    return found
