"""Bind the independently reviewed 192+30 DEV selection before any reference run."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_multibudget_expansion192';I=B/'e1_independent_review'
EXPECTED='a1142f412a4fd384b2683f068c345651962f98f2386fe9829a650f0548e915c9'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def inputs(p):
 rows=p['measurement_identities']+p['warmup_identities']
 if len(p['measurement_identities'])!=192 or len(p['warmup_identities'])!=30 or len(rows)!=222 or len({(r['task'],str(r['qid'])) for r in rows})!=222 or len({(r['task'],r['group_id']) for r in rows})!=222 or any(r.get('split')!='DEV' for r in rows):raise ValueError('proposal inputs')
 return rows
def load():
 path=D/'proposal002.json';proof=I/'e4c_expansion192_proposal_review.json';reviewer=I/'review_e4c_expansion192_proposal.py';monitor=I/'monitor_e4c_expansion192_proposal'
 if sha(path)!=EXPECTED:raise ValueError('proposal changed')
 p=json.loads(path.read_text());q=json.loads(proof.read_text());s=json.loads((monitor/'started.json').read_text());t=json.loads((monitor/'terminal.json').read_text())
 cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or sum((cwd/x).resolve()==reviewer.resolve() for x in cmd)!=1:raise ValueError('proposal independent supervisor')
 if p.get('GPU_authorized') is not False or p.get('GT_opened') is not False or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('proposal_sha256')!=sha(path) or q.get('reviewer_sha256')!=sha(reviewer) or q.get('measurement_images')!=192 or q.get('warmup_images')!=30 or q.get('jobs')!=7920 or q.get('fixed32_reference_arms')!=888 or q.get('GT_opened') is not False or q.get('expansion_predictions_opened') is not False or q.get('GPU_authorized') is not False or q.get('errors')!=[]:raise ValueError('proposal proof')
 return inputs(p)
