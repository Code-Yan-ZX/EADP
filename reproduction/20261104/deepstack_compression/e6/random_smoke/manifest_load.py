"""Bind all frozen random choices to frozen plan before model execution."""
import hashlib,json
from pathlib import Path
SEEDS=(20261104,20261105,20261106)
def load(path,plan):
 p=Path(path);data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=plan['selection_manifest_sha256']:raise ValueError('selection manifest hash')
 rows=[json.loads(x) for x in data.splitlines() if x.strip()];out={}
 for row in rows:
  key=(row['task'],str(row['qid']),row['seed'])
  if key in out or row['seed'] not in SEEDS:raise ValueError('duplicate or unknown seed')
  out[key]=row
 if len(rows)!=1152 or len(out)!=1152 or len({(a,b) for a,b,_ in out})!=384:raise ValueError('manifest incomplete')
 for t,q in {(a,b) for a,b,_ in out}:
  if any((t,q,s) not in out for s in SEEDS):raise ValueError('missing seed')
 return out
