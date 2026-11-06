"""Read-only cross-check of full verifier against accepted 42 intervention arms."""
import hashlib,json
from pathlib import Path
import numpy as np
from full_imports import local
check=local('validate_panel_record').check;HERE=Path(__file__).resolve().parent;SMOKE=HERE.parent/'e2_e3_smoke';run=SMOKE/'run_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((SMOKE/'run_001_verification.json').read_text());assert proof['status']=='passed' and not proof['GT_opened']
records=[json.loads(x) for x in (run/'records.jsonl').read_text().splitlines()];records=[x for x in records if x['family'] in ('E2','E3')];assert len(records)==42
for r in records:
 p=Path(r['arrays']['path']);assert sha(p)==r['arrays']['sha256']
 with np.load(p,allow_pickle=False) as arrays:check(r,arrays)
out=HERE/'probe_smoke001.json';assert not out.exists();out.write_text(json.dumps({'status':'passed','intervention_records_verified':len(records),'formal_smoke_sha256':sha(SMOKE/'run_001_verification.json'),'checker_sha256':sha(HERE/'validate_panel_record.py'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False},indent=2)+'\n');print('passed',len(records),sha(out))
