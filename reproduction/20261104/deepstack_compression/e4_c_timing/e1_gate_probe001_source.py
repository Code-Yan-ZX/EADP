"""CPU-only positive E1 identity gate integration; never reads E4 output or GT."""
import hashlib,json,os
from pathlib import Path
from input_preflight import load_accepted_e1_references
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
paths=[BASE/'e1/e1full001_completion.json',BASE/'e1/monitorverifye1full001/started.json',BASE/'e1/monitorverifye1full001/terminal.json']
rows,refs=load_accepted_e1_references(*paths)
if len(rows)!=384 or len(refs)!=384 or any(r.get('split')!='DEV' for r in rows):raise ValueError('E1 DEV inventory invalid')
source=[HERE/'input_preflight.py',HERE/'e1_reference_gate.py',Path(__file__)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
items=sorted(refs.values(),key=lambda r:(r['task'],r['qid']))
payload={'status':'passed','n_dev':len(rows),'n_reference':len(refs),'source_sha256':{str(p):sha(p) for p in source},'proof_sha256':{str(p):sha(p) for p in paths},'references_sha256':hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'scope':'accepted E1 DEV identity only; no E4 predictions, GT, GPU, or score'}
out=HERE/'e1_gate_probe001.json'
with out.open('x') as f:json.dump(payload,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'status':'passed','n_dev':len(rows)}))
