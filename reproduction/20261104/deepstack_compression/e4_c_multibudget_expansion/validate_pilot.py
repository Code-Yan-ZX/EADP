"""Rebuild the CPU-only pilot plan and attest the explicit no-GPU lock."""
import argparse,json,os,time
from pathlib import Path
import gate
from pilot_contract import TASKS,build
D=Path(__file__).resolve().parent;B=D.parent
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();path=D/'pilot_plan001.json';p=json.loads(path.read_text());gate.check_plan(p,time.time(),7200,3600)
source=json.loads((B/'e4_c_latency_full/full_001/frozen.json').read_text())['schedule']['cohort'];mapping={(r['task'],r['group_id']):r for r in p['input_identities']}
# The accepted cohort has 222 rows; read only the answer-free frozen input list.
inputs={}
for line in (B/'e1/input_preparation_002/inputs.jsonl').open():
 row=json.loads(line);inputs[(row['task'],row['group_id'])]={k:row[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')}
inv,jobs=build(source,inputs)
if p['cohort']!=inv or p['jobs']!=jobs or len(mapping)!=62 or any(mapping[k]!=inputs[k] for k in mapping) or set(mapping)!={(task,g) for task in TASKS for phase in ('measured','warmup') for g in inv[task][phase]}:raise ValueError('pilot reconstruction mismatch')
try:gate.require_gpu()
except RuntimeError as e:
 if 'GPU LOCKED' not in str(e):raise
else:raise ValueError('unexpected GPU authorization')
out={'status':'passed','scope':'planning only; no GPU authorization','manifest_sha256':gate.sha(path),'validator_sha256':gate.sha(__file__),'n_input_identities':len(mapping),'n_measured_images':32,'n_warmup_images':30,'n_jobs':len(jobs),'warmup_jobs':240,'measured_jobs':1280,'GPU_authorized':False,'GT_opened':False,'predictions_opened':False}
with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
