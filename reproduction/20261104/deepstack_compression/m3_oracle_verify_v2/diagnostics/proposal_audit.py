import json,sys,collections
from pathlib import Path
root=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');sys.path.insert(0,str(root/'m3_oracle_exec'));import search
panel=root/'m3_oracle_exec/runtime_001/panel';c=collections.Counter();versions=collections.Counter();n=0
for f in sorted([p for p in panel.glob('*.jsonl') if len(p.stem)==64]):
 start=grad=None
 with f.open('rb') as handle:
  for line in handle:
   if line.startswith(b'{"event":"question_started",'):start=json.loads(line)
   elif line.startswith(b'{"event":"full_gradient_completed",'):grad=json.loads(line)
 if not start or not grad:raise RuntimeError('missing metadata')
 logged=start['proposal'];replayed=search.p.proposals(start['qid'],logged['greedy_order'],grad['utility']);versions[start['python_version'].split()[0]]+=1;n+=1
 if logged!=replayed:
  c['mismatch']+=1
  for name in ('MAIN','MULTI','RANDOM'):
   if logged['pools'][name]!=replayed['pools'][name]:c['pool_'+name]+=1
  for r in ('8','16'):
   for name in ('MAIN','MULTI','RANDOM'):
    if logged['by_r'][r]['sources'][name]!=replayed['by_r'][r]['sources'][name]:c['source_'+r+'_'+name]+=1
print('files',n,'recorded_python_versions',dict(versions),'audit_python',sys.version.split()[0],'aggregate_differences',dict(c))
