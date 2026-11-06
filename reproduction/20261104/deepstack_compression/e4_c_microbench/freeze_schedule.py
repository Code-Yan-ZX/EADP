"""Bind preregistered microbenchmark groups to accepted DEV identities only."""
import hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;TIMING=BASE/'e4_c_timing'
sys.path.insert(0,str(TIMING))
from input_preflight import load_accepted_e1_references
from contract import make_schedule
from microbench_contract import make_microbench_schedule

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()

def main():
 e1=BASE/'e1';proof=e1/'e1full001_completion.json';started=e1/'monitorverifye1full001/started.json';terminal=e1/'monitorverifye1full001/terminal.json'
 sources=[HERE/'freeze_schedule.py',TIMING/'input_preflight.py',TIMING/'e1_reference_gate.py',TIMING/'contract.py',TIMING/'microbench_contract.py']
 frozen={str(p):sha(p) for p in sources};rows,refs=load_accepted_e1_references(proof,started,terminal)
 oldpath=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl';old={json.loads(x)['group_id'] for x in oldpath.read_text().splitlines() if x.strip()}
 if len(old)!=16:raise ValueError('anchor exclusions changed')
 timing=make_schedule(rows,old);micro=make_microbench_schedule(timing);bygroup={(r['task'],r['group_id']):r for r in rows};identities=[]
 for task,group in micro['images']:
  r=bygroup[(task,group)];ref=refs[(task,str(r['qid']))]
  if r['split']!='DEV' or any(r.get(k)!=ref.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('input identity mismatch')
  identities.append({k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')}|{'processor':ref['processor'],'image_start':ref['image_start']})
 if len(identities)!=32 or any(sha(k)!=v for k,v in frozen.items()):raise ValueError('identity or source changed')
 payload={'status':'prepared','scope':'DEV inputs and component schedule only; no GT, scoring, GPU or performance','schedule':micro,'input_identities':identities,
          'source_sha256':frozen,'e1_acceptance_sha256':sha(proof),'e1_started_sha256':sha(started),'e1_terminal_sha256':sha(terminal),'excluded_anchor_groups_sha256':sha(oldpath)}
 with (HERE/'schedule001.json').open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'prepared','images':len(identities),'jobs':len(micro['jobs'])}))
if __name__=='__main__':main()
