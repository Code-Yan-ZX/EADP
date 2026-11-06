"""CPU-only reconstruction of component schedule and accepted DEV identity.

No GPU, model, predictions, truth or scoring. This does not authorize running
the profiler; real-model identity and independently reviewed lifecycle remain
required.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from freeze_schedule import BASE,HERE,TIMING,sha,load_accepted_e1_references,make_schedule,make_microbench_schedule

def reconstruct():
 frozen_path=HERE/'schedule001.json';frozen=json.loads(frozen_path.read_text());e1=BASE/'e1'
 proof=e1/'e1full001_completion.json';started=e1/'monitorverifye1full001/started.json';terminal=e1/'monitorverifye1full001/terminal.json';anchor=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl'
 for field,path in [('e1_acceptance_sha256',proof),('e1_started_sha256',started),('e1_terminal_sha256',terminal),('excluded_anchor_groups_sha256',anchor)]:
  if frozen.get(field)!=sha(path):raise ValueError('accepted input proof changed: '+field)
 if any(sha(path)!=digest for path,digest in frozen['source_sha256'].items()):raise ValueError('schedule source changed')
 rows,refs=load_accepted_e1_references(proof,started,terminal)
 old={json.loads(x)['group_id'] for x in anchor.read_text().splitlines() if x.strip()}
 if len(old)!=16:raise ValueError('anchor group inventory changed')
 rebuilt=make_microbench_schedule(make_schedule(rows,old));bygroup={(r['task'],r['group_id']):r for r in rows};identities=[]
 for task,group in rebuilt['images']:
  r=bygroup[(task,group)];ref=refs[(task,str(r['qid']))]
  if r['split']!='DEV' or any(r.get(k)!=ref.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('DEV reference mismatch')
  identities.append({k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')}|{'processor':ref['processor'],'image_start':ref['image_start']})
 if json.dumps(rebuilt,sort_keys=True)!=json.dumps(frozen['schedule'],sort_keys=True) or identities!=frozen['input_identities']:raise ValueError('rebuilt schedule or inputs differ')
 return {'images':len(identities),'jobs':len(rebuilt['jobs']),'schedule_sha256':sha(frozen_path),'accepted_sha256':sha(proof)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();out=HERE/('cpu_preflight'+a.run_id);out.mkdir(exist_ok=False)
 names=('preflight.py','worker.py','capture.py','runner.py','test_worker.py','test_capture.py','test_runner.py','freeze_schedule.py','schedule001.json')
 paths=[HERE/n for n in names]+[TIMING/n for n in ('contract.py','microbench_contract.py','input_preflight.py','e1_reference_gate.py')]+[BASE/'e4_c_latency/light_adapter.py',BASE/'e4/skip_deepstack.py']
 sources={str(path):sha(path) for path in paths};details=reconstruct()
 proc=subprocess.run([sys.executable,'-m','unittest','-v','test_capture','test_runner','test_worker'],cwd=HERE,text=True,capture_output=True,timeout=120)
 (out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr)
 if proc.returncode or 'Ran 9 tests' not in proc.stderr or any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('CPU tests failed or source changed')
 result={'status':'passed','scope':'CPU-only; no GPU, model, predictions, truth or scoring; profiler remains locked','details':details,'tests':9,'source_sha256':sources,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log')}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed',**details}))
if __name__=='__main__':main()
