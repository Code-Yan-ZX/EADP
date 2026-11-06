"""Completed full-panel CPU acceptance; no GT or scoring.

Non-anchor E3 tensors are represented by runtime checks and summaries, not
reconstructed. Independent review is additionally required before scoring.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
from full_imports import local
_gate=local('panel_gate');sha=_gate.sha;smoke_gate=_gate.smoke_gate;b_gate=_gate.b_gate
_runner=local('panel_runner');JOURNAL_FIELDS=_runner.JOURNAL_FIELDS;validate_against_e1=_runner.validate_against_e1;validate_endpoint=_runner.validate_endpoint
inputs=local('panel_worker').inputs
_contract=local('contract');validate_records=_contract.validate_records;e3_views=_contract.e3_views
check=local('validate_panel_record').check
sys.path.insert(0,str(BASE/'e4'))
from identity_worker import PLAN,WEIGHTS,check_e1

def read(path):return json.loads(Path(path).read_text())
def lines(path):
 data=Path(path).read_bytes()
 if data and not data.endswith(b'\n'):raise ValueError('partial journal')
 return [json.loads(x) for x in data.splitlines()]

def validate_exit(run,started,terminal,launch,ex,budget):
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid') or
     launch.get('controller',{}).get('pid')!=started.get('worker_pid') or launch.get('controller')!=ex.get('controller')):raise ValueError('supervisor/controller failed')
 for key,expected in (('timeout_seconds',budget['outer_timeout_seconds']),('stale_seconds',budget['journal_stale_seconds']),('snapshot_seconds',budget['snapshot_seconds']),('poll_seconds',budget['poll_seconds']),('grace_seconds',budget['grace_seconds']),('progress_paths',[str(run/'data/generated.jsonl')])):
  if started.get(key)!=expected:raise ValueError('supervision setting mismatch: '+key)
 cwd=Path(started.get('cwd','/'));cmd=started.get('command',[])
 if ([(cwd/x).resolve() for x in cmd if Path(x).name=='panel_launch.py']!=[(HERE/'panel_launch.py').resolve()] or cmd.count('--run-id')!=1 or run.name!='run_'+cmd[cmd.index('--run-id')+1]):raise ValueError('wrong supervisor command')
 if (launch.get('launcher_sha256')!=sha(HERE/'panel_launch.py') or launch.get('worker_sha256')!=sha(HERE/'panel_worker.py') or launch.get('budget_sha256')!=sha(HERE/'budget_plan.json') or
     launch['started_unix']+budget['outer_timeout_seconds']>budget['budget_end_unix']):raise ValueError('launch code/deadline mismatch')
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure') is not None or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]):raise ValueError('unclean container exit')
 dock=launch.get('command',[])
 if (dock.count(str(HERE/'panel_worker.py'))!=1 or dock.count('--out')!=1 or Path(dock[dock.index('--out')+1]).resolve()!=run or '--preflight-only' in dock):raise ValueError('wrong full worker/container command')
 container=ex['container_identity'];cid=container['container_id']
 from completion import alive
 for ident in (ex['controller'],container.get('host_init')):
  if not isinstance(ident,dict) or not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int) or ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident):raise ValueError('invalid/live process identity')
 ins=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=10))[0]
 if ins['Id']!=cid or ins['State']['Running'] or ins['State']['Pid']!=0 or ins['State']['ExitCode']!=0 or ins['State']['OOMKilled'] or ins['Image']!=container.get('image'):raise ValueError('real container not cleanly stopped')
 if ins['Config']['Cmd']!=dock[dock.index(ins['Config']['Image'])+1:]:raise ValueError('actual container command differs')

def main():
 p=argparse.ArgumentParser()
 for name in ('e1-acceptance','e1-started','e1-terminal','run','run-started','run-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();budget=read(HERE/'budget_plan.json');initial=BASE/'initial_plan.json'
 if sha(initial)!=budget['budget_source_sha256'] or read(initial)['budget_end_unix']!=budget['budget_end_unix']:raise ValueError('deadline source changed')
 validate_exit(run,read(a.run_started),read(a.run_terminal),read(run/'launch.json'),read(run/'exit.json'),budget)
 upstream={'smoke':smoke_gate(),'e4b':b_gate()};proof=check_e1(a.e1_acceptance,a.e1_started,a.e1_terminal);rows,refs=inputs(proof)
 frozen=read(run/'frozen.json');done=read(run/'worker_complete.json');inner=read(run/'data/worker_complete.json')
 if (frozen.get('GT_opened') is not False or frozen.get('scoring_performed') is not False or frozen.get('approval')!=local('panel_gate').execution_gate() or frozen.get('approval')!=read(run/'launch.json')['approval'] or frozen.get('upstream')!=upstream or frozen.get('e1_proof_sha256')!=sha(a.e1_acceptance) or
     frozen.get('plan_sha256')!=sha(PLAN) or frozen.get('full_plan_sha256')!=sha(HERE/'plan.json') or done!={'questions':384,'physical_generations':3840,'weight_stats_unchanged':True,'source_unchanged':True} or
     inner!={'questions':384,'physical_generations':3840,'GT_opened':False,'scoring_performed':False}):raise ValueError('full frozen completion mismatch')
 if set(frozen['code_sha256'])!=set(local('source_inventory').sources()):raise ValueError('full source inventory incomplete')
 for path,digest in frozen['code_sha256'].items():
  if sha(path)!=digest:raise ValueError('full source changed')
 identity=read(WEIGHTS);stats=frozen['weight_stats_before'];paths=[str(Path(identity['local_snapshot'])/('model-%05d-of-00004.safetensors'%i)) for i in range(1,5)]
 if frozen['weights_identity_sha256']!=sha(WEIGHTS) or [x['path'] for x in identity['files']]!=paths or set(stats)!=set(paths):raise ValueError('weight inventory mismatch')
 for item in identity['files']:
  s=Path(item['path']).stat()
  if s.st_size!=item['bytes'] or [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=stats[item['path']]:raise ValueError('weights stat changed')
 records=lines(run/'data/records.jsonl');journal=lines(run/'data/generated.jsonl');views=lines(run/'data/e3_views.jsonl');counts=validate_records(rows,records)
 if len(journal)!=3840 or len(views)!=384 or views!=e3_views(rows,records):raise ValueError('journal/alias denominator')
 anchors={(x['task'],str(x['qid'])) for x in frozen['anchors']};smoke_anchors=read(BASE/'e2_e3_smoke/run_001/frozen.json')['selected']
 if len(anchors)!=3 or frozen['anchors']!=smoke_anchors:raise ValueError('diagnostic anchors changed')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(read(PLAN)['recipe']['model_path'],local_files_only=True,trust_remote_code=False);array_hashes={}
 for row,record,j in zip((x for x in rows for _ in range(10)),records,journal):
  if j!={k:record[k] for k in JOURNAL_FIELDS}:raise ValueError('actual generated journal differs')
  key=(row['task'],str(row['qid']));ref=refs[(*key,'P3-D1')];validate_against_e1(record,row,ref)
  if record['family']=='E2' and record['arm'] in ('000','111'):validate_endpoint(record,refs[(*key,'P3-D0' if record['arm']=='000' else 'P3-D1')])
  if tok.decode(record['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=record['answer']:raise ValueError('token decoding differs')
  arc=record['arrays']
  if key in anchors:
   path=Path(arc['path']).resolve()
   if arc['kind']!='full_arrays' or path.parent!=(run/'data/arrays').resolve() or path in array_hashes or sha(path)!=arc['sha256']:raise ValueError('anchor archive changed')
   array_hashes[path]=arc['sha256']
   with np.load(path,allow_pickle=False) as arrays:check(record,arrays)
  else:
   if arc['kind']!='summary_only':raise ValueError('non-anchor archive policy')
   check(record)
 if len(array_hashes)!=30 or set(array_hashes)!={p.resolve() for p in (run/'data/arrays').glob('*.npz')}:raise ValueError('anchor array inventory')
 files=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.run_started,a.run_terminal]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl','data/e3_views.jsonl')]
 result=dict(status='passed',**counts,upstream=upstream,files_sha256={str(p.resolve()):sha(p) for p in files},array_sha256={str(k):v for k,v in array_hashes.items()},verifier_sha256=sha(__file__),limits=['non-anchor high-dimensional tensors cannot be reconstructed offline','no performance or scoring interpretation','acceptance rechecks shard stats, not full payload hashes'])
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
