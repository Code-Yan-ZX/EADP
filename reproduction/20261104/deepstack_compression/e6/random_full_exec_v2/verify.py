"""Predeclared no-GT completion acceptance; requires actual stopped full run."""
import argparse,json,subprocess,sys
from pathlib import Path
import gate
from worker import load
from verify_contract import check_record,inventory as check_inventory
D=Path(__file__).resolve().parent;B=D.parents[1];LAT=B/'e4_c_latency';RS=B/'e6/random_smoke';RUN=D/'run_001'
def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 if args.out.exists():raise ValueError('report already exists')
 plan=gate.plan();approval=gate.admit(False);vp=gate.read(D/'verification_plan001.json')
 if vp.get('runtime_plan_sha256')!=gate.sha(D/'runtime_plan001.json') or vp.get('verifier_sha256')!=gate.sha(__file__) or vp.get('contract_sha256')!=gate.sha(D/'verify_contract.py') or vp.get('snapshot_sha256')!=gate.sha(D/'snapshot.py') or str(args.out.resolve())!=vp.get('report'):raise ValueError('verification plan mismatch')
 terminal=gate.terminal(D/'monitor_run001/terminal.json');gate.terminal(D/'monitor_snapshot001/terminal.json')
 started=gate.read(D/'monitor_run001/started.json');launch=gate.read(RUN/'launch.json');ex=gate.read(RUN/'exit.json');snap=gate.read(D/'run_001_sources/manifest.json')
 cwd=Path(started['cwd']);cmd=started['command']
 if sum((cwd/x).resolve()==(D/'launch.py').resolve() for x in cmd)!=1:raise ValueError('run supervisor command')
 if (terminal.get('worker_pid')!=started.get('worker_pid') or launch.get('controller',{}).get('pid')!=started.get('worker_pid') or launch.get('approval_bundle')!=approval or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or not ex.get('container_identity') or ex.get('gpu_pids_after')!=[]):raise ValueError('actual run or container exit')
 cid=ex['container_identity']['container_id'];doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
 if doc['Id']!=cid or doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['State']['OOMKilled'] is not False or doc['HostConfig']['NetworkMode']!='none':raise ValueError('docker inspect failed')
 if launch.get('worker_sha256')!=gate.sha(D/'worker.py') or launch.get('launcher_sha256')!=gate.sha(D/'launch.py') or launch.get('run_out')!=str(RUN):raise ValueError('launcher source or output identity')
 if snap.get('status')!='passed' or snap.get('runtime_plan_sha256')!=gate.sha(D/'runtime_plan001.json') or set(snap.get('sources',{}))!=set(plan['source_sha256']):raise ValueError('source snapshot inventory')
 for name,item in snap['sources'].items():
  if item.get('sha256')!=plan['source_sha256'][name] or gate.sha(item['snapshot'])!=item['sha256']:raise ValueError('source snapshot mismatch')
 completed=gate.read(RUN/'worker_complete.json');provenance=gate.read(RUN/'completion_provenance.json')
 if completed.get('status')!='completed' or completed.get('questions')!=384 or completed.get('natural_generations')!=1152 or completed.get('GT_opened') is not False or completed.get('scoring_performed') is not False or provenance.get('status')!='passed' or provenance.get('weight_stats_before')!=provenance.get('weight_stats_after') or provenance.get('source_sha256')!=plan['source_sha256']:raise ValueError('completion provenance')
 sys.path[:0]=[str(LAT)]
 for name in ('input_preflight','real_worker','latency_gate','runner','contract'):
  old=sys.modules.get(name)
  if old is not None and (not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=(LAT/(name+'.py')).resolve()):raise ValueError('foreign module '+name)
 from input_preflight import load_accepted_e1_references
 from real_worker import references_to_rows
 e1=B/'e1';rows,prefs=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');rows=references_to_rows(rows,prefs)
 admission=load('_random_full_verify_e1',B/'m3_oracle_exec/e1_admission.py');refs=load('_random_full_verify_refs',D/'references.py').load(e1/'e1full001/raw_predictions.jsonl',admission.validate(metadata_only=False),rows)
 choices=load('_random_full_verify_manifest',RS/'manifest_load.py').load(RS/'selection_manifest001.jsonl',gate.read(RS/'plan001.json'))
 inv=load('_random_full_verify_inventory',B/'e6/random_full/inventory.py');schedule=inv.schedule(rows,choices)
 records=lines(RUN/'records.jsonl');generated=lines(RUN/'generated.jsonl');journal=lines(RUN/'journal.jsonl');check_inventory(records,generated,journal,schedule)
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(gate.read(ROOT_PLAN)['recipe']['model_path'],local_files_only=True)
 decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 for record,(row,seed) in zip(records,schedule):
  key=(row['task'],str(row['qid']),seed);check_record(record,choices[key],refs[key[:2]][0],decode)
 files=[RUN/x for x in ('launch.json','exit.json','model_ready.json','completion_provenance.json','worker_complete.json','journal.jsonl','generated.jsonl','records.jsonl')]+[D/'run_001_sources/manifest.json']
 proof={'status':'passed','stage':'random_full_dev','questions':384,'generations':1152,'seeds':list(inv.SEEDS),'GT_opened':False,'predictions_opened':True,'scoring_performed':False,'timing_eligible':False,'files_sha256':{str(x):gate.sha(x) for x in files},'runtime_plan_sha256':gate.sha(D/'runtime_plan001.json'),'verifier_sha256':gate.sha(__file__),'run_started_sha256':gate.sha(D/'monitor_run001/started.json'),'run_terminal_sha256':gate.sha(D/'monitor_run001/terminal.json'),'errors':[],'limitations':['summary-only tensor evidence; no independent tensor-byte reconstruction','accepted E1 DEV mixed predictions parsed for identity','author greedy also runs; timing is not a performance comparison','post-run source snapshot and weight stats cannot exclude transient replacement']}
 with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())
ROOT_PLAN=B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
if __name__=='__main__':main()
