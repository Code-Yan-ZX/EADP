"""CPU acceptance of complete 5940-job full latency run; no labels or deletion."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PILOT=BASE/'e4_c_latency';sys.path.insert(0,str(PILOT))
from identity_gate import sha
from real_worker import PLAN
from latency_admission import terminal
from input_preflight import load_accepted_e1_references
from latency_worker import sources as pilot_sources
from schedule import build
from verify_core import annotate_validated
from full_admission import admit
from container_identity import validate as validate_container

def read(path):return json.loads(Path(path).read_text())
def lines(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]

def accept(out,monitor,annotation_path):
 out=Path(out).resolve();monitor=Path(monitor).resolve();started=monitor/'started.json';ended=monitor/'terminal.json';s=terminal(started,ended)
 launch=read(out/'launch.json');ex=read(out/'exit.json');frozen=read(out/'frozen.json');done=read(out/'worker_complete.json');provenance=read(out/'full_provenance.json');approval=admit(preflight_only=False,host=True)
 controller=launch.get('controller') or {};container=ex.get('container_identity') or {};host=container.get('host_init') or {};pilot_source={str(x):sha(x) for x in pilot_sources()}
 if (launch.get('approval_bundle')!=approval or launch.get('source_sha256')!=approval['source_sha256'] or provenance.get('full_source_sha256')!=approval['source_sha256'] or provenance.get('pilot_identity')!=approval['pilot_identity'] or
     provenance.get('stage')!='full' or provenance.get('budget_sha256')!=sha(HERE/'budget_plan.json') or provenance.get('GT_opened') is not False or provenance.get('scoring_performed') is not False or
     frozen.get('source_sha256')!=pilot_source or frozen.get('identity')!=approval['pilot_identity']['light_identity'] or frozen.get('stage')!='full' or frozen.get('GT_opened') is not False or frozen.get('scoring_performed') is not False or
     done.get('status')!='completed' or done.get('stage')!='full' or done.get('jobs')!=5940 or controller.get('pid')!=s.get('worker_pid') or ex.get('controller')!=controller):raise ValueError('full run identity/completion mismatch')
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or ex.get('client_exited_while_container_running') is not False or
     ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[] or not container.get('container_id') or str(HERE/'full_worker.py') not in container.get('host_argv',[]) or container.get('container_id','') not in container.get('host_cgroup','') or
     type(host.get('pid')) is not int or type(host.get('start_ticks')) is not int or not host.get('boot_id')):raise ValueError('full container exit incomplete')
 raw=subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True);item=json.loads(raw.stdout)[0];state=item['State'];plan=read(PLAN)
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command
 inspection=validate_container(out,launch,ex,item,plan,container_command)
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()==host['boot_id']:
  try:ticks=int(Path('/proc',str(host['pid']),'stat').read_text().rsplit(')',1)[1].split()[19])
  except FileNotFoundError:ticks=None
  if ticks==host['start_ticks']:raise ValueError('full container process remains')
 e1=BASE/'e1';proof=e1/'e1full001_completion.json';es=e1/'monitorverifye1full001/started.json';et=e1/'monitorverifye1full001/terminal.json'
 if any(frozen.get(k)!=sha(path) for k,path in (('e1_acceptance_sha256',proof),('e1_started_sha256',es),('e1_terminal_sha256',et))):raise ValueError('E1 proof identity changed')
 rows,_refs=load_accepted_e1_references(proof,es,et);anchor=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl'
 if frozen.get('anchor_sha256')!=sha(anchor):raise ValueError('anchor exclusion changed')
 excluded={json.loads(x)['group_id'] for x in anchor.read_text().splitlines() if x.strip()};expected=build(rows,excluded,'full')
 if len(excluded)!=16 or frozen.get('schedule')!=expected or len(expected['schedule'])!=5940:raise ValueError('full schedule changed')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False);decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 events=lines(out/'journal.jsonl');snapshots=lines(monitor/'health.jsonl')
 if not events or events[0].get('worker_pid')!=1:raise ValueError('unexpected worker PID namespace')
 namespace=events[0]['result']['boundaries']['time_namespace'];result=annotate_validated(expected,events,1,decode,snapshots,namespace,s,read(ended))
 with Path(annotation_path).open('x') as f:
  for item in result['events']:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 evidence=[out/x for x in ('launch.json','exit.json','full_provenance.json','frozen.json','worker_complete.json','journal.jsonl')]+[started,ended,monitor/'health.jsonl',Path(annotation_path)]
 return {'status':'passed','stage':'full','n_jobs':5940,'n_generated':result['n_generated'],'n_monitor_overlaps':result['n_monitor_overlaps'],'GT_opened':False,'scoring_performed':False,'errors':[],
         'evidence_sha256':{str(p.resolve()):sha(p) for p in evidence},'source_sha256':approval['source_sha256'],'verifier_sha256':sha(__file__),
         'inspection':inspection,'scope':'all full DEV warm-cache latency jobs retained; hourly health-overlap flags only; no quality scoring',
         'limitations':['health window overlap does not exclude nearby residual interference','pilot host had unrelated Java container restart activity, not quantified','single model process interleaves configurations; memory is separate']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();begin=time.time()
 try:r=accept(a.run_dir,a.monitor,a.annotations)
 except Exception as exc:r={'status':'failed','errors':[{'type':type(exc).__name__,'message':str(exc)}],'GT_opened':False,'scoring_performed':False}
 r['started_unix']=begin;r['finished_unix']=time.time()
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
