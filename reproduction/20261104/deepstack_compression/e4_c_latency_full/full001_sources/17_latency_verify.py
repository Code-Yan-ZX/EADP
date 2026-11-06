"""CPU acceptance for completed latency smoke; no GT, score or sample removal."""
import argparse,json,os,subprocess,time
from pathlib import Path
from real_worker import HERE,BASE,PLAN,sha
from latency_admission import admit,terminal
from input_preflight import load_accepted_e1_references
from schedule import build
from verify_core import annotate_validated

def lines(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]

def accept(out,monitor,annotation_path):
 out=Path(out).resolve();monitor=Path(monitor).resolve();started=monitor/'started.json';ended=monitor/'terminal.json';s=terminal(started,ended)
 launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());frozen=json.loads((out/'frozen.json').read_text());done=json.loads((out/'worker_complete.json').read_text())
 approval=admit('smoke',preflight_only=False);controller=launch.get('controller') or {};container=ex.get('container_identity') or {};host=container.get('host_init') or {}
 if (launch.get('approval_bundle')!=approval or launch.get('source_sha256')!=approval['source_sha256'] or frozen.get('source_sha256')!=approval['source_sha256'] or
     frozen.get('identity')!=approval['light_identity'] or frozen.get('stage')!='smoke' or frozen.get('GT_opened') is not False or frozen.get('scoring_performed') is not False or
     done.get('status')!='completed' or done.get('stage')!='smoke' or done.get('jobs')!=72 or controller.get('pid')!=s.get('worker_pid') or ex.get('controller')!=controller):raise ValueError('timing run identity/completion mismatch')
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[] or
     not container.get('container_id') or str(HERE/'latency_worker.py') not in container.get('host_argv',[]) or container.get('container_id','') not in container.get('host_cgroup','') or
     type(host.get('pid')) is not int or type(host.get('start_ticks')) is not int or not host.get('boot_id')):raise ValueError('timing container exit proof incomplete')
 raw=subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True);item=json.loads(raw.stdout)[0];state=item['State'];plan=json.loads(PLAN.read_text())
 if (item['Id']!=container['container_id'] or item['Image']!=container['image'] or item['Image']!=plan['container_image'] or item['HostConfig']['PidMode'] not in ('',None) or
     state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0):raise ValueError('timing actual container state mismatch')
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()==host['boot_id']:
  try:ticks=int(Path('/proc',str(host['pid']),'stat').read_text().rsplit(')',1)[1].split()[19])
  except FileNotFoundError:ticks=None
  if ticks==host['start_ticks']:raise ValueError('timing container process still exists')
 e1=BASE/'e1';proof=e1/'e1full001_completion.json';es=e1/'monitorverifye1full001/started.json';et=e1/'monitorverifye1full001/terminal.json'
 if any(frozen.get(k)!=sha(path) for k,path in (('e1_acceptance_sha256',proof),('e1_started_sha256',es),('e1_terminal_sha256',et))):raise ValueError('E1 proof identity changed')
 rows,_refs=load_accepted_e1_references(proof,es,et);anchor=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl'
 if frozen.get('anchor_sha256')!=sha(anchor):raise ValueError('anchor exclusions changed')
 excluded={json.loads(x)['group_id'] for x in anchor.read_text().splitlines() if x.strip()};expected=build(rows,excluded,'smoke')
 if len(excluded)!=16 or frozen.get('schedule')!=expected:raise ValueError('actual schedule differs from frozen DEV protocol')
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 events=lines(out/'journal.jsonl');snapshots=lines(monitor/'health.jsonl')
 if not events or events[0].get('worker_pid')!=1:raise ValueError('unexpected Docker worker PID')
 namespace=events[0]['result']['boundaries']['time_namespace'];result=annotate_validated(expected,events,1,decode,snapshots,namespace,s,json.loads(ended.read_text()))
 with Path(annotation_path).open('x') as f:
  for item in result['events']:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 evidence=[out/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','journal.jsonl')]+[started,ended,monitor/'health.jsonl',Path(annotation_path)]
 return {'status':'passed','stage':'smoke','n_jobs':len(expected['schedule']),'n_generated':result['n_generated'],'n_monitor_overlaps':result['n_monitor_overlaps'],
         'GT_opened':False,'scoring_performed':False,'errors':[],'evidence_sha256':{str(x.resolve()):sha(x) for x in evidence},
         'source_sha256':approval['source_sha256'],'verifier_sha256':sha(__file__),
         'scope':'all 72 warm-cache DEV latency smoke jobs retained; health-overlap flags only; no accuracy or performance extrapolation',
         'limitations':['health window overlap does not exclude nearby residual interference','single model process interleaves configurations; memory is separate']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();begin=time.time()
 try:r=accept(args.run_dir,args.monitor,args.annotations)
 except Exception as exc:r={'status':'failed','errors':[{'type':type(exc).__name__,'message':str(exc)}],'GT_opened':False,'scoring_performed':False}
 r['started_unix']=begin;r['finished_unix']=time.time()
 with args.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
