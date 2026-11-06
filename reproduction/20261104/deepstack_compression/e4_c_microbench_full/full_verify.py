"""CPU verification of complete 32-image component run; no GT or scoring."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PILOT=BASE/'e4_c_microbench_pilot';sys.path[:0]=[str(PILOT),str(BASE/'e4_c_latency')]
from full_admission import admit,validate_container
from full_contract import split_schedule
from latency_admission import terminal
from real_worker import sha
from pilot_verify import check_events
from pilot_worker import get_reference
from identity_contract import validate
from fingerprint import fingerprint
from input_preflight import load_accepted_e1_references,tensor_meta

def group_events(events):
 chunks=[];current=None
 for event in events:
  idx=event.get('image_index');status=event.get('status')
  if status=='image_started':
   if current is not None or idx!=len(chunks):raise ValueError('image start/order mismatch')
   current=[event]
  elif status=='image_completed':
   if current is None or idx!=len(chunks):raise ValueError('image completion/order mismatch')
   current.append(event);chunks.append(current);current=None
  else:
   if current is None or idx!=len(chunks):raise ValueError('orphan/mislabelled event')
   current.append(event)
 if current is not None or len(chunks)!=32:raise ValueError('incomplete image events')
 return chunks

def check_event_identity(chunk,row):
 for kind in ('capture_generated','capture_identity_passed'):
  matched=[e for e in chunk if e.get('status')==kind]
  if len(matched)!=1 or any(str(matched[0].get(k))!=str(row.get(k)) for k in ('task','qid','group_id')):raise ValueError('full '+kind+' identity mismatch')

def accept(out,monitor):
 import torch
 out=Path(out).resolve();monitor=Path(monitor).resolve();started=terminal(monitor/'started.json',monitor/'terminal.json');launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());done=json.loads((out/'worker_complete.json').read_text());approval=admit(preflight_only=False,host=True)
 if launch.get('approval_bundle')!=approval or done.get('source_sha256')!=approval['source_sha256'] or done.get('status')!='completed' or done.get('images')!=32 or done.get('jobs')!=1920 or done.get('diagnostic_generations')!=32 or done.get('GT_opened') is not False or done.get('scoring_performed') is not False or (launch.get('controller') or {}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller'):raise ValueError('full run/source mismatch')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[]:raise ValueError('full container exit mismatch')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];inspection=validate_container(launch,ex,obj,gpu=True);schedule_path=BASE/'e4_c_microbench/schedule001.json';payload=json.loads(schedule_path.read_text());schedules=split_schedule(payload)
 if done.get('schedule_sha256')!=sha(schedule_path):raise ValueError('full schedule hash mismatch')
 e1=BASE/'e1';rows,_=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');index={(r['task'],str(r['qid'])):r for r in rows};events=[json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines() if x.strip()];chunks=group_events(events);files=[out/x for x in ('launch.json','exit.json','journal.jsonl','worker_complete.json')]+[monitor/'started.json',monitor/'terminal.json'];array_sha={}
 for i,(chunk,schedule,identity) in enumerate(zip(chunks,schedules,payload['input_identities'])):
  row=index[(identity['task'],str(identity['qid']))]
  if row.get('split')!='DEV' or any(row.get(k)!=identity.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('full input mismatch')
  for e in (chunk[0],chunk[-1]):
   if any(str(e.get(k))!=str(row.get(k)) for k in ('task','qid','group_id')):raise ValueError('full image identity mismatch')
  if chunk[-1].get('result',{}).get('measurements')!=60:raise ValueError('full image incomplete')
  check_event_identity(chunk,row)
  ref=get_reference(row);path=out/('image_%03d'%i)/'reference.pt';data=torch.load(path,map_location='cpu',weights_only=True);validate(data['actual'],ref,tensor_meta);fps={k:fingerprint(torch,v) for k,v in data['references'].items()};archive=check_events(chunk[1:-1],schedule,fps)
  if archive.get('path')!=str(path) or archive.get('sha256')!=sha(path):raise ValueError('full archive mismatch')
  array_sha[str(path)]=sha(path)
 return {'status':'passed','images':32,'jobs':1920,'warmups':288,'diagnostic_generations':32,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'inspection':inspection,'files_sha256':{str(p):sha(p) for p in files},'array_sha256':array_sha,'source_sha256':approval['source_sha256'],'verifier_sha256':sha(__file__),'limitations':['expansion decided after single-image result','each output compared online; per-measurement output tensors not archived','ds_injection is selection/mask staging, excludes decoder addition','warm-cache component diagnostics, not end-to-end TTFT']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:r=accept(a.run_dir,a.monitor)
 except Exception as e:r={'status':'failed','errors':[{'type':type(e).__name__,'message':str(e)}],'GT_opened':False,'scoring_performed':False}
 r['finished_unix']=time.time()
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
