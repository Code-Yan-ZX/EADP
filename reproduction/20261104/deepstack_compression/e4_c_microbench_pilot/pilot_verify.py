"""CPU-only acceptance of one complete component pilot; no scoring."""
import argparse,json,math,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from pilot_admission import admit,validate_container
from latency_admission import terminal
from real_worker import sha
from pilot_worker import pilot_schedule,get_reference
from identity_contract import validate
from fingerprint import fingerprint
from input_preflight import load_accepted_e1_references,tensor_meta

def check_events(events,schedule,fingerprints):
 if not events or sum(e.get('status')=='capture_generated' for e in events)!=1:raise ValueError('diagnostic generation count')
 clean=[e for e in events if e.get('status')=='capture_cleanup'];ident=[e for e in events if e.get('status')=='capture_identity_passed'];archives=[e for e in events if e.get('status')=='archive_written'];refs=[e for e in events if e.get('status')=='reference_fingerprints'];warm=[e for e in events if e.get('status')=='warmup_validated'];measured=[e for e in events if e.get('status')=='measured']
 if len(clean)!=1 or clean[0].get('passed') is not True or clean[0].get('capture_completed') is not True or len(ident)!=1 or ident[0].get('identity',{}).get('status')!='passed' or len(archives)!=1 or len(refs)!=1 or refs[0].get('sha256')!=fingerprints or len(warm)!=9 or len(measured)!=60:raise ValueError('capture or measurement inventory')
 if any(e.get('status') not in {'capture_generated','capture_cleanup','capture_identity_passed','archive_written','reference_fingerprints','warmup_validated','measured'} for e in events):raise ValueError('unknown/failed journal event')
 pos=lambda x:events.index(x)
 if not (pos(clean[0])<pos(archives[0])<pos(ident[0])<pos(refs[0])<pos(measured[0])):raise ValueError('capture event order')
 for c in schedule['components']:
  w=[e for e in warm if e.get('component')==c]
  if [e.get('index') for e in w]!=[0,1,2] or any(e.get('full_output_equal') is not True or e.get('output_sha256')!=fingerprints[c] for e in w):raise ValueError('warmup mismatch')
  first=next(e for e in measured if e['job']['component']==c)
  if not all(pos(e)<pos(first) for e in w):raise ValueError('warmup ordering')
 for i,e in enumerate(measured):
  job=schedule['jobs'][i];c=job['component']
  if e.get('job')!=job or e.get('full_output_equal') is not True or e.get('output_sha256')!=fingerprints[c] or any(type(e.get(k)) not in (int,float) or not math.isfinite(e[k]) or e[k]<0 for k in ('gpu_event_ms','host_wall_ms')):raise ValueError('measured output/timing mismatch')
  if c=='ds_injection' and 'excludes decoder DS addition' not in e.get('scope',''):raise ValueError('DS staging scope mismatch')
 if Counter(e['job']['component'] for e in measured)!={c:20 for c in schedule['components']}:raise ValueError('job distribution')
 return archives[0]

def accept(out,monitor):
 import torch
 out=Path(out).resolve();monitor=Path(monitor).resolve();started=terminal(monitor/'started.json',monitor/'terminal.json');launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());done=json.loads((out/'worker_complete.json').read_text());approval=admit(preflight_only=False,host=True)
 if launch.get('approval_bundle')!=approval or done.get('source_sha256')!=approval['source_sha256'] or done.get('status')!='completed' or done.get('result',{}).get('measurements')!=60 or done.get('GT_opened') is not False or done.get('scoring_performed') is not False or (launch.get('controller') or {}).get('pid')!=started.get('worker_pid') or ex.get('controller')!=launch.get('controller'):raise ValueError('run source or completion mismatch')
 if ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('failure_type') is not None or ex.get('container_stopped_verified') is not True or ex.get('client_exited_while_container_running') is not False or ex.get('gpu_pids_after')!=[]:raise ValueError('container exit mismatch')
 obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];inspection=validate_container(launch,ex,obj,gpu=True)
 s=pilot_schedule()
 if done.get('schedule')!=s:raise ValueError('frozen schedule mismatch')
 e1=BASE/'e1';rows,_=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json');row=[r for r in rows if r['task']==s['images'][0][0] and r['group_id']==s['images'][0][1] and str(r['qid'])=='16539']
 if len(row)!=1:raise ValueError('reference row mismatch')
 row=row[0];ref=get_reference(row);data=torch.load(out/'reference.pt',map_location='cpu',weights_only=True);identity=validate(data['actual'],ref,tensor_meta);fps={k:fingerprint(torch,v) for k,v in data['references'].items()};events=[json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines() if x.strip()];archive=check_events(events,s,fps)
 if archive.get('path')!=str(out/'reference.pt') or archive.get('sha256')!=sha(out/'reference.pt'):raise ValueError('archive binding mismatch')
 files=[out/x for x in ('launch.json','exit.json','journal.jsonl','reference.pt','worker_complete.json')]+[monitor/'started.json',monitor/'terminal.json']
 return {'status':'passed','jobs':60,'warmups':9,'diagnostic_generations':1,'GT_opened':False,'scoring_performed':False,'identity':identity,'inspection':inspection,'files_sha256':{str(p):sha(p) for p in files},'source_sha256':approval['source_sha256'],'verifier_sha256':sha(__file__),'limitations':['each output was fully compared online; individual output tensors were not archived','ds_injection is selection/mask staging, excludes decoder addition','component event times are warm-cache diagnostics, not end-to-end TTFT']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:r=accept(a.run_dir,a.monitor)
 except Exception as e:r={'status':'failed','errors':[{'type':type(e).__name__,'message':str(e)}],'GT_opened':False,'scoring_performed':False}
 r['finished_unix']=time.time()
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if r['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
