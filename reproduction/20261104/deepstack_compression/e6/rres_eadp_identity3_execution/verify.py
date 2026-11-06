"""Post-run complete six-case diagnostic acceptance; no latency or GT."""
import argparse,hashlib,json,os,struct,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ended(s,t):
 a=read(s);b=read(t);return type(a.get('worker_pid')) is int and b.get('worker_pid')==a['worker_pid'] and b.get('reason')=='completed' and b.get('worker_returncode')==0 and b.get('remaining_process_group')==[]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=a.out.resolve()
 if out!=(HERE/'verification001.json').resolve() or out.exists():raise ValueError('unplanned output')
 plan=read(HERE/'gpu_plan001.json');run=HERE/'run001';mon=HERE/'monitor_run001';snapshot=HERE/'run001_sources/manifest.json';sm=HERE/'monitor_snapshot001'
 formal=read(HERE/'verification_plan001.json')
 if formal.get('gpu_plan_sha256')!=sha(HERE/'gpu_plan001.json') or formal.get('verifier_sha256')!=sha(__file__) or formal.get('snapshot_sha256')!=sha(HERE/'snapshot.py'):raise ValueError('verification plan changed')
 if not ended(mon/'started.json',mon/'terminal.json') or not ended(sm/'started.json',sm/'terminal.json') or read(run/'controller_complete.json')['status']!='passed':raise ValueError('run/snapshot failed')
 manifest=read(snapshot)
 if manifest.get('status')!='passed' or manifest.get('post_run_snapshot') is not True or set(manifest['source_files'])!=set(plan['source_sha256']):raise ValueError('source snapshot inventory')
 for path,digest in plan['source_sha256'].items():
  item=manifest['source_files'][path]
  if item['sha256']!=digest or sha(item['copy'])!=digest or sha(path)!=digest:raise ValueError('source snapshot mismatch')
 sys.path[:0]=[str(IDENTITY),str(TIMING),str(ROOT/'e4')]
 import reference_inputs,formal_contract,decode_observer,docker_contract
 refs=reference_inputs.load();anchors=(('DocVQA_VAL','64588'),('TextVQA_VAL','36511'),('AI2D_TEST','207406'));files={};cases=[];artifacts={str((run/'controller_complete.json').resolve()):sha(run/'controller_complete.json')}
 baseline_cid=read(IDENTITY/'preflight001/container_identity.json')['container_id'];baseline=json.loads(subprocess.run(['docker','inspect',baseline_cid],capture_output=True,text=True,check=True,timeout=30).stdout)[0]
 if baseline['Id']!=baseline_cid or baseline['State']['Running'] or baseline['State']['Pid']!=0 or baseline['State']['ExitCode']!=0 or baseline['HostConfig'].get('Devices') not in (None,[]):raise ValueError('baseline noGPU container')
 old_command=read(ROOT/'e6/rres_k256_dev384/run001/launch.json')['command'];supervisor_pid=read(mon/'started.json')['worker_pid']
 for arm in ('rres','eadp'):
  armdir=run/arm;exit=read(armdir/'exit.json');identity=read(armdir/'container_identity.json');launch=read(armdir/'launch.json');cid=identity['container_id']
  if exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('container_stopped_verified') is not True or exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('gpu_pids_after')!=[] or launch.get('arm')!=arm or launch.get('gpu_pids_before')!=[]:raise ValueError('container completion '+arm)
  actual=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=30);details=json.loads(actual.stdout)[0]
  docker_contract.check(actual=details,baseline=baseline,old_command=old_command,launch=launch,identity=identity,exit=exit,arm=arm,worker=HERE/'gpu_worker.py',out=armdir/'observations',supervisor_pid=supervisor_pid)
  ob=armdir/'observations';complete=read(ob/'worker_complete.json');endpoints=read(ob/'identity_endpoints.json')
  if complete.get('status')!='passed' or complete.get('anchors')!=3 or complete.get('timing_claims') is not False or endpoints.get('status')!='passed' or endpoints.get('weights_before')!=endpoints.get('weights_after'):raise ValueError('worker identity endpoints')
  for path in (armdir/'launch.json',armdir/'exit.json',armdir/'container_identity.json',armdir/'docker.log',ob/'progress.jsonl',ob/'worker_complete.json',ob/'identity_endpoints.json'):artifacts[str(path.resolve())]=sha(path)
  for task,qid in anchors:
   row=refs['rows'][(task,qid)];ref=dict(refs['references'][(arm,task,qid)]);ref['image_start']=row['image_start'];name=task+'_'+qid;rp=ob/(name+'.json');rawp=ob/(name+'_raw.json');record=read(rp);raw=read(rawp)
   if raw.get('arm')!=arm or raw.get('task')!=task or str(raw.get('qid'))!=qid:raise ValueError('raw arm/task/qid identity')
   formal_contract.check(record,raw,ref,decode_observer)
   if arm=='rres':
    prefix=raw['full_sequence'][0][:record['input_length']]
    input_meta={'dtype':'torch.int64','shape':[1,len(prefix)],'sha256':hashlib.sha256(struct.pack('<'+'q'*len(prefix),*prefix)).hexdigest()}
    if input_meta!=ref['processor']['input_ids']:raise ValueError('raw R-res prompt prefix')
   for path in (rp,rawp):files[str(path.resolve())]=sha(path)
   cases.append({'arm':arm,'task':task,'qid':qid,'status':'passed'})
 if len(cases)!=6 or len(files)!=12:raise ValueError('diagnostic denominator')
 result={'status':'passed','scope':'three anchors x two implementations; identity only','generations':6,'images':3,'cases':cases,'files_sha256':files,'run_artifact_sha256':artifacts,'source_manifest_sha256':sha(snapshot),'source_plan_sha256':sha(HERE/'gpu_plan001.json'),'run_started_sha256':sha(mon/'started.json'),'run_terminal_sha256':sha(mon/'terminal.json'),'verifier_sha256':sha(__file__),'GT_opened':False,'existing_predictions_opened':True,'scoring_performed':False,'timing_claims':False,'errors':[],'limits':['Diagnostics added synchronization/hooks; no latency claim','EADP prior prefill logits not available for comparison','High-dimensional tensors are not archived','Source snapshot post-run cannot exclude transient changes','Endpoint weight stats cannot exclude transient replacement']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
