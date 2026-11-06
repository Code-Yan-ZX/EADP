"""Independent full 192-image timing integrity and interference-window review; no timing aggregates."""
import argparse,hashlib,json,os,subprocess
from collections import Counter
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_c_multibudget_worker192_v6';A=B/'e4_c_multibudget_worker192_v6_accept';REF=B/'e4_c_multibudget_references192';RA=B/'e4_c_multibudget_references192_accept';ENV=B/'e4_c_multibudget_environment192_v2';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()],str(mon)
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def windows_health(path,ns):
 rows=lines(path);assert rows;out=[]
 for r in rows:
  w=r.get('health_snapshot_window_monotonic_ns');assert r.get('time_namespace')==ns and isinstance(w,list) and len(w)==2 and all(type(x) is int and x>=0 for x in w) and w[0]<=w[1];out.append(w)
 return out
def overlaps(ws,a,b):return [i for i,(x,y) in enumerate(ws) if x<=b and a<=y]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();formal=A/'verification001.json';terminal(A/'monitor_verify001',A/'verify.py',formal);terminal(A/'monitor_snapshot001',A/'snapshot.py');s,t=terminal(D/'monitor_run001',D/'launch.py');terminal(ENV/'monitor_watch001',ENV/'watch.py')
 f=read(formal);assert f['status']=='passed' and (f['jobs'],f['warmups'],f['measured'])==(7920,240,7680) and f['GT_opened'] is False and f['scoring_performed'] is False and f['samples_excluded']==0 and f['errors']==[] and f['verifier_sha256']==sha(A/'verify.py') and f['verification_plan_sha256']==sha(A/'verification_plan001.json')
 run=D/'run_001';health=D/'monitor_run001/health.jsonl';journal=run/'journal.jsonl';snapshot=D/'run_001_sources/manifest.json'
 for k,path in {'plan_sha256':D/'gpu_plan001.json','run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','launch_sha256':run/'launch.json','exit_sha256':run/'exit.json','worker_complete_sha256':run/'worker_complete.json','journal_sha256':journal,'health_sha256':health,'snapshot_manifest_sha256':snapshot,'snapshot_started_sha256':A/'monitor_snapshot001/started.json','snapshot_terminal_sha256':A/'monitor_snapshot001/terminal.json'}.items():assert f[k]==sha(path),k
 plan=read(D/'gpu_plan001.json');sm=read(snapshot);assert sm['status']=='passed' and sm['runtime_plan_sha256']==sha(D/'gpu_plan001.json') and set(sm['sources'])==set(plan['source_sha256'])
 for path,digest in plan['source_sha256'].items():
  item=sm['sources'][path];copy=Path(item['snapshot']).resolve();assert item['source']==path and item['sha256']==digest and copy.is_relative_to(snapshot.parent.resolve()) and sha(copy)==digest
 launch=read(run/'launch.json');ex=read(run/'exit.json');complete=read(run/'worker_complete.json');cid=ex['container_identity']['container_id']
 assert s['worker_pid']==launch['controller']['pid'] and ex['controller']==launch['controller'] and ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['client_exited_while_container_running'] is False and ex['container_stopped_verified'] is True and ex['gpu_pids_after']==[] and ex['last_inspect']['id']==cid and complete['status']=='passed' and complete['journal_sha256']==sha(journal)
 preid=read(D/'run_preflight001/exit.json')['container_identity']['container_id'];actual,pre=[json.loads(subprocess.check_output(['docker','inspect',x]))[0] for x in (cid,preid)]
 assert actual['Id']==cid and not actual['State']['Running'] and actual['State']['Pid']==actual['State']['ExitCode']==0 and actual['Image']==pre['Image']==ex['last_inspect']['image'] and actual['HostConfig']['NetworkMode']=='none' and actual['Config']['Cmd']==[str(D/'worker.py'),'--out',str(run)]
 for k in ('Entrypoint','User','WorkingDir'):assert actual['Config'][k]==pre['Config'][k]
 ae={x.split('=',1)[0]:x for x in actual['Config']['Env']};pe={x.split('=',1)[0]:x for x in pre['Config']['Env']};assert len(ae)==len(actual['Config']['Env']) and len(pe)==len(pre['Config']['Env']) and ae.pop('CUDA_VISIBLE_DEVICES')=='CUDA_VISIBLE_DEVICES=0' and pe.pop('CUDA_VISIBLE_DEVICES')=='CUDA_VISIBLE_DEVICES=' and ae==pe
 assert sorted(json.dumps(x,sort_keys=True) for x in actual['Mounts'])==sorted(json.dumps(x,sort_keys=True) for x in pre['Mounts']) and {x['PathOnHost'] for x in actual['HostConfig']['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'} and not actual['HostConfig']['DeviceRequests']
 # Confirm accepted reference evidence before opening actual reference records.
 refproof=I/'e4c_multibudget_references192_result_review.json';rf=RA/'verification001.json';terminal(RA/'monitor_verify001',RA/'verify.py',rf);terminal(I/'monitor_e4c_multibudget_references192_result',I/'review_e4c_multibudget_references192_result.py',refproof);q=read(refproof);r=read(rf);records=REF/'run_001/diagnostic/records.jsonl';assert r['status']==q['status']=='passed' and q['independent_review'] is True and (q['inputs'],q['generations'])==(222,888) and q['formal_sha256']==sha(rf) and q['records_sha256']==r['records_sha256']==sha(records) and q['reviewer_sha256']==sha(I/'review_e4c_multibudget_references192_result.py')
 proposal=read(B/'e4_c_multibudget_expansion192/proposal002.json');assert sha(B/'e4_c_multibudget_expansion192/proposal002.json')==plan['proposal_sha256'];inputs=proposal['measurement_identities']+proposal['warmup_identities'];jobs=proposal['jobs'];by={(x['task'],str(x['qid'])):x for x in inputs};assert len(by)==222 and len(jobs)==7920
 fixed={}
 for x in lines(records):
  key=(x['task'],str(x['qid']),x['budget']);assert key not in fixed and key[:2] in by and x['protocol']=='fixed32' and all(x[k]==by[key[:2]][k] for k in ('group_id','image_sha256','message_sha256'));fixed[key]=x
 assert len(fixed)==888
 panelpath=B/'e4_panel_v2/panel_003/records.jsonl';e4formal=read(B/'e4_panel_v2/panel_003_verification.json');assert e4formal['status']=='passed' and e4formal['files_sha256'][str(panelpath)]==sha(panelpath)
 panel={}
 for x in lines(panelpath):
  key=(x['task'],str(x['qid']),x['budget'])
  if key[:2] in by and x.get('budget') in (128,256,512,1024) and x.get('ds_on') is True and x.get('forced') is False:
   assert key not in panel and all(x[k]==by[key[:2]][k] for k in ('group_id','image_sha256','message_sha256')) and x['processor']==by[key[:2]]['processor_reference'] and x['image_start']==by[key[:2]]['image_start_reference'];panel[key]=x
 assert len(panel)==888
 # Rebuild all observation windows independently; absence is never treated as no interference.
 ns=launch['time_namespace'];model=windows_health(health,ns);watchhealth=windows_health(ENV/'monitor_watch001/health.jsonl',ns);assert model==f['health_windows']
 envfiles={str(x) for x in (ENV/'samples001.jsonl',ENV/'manual_checks001.jsonl',ENV/'monitor_watch001/health.jsonl',ENV/'monitor_watch001/started.json',ENV/'monitor_watch001/terminal.json',ENV/'stop001')};assert set(f['environment_files_sha256'])==envfiles and all(sha(k)==v for k,v in f['environment_files_sha256'].items())
 observe=lines(ENV/'samples001.jsonl');assert len(observe)>=3 and len(observe)%2==1 and observe[-1]['event']=='watch_end' and observe[-1]['reason']=='stop_marker' and observe[-1]['time_namespace']==ns and observe[-1]['samples']==(len(observe)-1)//2
 query=[];first=[];prev=-1;start=None
 for i in range((len(observe)-1)//2):
  x,y=observe[2*i:2*i+2];assert x['event']=='snapshot' and y['event']=='logging_complete' and x['index']==y['index']==i and x['time_namespace']==y['time_namespace']==ns and x['returncode']==0 and isinstance(x['stdout'],str)
  if start is None:start=x['watch_start_monotonic_ns']
  a,b,c=x['query_start_ns'],x['query_end_ns'],y['first_fsync_end_ns'];assert all(type(v) is int for v in (start,a,b,c)) and x['watch_start_monotonic_ns']==start and start<=a and prev<a<=b<=c and y['query_start_ns']==a and x['query_start_utc'] and x['query_end_utc'] and y['first_fsync_end_utc'];query.append([a,b]);first.append([a,c]);prev=c
 assert type(observe[-1]['end_ns']) is int and observe[-1]['end_ns']>=prev and observe[-1]['end_utc']
 manual=[]
 for x in lines(ENV/'manual_checks001.jsonl'):
  a,b=x['start_monotonic_ns'],x['end_monotonic_ns'];assert x['event']=='manual_health_check' and x['time_namespace']==ns and type(a) is int and type(b) is int and 0<=a<=b and x['start_utc'] and x['end_utc'] and isinstance(x['details'],dict);manual.append([a,b])
 window_sets={'model_health':model,'query':query,'query_first_fsync':first,'watcher_health':watchhealth,'manual_health':manual}
 ev=lines(journal);assert len(ev)==23760 and len(f['annotations'])==len(f['environment_job_details'])==7920
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 decoded={};seen=Counter();phase=Counter();groups=Counter();bounds=[]
 for i,job in enumerate(jobs):
  assert job['order']==i and job['phase'] in ('warmup','measure') and job['budget'] in (128,256,512,1024) and job['protocol'] in ('natural','fixed32')
  key=(job['task'],str(job['qid']),job['budget']);assert key in panel and key in fixed and job['group_id']==by[key[:2]]['group_id']
  expected_reps=range(5) if job['phase']=='measure' else (0,);assert type(job['replicate']) is int and job['replicate'] in expected_reps;seen[(key,job['phase'],job['protocol'],job['replicate'])]+=1;phase[job['phase']]+=1
  if job['phase']=='measure':groups[(job['task'],job['protocol'],job['budget'])]+=1
  trio=ev[3*i:3*i+3];assert [e['event'] for e in trio]==['job_started','raw_generation','job_completed'] and all(e['job']==job for e in trio)
  raw,rec=trio[1]['raw'],trio[2]['record']
  for k in ('task','qid','group_id','budget','protocol'):
   assert str(raw[k])==str(rec[k])==str(job[k])
  for k in ('output_ids','generation_scalars','min_new_tokens','eos_ids'):assert raw[k]==rec[k]
  assert rec['selector_calls']==(0 if job['budget']==1024 else 1)
  expected=panel[key] if job['protocol']=='natural' else fixed[key]
  fields=('output_ids','answer','generation_scalars','eos_ids','eos','truncated') if job['protocol']=='natural' else ('output_ids','answer','generation_scalars','min_new_tokens','eos_ids')
  assert all(rec[k]==expected[k] for k in fields)
  if job['protocol']=='natural':assert rec['min_new_tokens'] is None
  else:assert len(rec['output_ids'])==32 and not any(x in rec['eos_ids'] for x in rec['output_ids'][:-1])
  ids=tuple(rec['output_ids']);assert ids and all(type(x) is int for x in ids)
  if ids not in decoded:decoded[ids]=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
  assert decoded[ids]==rec['answer']
  t=rec['boundaries'];a,b,c=(t[k] for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns'))
  assert all(type(x) is int for x in (a,b,c)) and 0<=a<b<=c and raw['generation_complete_ns']==c and t['time_namespace']==ns and t['first_criteria_length']==1 and t['criteria_calls']==len(ids) and t['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)' and t['explicit_sync_boundaries']==['before_t0','first_token','generate_complete'];bounds.append((a,b,c))
  ann={'order':i,'phase':job['phase'],'task':job['task'],'budget':job['budget'],'protocol':job['protocol'],'ttft_health_overlap_indices':overlaps(model,a,b),'full_job_health_overlap_indices':overlaps(model,a,c)};assert f['annotations'][i]==ann
  row={k:job[k] for k in ('order','phase','task','qid','group_id','budget','protocol','replicate')};row.update(t0_before_image_read_ns=a,first_token_selected_ns=b,generation_complete_ns=c,token_count=len(ids),eos=rec['eos'],truncated=rec['truncated'],overlap_indices={name:{'ttft':overlaps(ws,a,b),'generation':overlaps(ws,a,c)} for name,ws in window_sets.items()});assert f['environment_job_details'][i]==row
 assert len(seen)==7920 and all(x==1 for x in seen.values()) and phase=={'warmup':240,'measure':7680} and len(groups)==24 and set(groups.values())=={320} and query[0][0]<=min(x[0] for x in bounds) and observe[-1]['end_ns']>=max(x[2] for x in bounds)
 # Hour coverage is checked independently, without changing or suppressing samples.
 lo=min(x[0] for x in bounds);hi=max(x[2] for x in bounds);hours=[];n=1;period=3600*10**9;slack=300*10**9;combined=model+watchhealth
 while lo+n*period<=hi:
  at=lo+n*period;m=[i for i,(a,b) in enumerate(manual) if a<=at+slack and b>=at-slack];h=[i for i,(a,b) in enumerate(combined) if a<=at+slack and b>=at-slack];hours.append({'hour':n,'manual_indices':m,'supervisor_health_indices':h,'manual_status':'observed' if m else 'unknown','supervisor_health_status':'observed' if h else 'unknown'});n+=1
 assert hours==f['hourly_health_coverage'] and 'unknown' in f['environment_limits']
 out={'status':'passed','independent_review':True,'jobs':7920,'warmups':240,'measured':7680,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'samples_excluded':0,'formal_sha256':sha(formal),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'journal_sha256':sha(journal),'health_sha256':sha(health),'snapshot_manifest_sha256':sha(snapshot),'reference_formal_sha256':sha(rf),'reference_review_sha256':sha(refproof),'environment_files_sha256':f['environment_files_sha256'],'environment_jobs_verified':7920,'source_files_verified':len(plan['source_sha256']),'scope':'independent complete identity, output, boundary and recorded-observation review; no timing aggregation','limitations':['Second fsync, manual writes, marker writes, unrecorded background and residual effects have unknown overlap; all observations retained.','Snapshot and endpoint source/weight checks cannot exclude transient replacement.','Existing accepted DEV prediction records read only to validate endpoints; no GT or score.'],'errors':[]}
 with open(args.out,'x') as fp:json.dump(out,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
