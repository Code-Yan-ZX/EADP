"""Independent nine-container timing, raw-output and provenance reconstruction; no GT."""
import argparse,hashlib,json,struct,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_natural_timing_pilot_v4';R=D/'run_pilot001';I=B/'e1_independent_review';A=B/'e6/pixel_natural_timing_accept_v4';E=B/'e6/pixel_natural_timing_v4_environment'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'pixel_natural_timing_v4_result_review.json'
f=A/'verification002.json';formal=read(f);term(A/'monitor_verify002',A/'verify002.py',f);start=term(D/'monitor_pilot001',D/'gpu_launch.py');term(I/'monitor_pixel_timing_v4_prelaunch',I/'review_pixel_timing_v4_prelaunch.py',D/'independent_prelaunch_review.json');term(E/'monitor_summary002',E/'summarize002.py')
assert formal['status']=='passed' and formal['jobs']==9 and formal['generations']==18 and formal['measured']==9 and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['errors']==[]
vplan=read(A/'verification_plan002.json');assert sha(A/'verification_plan002.json')=='e522eb103cc1cb000cee6a6e24529395fd5101ec07c8c5f92cb47d6556f516da' and vplan['pilot_plan_sha256']==sha(D/'plan001.json') and all(sha(k)==v for k,v in vplan['sha256'].items()) and all(sha(k)==v for k,v in formal['evidence_sha256'].items())
plan=read(D/'plan001.json');assert len(plan['jobs'])==9 and len(plan['anchors'])==3 and plan['GT_opened'] is False and all(sha(k)==v for k,v in plan['sha256'].items())
anchor={(x['task'],str(x['qid'])):x for x in plan['anchors']};assert len(anchor)==3 and len({x['group_id'] for x in plan['anchors']})==3
ref={}
for arm,sub in [('pixel_t0','pixelprune_dev384'),('pixel_t045','pixelprune_t045_dev384')]:
 for key,x in anchor.items():
  d=B/'e6'/sub/'run_smoke001/observations';stem=key[0]+'_'+key[1];q=read(d/(stem+'.json'));raw=read(d/(stem+'_generated.json'));assert all(q[k]==x[k] for k in ('task','qid','group_id','message_sha256','image_sha256'));assert raw['full_sequence_ids'][raw['prompt_length']:]==raw['generated_token_ids'] and q['status']=='completed' and all(q['diagnostic_checks'].values());ref[(arm,*key)]={'tokens':raw['generated_token_ids'],'k':q['selected_count'],'eos':q['eos_token_ids'],'processor':q['processor_identity']['processor']}
e1=read(B/'e1/e1full001_completion.json');journal=Path(e1['files']['journal']['path']);assert sha(journal)==e1['files']['journal']['sha256']
with journal.open() as h:
 for line in h:
  x=json.loads(line);key=(x['task'],str(x['qid']))
  if key in anchor and x.get('arm')=='P3-D1' and x.get('forced') is False:
   k=('eadp_k256',*key);assert k not in ref and x['group_id']==anchor[key]['group_id'] and x['message_sha256']==anchor[key]['message_sha256'] and len(x['ordered_ids'])==256;ref[k]={'tokens':x['output_ids'],'k':256,'eos':x['eos_ids'],'processor':x['processor']}
assert len(ref)==9
for key in anchor:
 v=[ref[(arm,*key)] for arm in ('pixel_t0','pixel_t045','eadp_k256')];assert all(x['processor']==v[0]['processor'] and x['eos']==v[0]['eos'] for x in v)
controller=lines(R/'controller.jsonl');expected=['controller_started']+sum((['job_started','job_exited'] for _ in plan['jobs']),[])+['controller_complete'];assert [x['event'] for x in controller]==expected and controller[-1]['jobs']==9
recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True};identities=set();results=[];annotations=[]
for i,job in enumerate(plan['jobs']):
 assert job['order']==i and controller[1+2*i]['job']==controller[2+2*i]['job']==job
 d=R/('job_%03d'%i);l=read(d/'launch.json');x=read(d/'exit.json');release=read(d/'release.json');r=ref[(job['arm'],job['task'],str(job['qid']))];assert l['job']==job and l['plan_sha256']==sha(D/'plan001.json') and l['controller']==x['controller'] and l['controller']['pid']==start['worker_pid'] and x['returncode']==x['container_exit_code']==0 and x['container_stopped_verified'] is True and x['gpu_pids_after']==[] and x['failure_type'] is None and x['timed_out'] is False and x['interrupted'] is None and x['client_exited_while_container_running'] is False
 ci=x['container_identity'];host=ci['host_init'];identity=(host['pid'],host['start_ticks'],host['boot_id']);assert identity not in identities and host['pid']>0 and host['start_ticks']>0 and ci['container_id'] in ci['host_cgroup'];identities.add(identity)
 assert release['plan_sha256']==l['plan_sha256'] and release['launch_sha256']==sha(d/'launch.json') and release['job']==job and release['container_id']==ci['container_id'] and release['host_init']==host and release['controller']==l['controller']
 obj=json.loads(subprocess.run(['docker','inspect',ci['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];cfg=obj['Config'];hc=obj['HostConfig'];state=obj['State'];cmd=l['command'];image=obj['Image'];assert obj['Id']==ci['container_id'] and obj['Name']=='/'+l['container_name'] and image==cfg['Image']==ci['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(image)+1:] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and hc['NetworkMode']==cmd[cmd.index('--network')+1]
 assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in hc['Devices']}=={(cmd[j+1],cmd[j+1],'rwm') for j,v in enumerate(cmd[:-1]) if v=='--device'} and not hc.get('DeviceRequests')
 for j,v in enumerate(cmd[:-1]):
  if v=='--env':
   k=cmd[j+1].split('=',1)[0];assert [z for z in cfg['Env'] if z.startswith(k+'=')]==[cmd[j+1]]
 mounts=[]
 for j,v in enumerate(cmd[:-1]):
  if v=='--mount':
   parts=cmd[j+1].split(',');attrs=dict(z.split('=',1) for z in parts if '=' in z);mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
 actual=[(str(Path(z['Source']).resolve()),z['Destination'],z['RW']) for z in obj['Mounts'] if z['Type']=='bind'];assert len(actual)==len(obj['Mounts'])==len(mounts) and sorted(actual)==sorted(mounts)
 seq=lines(d/'observations/journal.jsonl');assert [z['event'] for z in seq]==['model_loading','model_ready','started','raw_generation','completed','started','raw_generation','completed','job_complete'] and all(z['job']==job for z in seq)
 for j,phase in ((2,'warmup'),(5,'measured')):
  assert all(z['phase']==phase for z in seq[j:j+3]);raw=seq[j+1]['ids'];result=seq[j+2]['result'];assert len(raw)==1;full=raw[0];offset=r['processor']['input_ids']['shape'][1] if job['arm'].startswith('pixel_') else 0;tokens=result['tokens'];assert full[offset:]==tokens==r['tokens'] and result['eos_ids']==r['eos'] and result['recipe']==recipe and result['processor_verified'] is True
  if offset:assert hashlib.sha256(struct.pack('<'+'q'*offset,*full[:offset])).hexdigest()==r['processor']['input_ids']['sha256']
  assert 1<=len(tokens)<=2048 and (tokens[-1] in r['eos'] or len(tokens)==2048) and result['eos']==(tokens[-1] in r['eos']) and result['truncated']==(tokens[-1] not in r['eos'] and len(tokens)==2048)
  t=result['boundaries'];t0=t['t0_before_image_read_ns'];t1=t['first_token_selected_ns'];t2=t['generation_complete_ns'];assert all(type(z) is int for z in (t0,t1,t2)) and t0<t1<=t2 and t['explicit_sync_boundaries']==['before_t0','first_token','generation_end'] and t['first_criteria_length']==(offset+1 if offset else 1) and t['criteria_calls']>=1 and t['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)' and t['time_namespace']
  assert (result['count_observer_enabled'] is False and result['dynamic_k'] is None) if phase=='warmup' else (result['count_observer_enabled'] is True and result['dynamic_k']==r['k'] and result['selection_calls']==1)
  mem=seq[j+2]['memory'];assert 'not NVML' in mem['scope'];
  for f0 in ('allocated','reserved'):assert mem['max_'+f0+'_bytes']>=max(mem['baseline'][f0+'_bytes'],mem['end'][f0+'_bytes'])>=0
  ann=formal['annotations'][2*i+(j==5)];assert (ann['task'],ann['qid'],ann['arm'],ann['phase'])==(job['task'],job['qid'],job['arm'],phase) and (ann['t0_ns'],ann['first_token_ns'],ann['generation_end_ns'])==(t0,t1,t2) and ann['ttft_health_window_indices'] is None and ann['generation_health_window_indices'] is None and ann['health_overlap_status'].startswith('unknown:')
  if phase=='measured':results.append({'task':job['task'],'qid':job['qid'],'arm':job['arm'],'K':r['k'],'ttft_ns':t1-t0,'generation_ns':t2-t0,'output_tokens':len(tokens),'allocator_max_allocated_bytes':mem['max_allocated_bytes'],'allocator_max_reserved_bytes':mem['max_reserved_bytes'],'health_overlap':'unknown'})
assert len(results)==9 and len(identities)==9 and len(formal['annotations'])==18
bg=read(E/'background_summary002.json');assert bg['status']=='completed' and bg['snapshot_count']==18 and all(sha(k)==v for k,v in bg['source_sha256'].items()) and formal['background_summary_sha256']==sha(E/'background_summary002.json')
out={'status':'passed','independent_review':True,'jobs':9,'generations':18,'measured':9,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(f),'formal_started_sha256':sha(A/'monitor_verify002/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify002/terminal.json'),'run_started_sha256':sha(D/'monitor_pilot001/started.json'),'run_terminal_sha256':sha(D/'monitor_pilot001/terminal.json'),'reviewer_sha256':sha(__file__),'measurement_recomputation':results,'background_sha256':sha(E/'background_summary002.json'),'limitations':['Only three DEV images, one warmup and one measured generation per arm; no population-level or statistical speed claim.','Supervisor health snapshots lack monotonic intervals and time namespace, so direct overlap is unknown for every measurement.','Process ps snapshots are sparse and include their own query/fsync; per-job effect and outside activity remain unknown.','Verifier was frozen after GPU completion but before timing interpretation; formal first attempt failed and remains recorded.','Pixel observer/EADP light adapter overhead is included. GPU allocator maxima are full-generation PyTorch values, not NVML/device-wide peak.','EADP full processor input tensor is not archived for offline rehash; runtime identity check is evidenced by recorded status.'],'errors':[]};a.out.write_text(json.dumps(out,indent=2)+'\n');print('passed',flush=True)
