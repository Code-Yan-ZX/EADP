"""Independent 24-process Pixel wavefront timing integrity review; no timing aggregation."""
import argparse,hashlib,json,os,struct,subprocess,sys
from collections import Counter
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5';I=B/'e1_independent_review';OB=B/'e6/pixel_wavefront_t020_timing_observer_v2';ID=B/'e6/pixel_wavefront_model_t020_run_v2';IA=B/'e6/pixel_wavefront_t020_accept_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);c=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()],str(mon)
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
 return s,t
def hit(ws,a,b):return [i for i,(x,y) in enumerate(ws) if x<=b and a<=y]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();assert sys.byteorder=='little'
 formal=A/'verification001.json';terminal(A/'monitor_verify001',A/'verify_entry.py',formal);terminal(A/'monitor_snapshot001',A/'snapshot_entry.py',R/'run_001_sources');started,term=terminal(R/'monitor_run001',R/'launch.py');terminal(OB/'monitor_watch001',OB/'observe.py');terminal(I/'monitor_pixel_wavefront_t020_timing_observer_v2_prelaunch',I/'review_pixel_wavefront_t020_timing_observer_v2_prelaunch.py',I/'pixel_wavefront_t020_timing_observer_v2_prelaunch_review.json')
 f=read(formal);plan=read(R/'gpu_plan001.json');assert f['status']=='passed' and (f['jobs'],f['warmups'],f['measured'])==(24,6,18) and f['GT_opened'] is False and f['scoring_performed'] is False and f['all_retained'] is True and f['errors']==[] and f['formal_plan_sha256']==sha(A/'verification_plan001.json') and f['gpu_plan_sha256']==sha(R/'gpu_plan001.json') and f['run_health_sha256']==sha(R/'monitor_run001/health.jsonl')
 snap=R/'run_001_sources/manifest.json';sm=read(snap);assert f['snapshot_manifest_sha256']==sha(snap) and sm['status']=='passed' and sm['gpu_plan_sha256']==sha(R/'gpu_plan001.json') and set(sm['sources'])==set(plan['source_sha256'])
 for path,digest in plan['source_sha256'].items():
  x=sm['sources'][path];copy=Path(x['copy']).resolve();assert x['sha256']==digest and copy.is_relative_to(snap.parent.resolve()) and sha(copy)==digest
 design=read(B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json');jobs=plan['jobs'];assert jobs==design['jobs'] and len(jobs)==24 and Counter(x['phase'] for x in jobs)=={'warmup':6,'measurement':18}
 refproof=I/'pixel_wavefront_t020_run_v2_result_review.json';terminal(IA/'monitor_verify001',IA/'verify.py',IA/'verification001.json');terminal(I/'monitor_pixel_wavefront_t020_run_v2_result',I/'review_pixel_wavefront_t020_run_v2_result.py',refproof)
 iq=read(refproof);iff=read(IA/'verification001.json');assert iq['status']==iff['status']=='passed' and iq['independent_review'] is True and iq['formal_sha256']==sha(IA/'verification001.json') and iq['reviewer_sha256']==sha(I/'review_pixel_wavefront_t020_run_v2_result.py') and iq['GT_opened'] is False and iq['scoring_performed'] is False
 refs={}
 for j in read(ID/'execution_plan001.json')['jobs']:
  path=ID/'run_001'/f"arm_{j['order']:03d}"/'observations/record.json';assert iff['files_sha256'][str(path)]==iq['files_sha256'][str(path)]==sha(path);r=read(path);key=(j['task'],str(j['qid']),j['arm']);assert key not in refs and r['status']=='completed' and r['GT_opened'] is False and r['scoring_performed'] is False and all(str(r[k])==str(j[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')) and r['path']==j['arm'];refs[key]=r
 assert len(refs)==6
 events=lines(R/'run_001/controller.jsonl');assert len(events)==50 and events[0]=={'event':'controller_started','jobs':24} and events[-1]=={'event':'all_completed','jobs':24,'warmups':6,'measured':18} and read(R/'run_001/final.json')=={'status':'completed','jobs':24,'warmups':6,'measured':18,'post_run_source_and_approval_checked':True}
 basecid=read(R/'run_preflight001/exit.json')['container_identity']['container_id'];base=json.loads(subprocess.check_output(['docker','inspect',basecid]))[0]
 health=lines(R/'monitor_run001/health.jsonl');healthw=[]
 for h in health:
  x=h['health_snapshot_window_monotonic_ns'];assert isinstance(x,list) and len(x)==2 and 0<x[0]<=x[1] and h['time_namespace'];healthw.append((x[0],x[1],h['time_namespace']))
 op=read(OB/'plan001.json');opr=read(I/'pixel_wavefront_t020_timing_observer_v2_prelaunch_review.json');assert op['gpu_plan_sha256']==sha(R/'gpu_plan001.json') and op['formal_plan_sha256']==sha(A/'verification_plan001.json') and opr['status']=='passed' and opr['plan_sha256']==sha(OB/'plan001.json') and (OB/'stop001').is_file()
 obs=lines(OB/'observations001.jsonl');assert len(obs)>=3 and len(obs)%2==1 and obs[-1]['event']=='observer_end' and obs[-1]['reason']=='stop' and obs[-1]['samples']==(len(obs)-1)//2
 ps=[];docker=[];first=[];previous=-1;ns=obs[-1]['time_namespace']
 for i in range((len(obs)-1)//2):
  x,y=obs[2*i:2*i+2];assert x['event']=='snapshot' and y['event']=='logging_complete' and x['index']==y['index']==i and x['time_namespace']==y['time_namespace']==ns
  p,d=x['ps'],x['docker'];a,b,c,e,z=p['query_start_ns'],p['query_end_ns'],d['query_start_ns'],d['query_end_ns'],y['first_fsync_end_ns'];assert all(type(v) is int for v in (a,b,c,e,z)) and previous<a<=b<=c<=e<=z and p['returncode']==d['returncode']==0;ps.append((a,b));docker.append((c,e));first.append((a,z));previous=z
 assert type(obs[-1]['end_ns']) is int and obs[-1]['end_ns']>=previous
 items=[];cids=set();processes=set();seen=Counter();minimum=None;maximum=None
 for i,j in enumerate(jobs):
  assert j['order']==i and events[2*i+1]=={'event':'job_started','job':j};done=events[2*i+2];assert done['event']=='job_complete' and done['job']==j
  d=R/'run_001'/f'job_{i:03d}';launch=read(d/'launch.json');release=read(d/'release.json');ex=read(d/'exit.json');stored=read(d/'record.json');raw=read(d/'raw.json');r=stored['record'];metric=stored['metrics'];ref=refs[(j['task'],str(j['qid']),j['arm'])]
  assert launch['job']==release['job']==stored['job']==raw['job']==j and all(r[k]==v for k,v in j.items()) and launch['plan_sha256']==sha(R/'gpu_plan001.json') and release['plan_sha256']==sha(R/'gpu_plan001.json') and release['launch_sha256']==sha(d/'launch.json')
  ci=ex['container_identity'];cid=ci['container_id'];host=ci['host_init'];proc=(host['pid'],host['start_ticks'],host['boot_id']);assert len(cid)==64 and cid not in cids and proc not in processes and release['container_id']==done['container_id']==cid and release['host_init']==done['host_init']==host and release['controller']==launch['controller']==ex['controller']
  cids.add(cid);processes.add(proc);assert ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['client_exited_while_container_running'] is False and ex['container_stopped_verified'] is True and ex['gpu_pids_after']==[] and ex['failure_type'] is None
  actual=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert actual['Id']==cid and actual['State']['Running'] is False and actual['State']['Pid']==actual['State']['ExitCode']==0 and actual['Image']==base['Image'] and actual['HostConfig']['NetworkMode']=='none' and actual['Config']['Cmd']==[str(R/'worker.py'),'--job-index',str(i)]
  for k in ('Entrypoint','User','WorkingDir'):assert actual['Config'][k]==base['Config'][k]
  av={x.split('=',1)[0]:x for x in actual['Config']['Env']};bv={x.split('=',1)[0]:x for x in base['Config']['Env']};assert len(av)==len(actual['Config']['Env']) and len(bv)==len(base['Config']['Env']) and av.pop('CUDA_VISIBLE_DEVICES')=='CUDA_VISIBLE_DEVICES=0' and bv.pop('CUDA_VISIBLE_DEVICES')=='CUDA_VISIBLE_DEVICES=' and av==bv and sorted(json.dumps(x,sort_keys=True) for x in actual['Mounts'])==sorted(json.dumps(x,sort_keys=True) for x in base['Mounts'])
  assert {x['PathOnHost'] for x in actual['HostConfig']['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'} and not actual['HostConfig']['DeviceRequests']
  for k,q in [('processor','processor'),('selected_indices','selected_indices'),('output_ids','output_ids'),('eos_ids','eos_ids'),('generation_scalars','generation_scalars')]:assert r[k]==ref[q],(i,k)
  assert r['answer']==ref['answer'] and r['eos']==ref['eos'] and r['truncated']==ref['truncated'] and r['dynamic_k']==len(r['selected_indices'])
  full=raw['full_ids'];n=raw['input_length'];assert type(n) is int and n>0 and full[n:]==r['output_ids'] and ref['processor']['input_ids']['dtype']=='torch.int64' and ref['processor']['input_ids']['shape']==[1,n] and all(type(x) is int and -(2**63)<=x<2**63 for x in full[:n]);prompthash=hashlib.sha256(struct.pack('<%dq'%n,*full[:n])).hexdigest();assert prompthash==ref['processor']['input_ids']['sha256']==f['prompt_sha256'][str(i)]
  a,b,c=(r[x] for x in ('start_monotonic_ns','first_token_monotonic_ns','completion_monotonic_ns'));assert all(type(x) is int for x in (a,b,c)) and 0<a<b<=c and r['first_token_synchronized'] is True and r['gpu_synchronized_before_start'] is True and r['boundaries']['time_namespace']==ns
  assert metric=={'ttft_ns':b-a,'completion_ns':c-a,'raw_sha256':sha(d/'raw.json'),'selected_count':len(r['selected_indices'])} and f['rows'][i]=={'order':i,'phase':j['phase'],'ttft_ns':b-a}
  for filename in ('launch.json','release.json','exit.json','record.json','raw.json'):
   path=d/filename;assert f['files_sha256'][str(path)]==sha(path)
  hw=[(x,y) for x,y,nsp in healthw if nsp==ns];assert len(hw)==len(healthw)
  expected={'order':i,'background':{'supervisor_health':{'status':'observed','overlap_indices':hit(hw,a,b),'reason':None,'background_activity_outside_queries':'unknown'},'manual_health':{'status':'unknown','overlap_indices':[],'reason':'query coverage absent'},'nvml':{'status':'unknown','overlap_indices':[],'reason':'query coverage absent'},'external_watcher':{'status':'unknown','overlap_indices':[],'reason':'query coverage absent'}}};assert f['background_annotations'][i]==expected
  items.append({'order':i,'phase':j['phase'],'arm':j['arm'],'ps_query_overlap_indices':hit(ps,a,b),'docker_query_overlap_indices':hit(docker,a,b),'query_through_first_fsync_overlap_indices':hit(first,a,b),'second_fsync_and_residual_background':'unknown','observer_supervisor_health_window':'unknown'});seen[j['phase']]+=1;minimum=a if minimum is None else min(minimum,a);maximum=c if maximum is None else max(maximum,c)
 assert seen=={'warmup':6,'measurement':18} and len(cids)==24 and ps[0][0]<=minimum and obs[-1]['end_ns']>=maximum
 out={'status':'passed','independent_review':True,'jobs':24,'warmups':6,'measured':18,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'formal_sha256':sha(formal),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(R/'monitor_run001/started.json'),'run_terminal_sha256':sha(R/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'gpu_plan_sha256':sha(R/'gpu_plan001.json'),'snapshot_manifest_sha256':sha(snap),'source_files_verified':len(plan['source_sha256']),'observer_plan_sha256':sha(OB/'plan001.json'),'observer_review_sha256':sha(I/'pixel_wavefront_t020_timing_observer_v2_prelaunch_review.json'),'observer_started_sha256':sha(OB/'monitor_watch001/started.json'),'observer_terminal_sha256':sha(OB/'monitor_watch001/terminal.json'),'observer_log_sha256':sha(OB/'observations001.jsonl'),'observer_stop_sha256':sha(OB/'stop001'),'observer_window_annotations':items,'samples_excluded':0,'scope':'independent job identity, prompt/output, container and directly recorded observation windows; no timing aggregation','limitations':['Second observer fsync and observer supervisor health queries have unknown overlap; point observations do not prove absence of other background.','Raw-output persistence can add later completion overhead; TTFT and generation completion have distinct boundaries.','Three posthoc anchors and separate-process warmups are exploratory; capture/source snapshots cannot exclude transient source replacement.'],'errors':[]}
 with open(args.out,'x') as fp:json.dump(out,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
