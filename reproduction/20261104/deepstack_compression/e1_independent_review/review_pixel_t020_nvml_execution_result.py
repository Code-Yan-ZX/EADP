"""Independent complete nine-process NVML post-run audit; no score/GT."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess,sys
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');I=B/'e1_independent_review';E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept';R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate_v2';REF=B/'e6/pixel_wavefront_t020_nvml_reference';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def terminal(m,cmd):
 s=read(m/'started.json');t=read(m/'terminal.json');assert s['command']==cmd and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[];return s,t
def load(name,p,sources):
 assert sha(p)==sources[str(p.resolve())] and name not in sys.modules
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def metrics(vals):return {'known':sum(v is not None for v in vals),'unknown':sum(v is None for v in vals),'max_known_bytes':max((v for v in vals if v is not None),default=None)}
def reconstruct(metadata,pid,uuid,bounds,maxgap):
 t=[bounds[x] for x in ('release','model_load_start','model_load_end','request_start','request_end','teardown_end')];assert all(type(x) is int and x>0 for x in t) and t[0]<t[1]<t[2]<=t[3]<t[4]<=t[5]
 assert metadata and metadata[0]['query_end_ns']<=t[0] and metadata[-1]['query_end_ns']>=t[-1]
 widths=[];gaps=[];spans=[];parsed=[]
 for i,m in enumerate(metadata):
  a,b=m['query_start_ns'],m['query_end_ns'];assert m['status']=='completed' and m['gpu_uuid']==uuid and 0<a<=b and b-a<=maxgap;widths.append(b-a)
  if i:prev=metadata[i-1];assert a>=prev['query_end_ns'] and b-prev['query_start_ns']<=maxgap;gaps.append(a-prev['query_end_ns']);spans.append(b-prev['query_start_ns'])
  dev=m['device_memory_bytes'];assert set(dev)=={'total','used','free','reserved'};procs=m['processes'];assert len({p['pid'] for p in procs})==len(procs)
  target=[p for p in procs if p['pid']==pid and p['type'] in ('C','C+G')];assert len(target)<=1
  assert not [p for p in procs if p['pid']!=pid and p['type'] in ('C','C+G')]
  graphics=[p['pid'] for p in procs if p['pid']!=pid and p['type']=='G'];parsed.append((target[0]['used_bytes'] if target else None,bool(target),dev,graphics))
 windows={'pre_cuda':(None,t[0]),'model_load':(t[1],t[2]),'request':(t[3],t[4]),'teardown':(t[4],t[5])};result={}
 for name,(a,b) in windows.items():
  ix=[i for i,s in enumerate(metadata) if (a is None or s['query_end_ns']>=a) and s['query_start_ns']<=b];inside=[i for i in ix if (a is None or metadata[i]['query_start_ns']>=a) and metadata[i]['query_end_ns']<=b]
  values={'target':[parsed[i][0] for i in ix]}
  for k in ('total','used','free','reserved'):values[k]=[parsed[i][2][k] for i in ix]
  result[name]={'overlap_indices':ix,'fully_contained_indices':inside,'overlap_metrics':{k:metrics(v) for k,v in values.items()},'target_absent':sum(not parsed[i][1] for i in ix),'target_present_unknown':sum(parsed[i][1] and parsed[i][0] is None for i in ix),'graphics_pids':sorted({p for i in ix for p in parsed[i][3]})}
 return {'status':'summarized','stages':result,'sample_count':len(metadata),'max_unsampled_gap_ns':max(gaps,default=None),'max_possible_sample_spacing_ns':max(spans,default=None),'max_query_duration_ns':max(widths)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();assert out==I/'pixel_t020_nvml_execution_result_review.json' and not out.exists()
 formal=A/'verification001.json';terminal(A/'monitor_verify001',[PY,str(A/'verify.py'),'--out',str(formal)]);terminal(E/'monitor_run001',[PY,str(E/'launch.py'),'--run-id','001']);v=read(formal);plan=read(E/'gpu_plan001.json');sources=plan['source_sha256'];assert v['status']=='passed' and v['jobs']==9 and v['GT_opened'] is False and v['scoring_performed'] is False and v['errors']==[] and v['gpu_plan_sha256']==sha(E/'gpu_plan001.json') and v['formal_plan_sha256']==sha(A/'verification_plan001.json') and v['verifier_sha256']==sha(A/'verify.py') and all(sha(k)==d for k,d in sources.items())
 ref=load('_independent_pixel_nvml_refs',REF/'reference.py',sources);references,_=ref.load_refs();guard=load('_independent_pixel_nvml_guard',R/'guard.py',sources);run=E/'run_001';jobs=plan['jobs'];journal=lines(run/'journal.jsonl');assert len(jobs)==9 and len(journal)==18 and read(run/'final.json')=={'status':'passed','jobs':9} and len(v['results'])==9
 cpu=read(B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');containers=set();identities=set();rows=[]
 for i,job in enumerate(jobs):
  assert job['order']==i and journal[2*i]['event']=='started' and journal[2*i+1]['event']=='completed' and journal[2*i]['job']==journal[2*i+1]['job']==job
  outdir=run/('job_%03d'%i);fm=v['results'][i];assert fm['job']==job and all(sha(k)==d for k,d in fm['files_sha256'].items())
  terminal(run/('monitor_job_%03d'%i),[PY,str(E/'controller_entry.py'),'--job-index',str(i)])
  launch=read(outdir/'launch.json');expected=read(outdir/'expected.json');ready=read(outdir/'worker_ready.json');release=read(outdir/'release.json');stopped=read(outdir/'stopped.json');life=read(outdir/'controller_result.json');ms=read(outdir/'memory_summary.json');identity=read(outdir/'identity.json');raw=read(outdir/'raw.json');done=read(outdir/'worker_complete.json')
  assert launch['job']==ms['job']==identity['job']==raw['job']==done['job']==job and done['status']=='completed' and done['post_source_and_weight_check'] is True and life['status']=='passed' and life['phase']=='completed' and life['error'] is None and journal[2*i+1]['result']==ms
  cid=expected['container_id'];pid=expected['host_pid'];uuid=expected['gpu_uuid'];assert cid not in containers and (pid,expected['start_ticks']) not in identities;containers.add(cid);identities.add((pid,expected['start_ticks']))
  assert expected['mode']==job['arm'] and all(ready[k]==release[k]==expected[k] for k in ('mode','nonce','container_id')) and ready['namespace_pid']==1 and ready['cuda_initialized'] is False and release['status']=='release' and all(stopped[k]==expected[k] for k in ('container_id','nonce','host_pid','start_ticks','boot_id')) and stopped['status']=='stopped'
  obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=obj['State'];assert obj['Id']==cid and state['Running'] is False and state['Pid']==state['ExitCode']==0 and state['OOMKilled'] is False and stopped['inspect_sha256']==sha(outdir/'stopped_inspect.json')
  saved=read(outdir/'running_inspect.json');worker=[str(E/'worker.py'),'--job-index',str(i),'--out',str(outdir),'--nonce',expected['nonce']];validated=guard.validate_running(saved[0],launch['docker_create'],cpu['container_image'],'eadp-memory-pixel-'+expected['nonce'],worker);assert validated['container_id']==cid and validated['host_pid']==pid and (obj['Config'].get('Labels') or {}).get('eadp.memory.nonce')==expected['nonce']
  events=lines(outdir/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample'];metadata=lines(outdir/'device_metadata.jsonl');assert len(samples)==len(metadata)>0 and events[0]['status']=='sampling' and events[-1]['type']=='summary' and events[-1]['samples']==len(samples)
  sm=outdir/'sampler_monitor';_,sterm=terminal(sm,launch['sampler_worker']);assert read(outdir/'sampler_complete.json')['status']=='completed' and read(outdir/'sampler_complete.json')['sample_count']==len(samples) and read(outdir/'sampler_complete.json')['interval_ns']==500000000 and read(outdir/'sampler_ready.json')=={'started':events[0],'first_sample':samples[0]}
  assert samples[0]['used_bytes'] is None and samples[0]['other_gpu_processes']==[] and release['first_sample_ns']==samples[0]['query_end_ns'] and samples[-1]['query_start_ns']>=stopped['monotonic_ns']
  for j,(m,s) in enumerate(zip(metadata,samples)):
   assert s['host_pid']==pid and s['gpu_uuid']==uuid and s['start_ticks']==expected['start_ticks'] and s['boot_id']==expected['boot_id'] and s['query_start_ns']<=m['query_start_ns']<=m['query_end_ns']<=s['query_end_ns'] and s['monotonic_ns']==s['query_end_ns'] and s['other_gpu_processes']==[]
   if j:assert s['query_start_ns']>=samples[j-1]['query_end_ns'] and s['query_end_ns']-samples[j-1]['query_start_ns']<=expected['max_sample_gap_ns']
   target=[p['used_bytes'] for p in m['processes'] if p['pid']==pid and p['type'] in ('C','C+G')];assert len(target)<=1 and s['used_bytes']==(target[0] if target else None)
  events=lines(outdir/'worker_events.jsonl');names=('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end');marks={}
  for e in events:
   if e.get('event') in names:assert e['event'] not in marks;marks[e['event']]=e
  assert set(marks)==set(names) and marks['barrier_released']['job']==job and marks['barrier_released']['first_sample_ns']==release['first_sample_ns'];bounds={'release':marks['barrier_released']['monotonic_ns'],**{n:marks[n]['monotonic_ns'] for n in names[1:]}}
  stages=reconstruct(metadata,pid,uuid,bounds,expected['max_sample_gap_ns']);assert all(stages[k]==ms['stage_summary'][k]==fm['stage_summary'][k] for k in stages)
  for event,field in (('model_load_end','allocator_model_load'),('request_end','allocator_request')):
   alloc={k:marks[event][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')};assert alloc==ms[field]==fm[field] and all(type(x) is int and x>=0 for x in alloc.values()) and alloc['reserved_peak_bytes']>=alloc['allocated_peak_bytes']
  r=references[(job['task'],str(job['qid']),job['arm'])];assert identity['status']=='passed' and identity['GT_opened'] is False and identity['scoring_performed'] is False and all(identity[a]==r[b] for a,b in (('processor','processor'),('selected_indices','selected_indices'),('output_ids','tokens'),('eos_ids','eos_ids'),('generation_kwargs','generation_kwargs')))
  n=raw['accepted_input_length'];full=raw['full_sequence_ids'];assert type(n) is int and r['processor']['input_ids']['shape']==[1,n] and r['processor']['input_ids']['dtype']=='torch.int64' and full[n:]==r['tokens'] and hashlib.sha256(struct.pack('<'+'q'*n,*full[:n])).hexdigest()==r['processor']['input_ids']['sha256']
  assert fm['container_id']==cid and fm['host_pid']==pid and fm['gpu_uuid']==uuid and fm['sample_count']==len(samples) and fm['identity']=={'prompt_length':n,'generated_length':len(full)-n,'selected_count':len(r['selected_indices'])}
  rows.append({'job':job,'container_id':cid,'sample_count':len(samples),'files_sha256':fm['files_sha256']})
 assert len(containers)==len(identities)==9 and all(sha(k)==d for k,d in sources.items())
 proof={'status':'passed','independent_review':True,'jobs':9,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'formal_sha256':sha(formal),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(E/'monitor_run001/started.json'),'run_terminal_sha256':sha(E/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'journal_sha256':sha(run/'journal.jsonl'),'rows':rows,'errors':[],'limits':['0.5-second MiB samples can miss transient peaks; target absence and unknown are not zero.','Device, process and allocator memory differ; overlapping query windows can extend beyond the stage.','Prior full visual intermediates not archived; online identity captures and memory instrumentation can affect memory.','Three posthoc anchors and different dynamic K do not represent equal compute or population; no TTFT claim.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
