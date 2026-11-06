"""Independent three-anchor graph-TTFT result review; no GT or new GPU execution."""
import argparse,collections,hashlib,json,math,os,random,statistics,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e5/graph_ttft_pilot_v3';O=D/'run_smoke001/observations'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with open(p) as f:return [json.loads(x) for x in f if x.strip()]
def req(x,m):
 if not x:raise ValueError(m)
def schedule(anchors):
 rng=random.Random(20261105);out=[]
 for phase,n in [('warmup',1),('measured',5)]:
  for rep in range(n):
   tmp=list(anchors);rng.shuffle(tmp)
   for row in tmp:
    methods=['original','graph'];rng.shuffle(methods)
    for method in methods:out.append({'phase':phase,'repeat':rep,'task':row['task'],'qid':row['qid'],'method':method,'protocol':'fixed32'})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json');la=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json');start=read(D/'monitor_smoke001/started.json');term=read(D/'monitor_smoke001/terminal.json');pre=read(D/'independent_prelaunch_review.json')
 req(len(plan['anchors'])==3 and len({(x['task'],x['qid']) for x in plan['anchors']})==3 and la['plan_sha256']==sha(D/'plan001.json'),'plan')
 for f,h in plan['sha256'].items():req(sha(f)==h,'source '+f)
 req(start['worker_pid']==term['worker_pid']==la['controller']['pid']==ex['controller']['pid'] and la['controller']==ex['controller'] and term['reason']=='completed' and term['worker_returncode']==0 and term['remaining_process_group']==[],'supervisor')
 req(ex['returncode']==0 and ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['interrupted'] is None and ex['gpu_pids_after']==[],'exit')
 obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],check=True,capture_output=True,text=True,timeout=15).stdout)[0];st=obj['State'];req(obj['Id']==ex['container_identity']['container_id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled'],'actual container')
 req(obj['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and obj['HostConfig']['NetworkMode']=='none','container command')
 devices=['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'];req({(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in obj['HostConfig']['Devices']}=={(x,x,'rwm') for x in devices},'devices')
 mounts=[]
 for i,arg in enumerate(la['command'][:-1]):
  if arg=='--mount':
   d=dict(z.split('=',1) if '=' in z else (z,True) for z in la['command'][i+1].split(','));mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in d))
 req(sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind'),'mounts')
 req(pre['status']=='passed' and pre['independent_review'] is True and pre['plan_sha256']==sha(D/'plan001.json') and pre['reviewer_sha256']==sha(pre['reviewer_path']),'prelaunch')
 preflight=read(D/'preflight001.json');req(preflight['status']=='passed' and not preflight['cuda_initialized'] and not preflight['model_loaded'] and preflight['source_plan_sha256']==sha(D/'plan001.json') and preflight['triton_cache_probe']['writable'] is True,'preflight')
 frozen=lines(B/'e1/input_preparation_002/inputs.jsonl');lookup={(x['task'],str(x['qid'])):x for x in frozen}
 for anchor in plan['anchors']:
  x=lookup[(anchor['task'],anchor['qid'])];req(all(x[k]==anchor[k] for k in ('group_id','message_sha256','image_sha256')) and sha(x['image_path'])==anchor['image_sha256'],'input anchor')
 jobs=schedule(plan['anchors']);req(preflight['schedule']==jobs and len(jobs)==36,'schedule')
 records=lines(O/'records.jsonl');raw=lines(O/'generated_raw.jsonl');gen=lines(O/'generated.jsonl');comps=lines(O/'comparisons.jsonl');progress=lines(O/'progress.jsonl');complete=read(O/'worker_complete.json');req(len(records)==len(raw)==len(gen)==36 and len(comps)==18 and len(progress)==73,'inventory');req(complete['status']=='completed' and complete['jobs']==36 and complete['comparisons']==18 and complete['no_GT'] and not complete['scoring'],'completion')
 expected_progress=[{'event':'worker_started'}]
 for j in jobs:expected_progress.extend([{'event':'job_started','job':j},{'event':'job_completed','job':j}])
 req(progress==expected_progress,'progress')
 setups=read(O/'graph_setup.json');req(setups['setup_excluded_from_timing'] is True and math.isfinite(setups['capture_wall_ms']) and setups['capture_wall_ms']>0,'setup');setup={'capture_wall_ms':setups['capture_wall_ms'],'allocated_before':setups['before_bytes']['allocated'],'allocated_after':setups['after_bytes']['allocated'],'allocated_delta':setups['after_bytes']['allocated']-setups['before_bytes']['allocated'],'reserved_before':setups['before_bytes']['reserved'],'reserved_after':setups['after_bytes']['reserved'],'reserved_delta':setups['after_bytes']['reserved']-setups['before_bytes']['reserved']}
 paired={};comparisons=[];per={};times=[]
 for job,r,x,y in zip(jobs,records,raw,gen):
  req(r['job']==x['job']==y['job']==job and r['status']=='completed' and r['result']==y['result'],'generation record')
  z=r['result'];ids=r['ordered_indices'];tokens=z['output_ids'];req(len(ids)==256 and len(set(ids))==256 and all(type(i) is int and 0<=i<1024 for i in ids),'selection')
  req(len(x['output_ids'])==1 and x['output_ids'][0]==tokens and len(tokens)==z['output_tokens']==32 and z['mode']=='C-ON' and z['protocol']=='fixed32' and z['min_new_tokens']==32 and z['generation_scalars']['max_new_tokens']==32 and z['generation_scalars']['do_sample'] is False,'fixed generation')
  b=z['boundaries'];req(b['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)' and b['first_criteria_length']==1 and b['criteria_calls']==32 and b['explicit_sync_boundaries']==['before_t0','first_token','generation_end'] and type(b['t0_before_image_read_ns']) is int and b['t0_before_image_read_ns']<b['first_token_selected_ns']<b['generation_complete_ns'],'first token boundary')
  d=(b['first_token_selected_ns']-b['t0_before_image_read_ns'])/1e6;times.append({'job':job,'first_token_ms':d});key=(job['phase'],job['repeat'],job['task'],job['qid'],job['protocol'])
  if key not in paired:paired[key]=r
  else:
   prev=paired.pop(key);req({prev['job']['method'],job['method']}=={'original','graph'},'pair method');eo=prev['ordered_indices']==ids;et=prev['result']['output_ids']==tokens;comparisons.append({'job':{k:job[k] for k in ('phase','repeat','task','qid','protocol')},'ordered_indices_equal':eo,'output_ids_equal':et,'eligible_for_equal_output_timing':eo and et})
 req(not paired and comparisons==comps,'pair check');unequal=sum(not x['eligible_for_equal_output_timing'] for x in comps);req(unequal==complete['unequal_comparisons'],'comparison count')
 for anchor in plan['anchors']:
  label=anchor['task']+':'+anchor['qid'];per[label]={}
  for method in ('original','graph'):
   values=[x['first_token_ms'] for x in times if x['job']['phase']=='measured' and x['job']['task']==anchor['task'] and x['job']['qid']==anchor['qid'] and x['job']['method']==method];req(len(values)==5,'repeats');per[label][method]={'n':5,'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'raw_ms':values}
  o=per[label]['original']['median_ms'];g=per[label]['graph']['median_ms'];per[label]['relative_change_pct']=(g/o-1)*100
 aggregate={}
 if unequal==0:
  for method in ('original','graph'):aggregate[method+'_median_of_anchor_medians_ms']=statistics.median(x[method]['median_ms'] for x in per.values())
  aggregate['relative_change_of_medians_pct']=(aggregate['graph_median_of_anchor_medians_ms']/aggregate['original_median_of_anchor_medians_ms']-1)*100
 files=[D/'plan001.json',D/'independent_prelaunch_review.json',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'preflight001.json',D/'background_before001.json',*[O/x for x in ('worker_complete.json','graph_setup.json','progress.jsonl','records.jsonl','generated.jsonl','generated_raw.jsonl','comparisons.jsonl')]]
 out={'status':'passed','independent_review':True,'anchors':3,'jobs':36,'comparisons':18,'unequal_comparisons':unequal,'per_anchor':per,'aggregate_only_if_all_pairs_equal':aggregate,'graph_setup':setup,'all_times':times,'files_sha256':{str(x):sha(x) for x in files},'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['Three DEV anchors, five warm-cache measured repetitions, fixed32; exploratory and not original full M1 criterion.','First-token boundary is from before image read to selected first token; graph capture/setup excluded and reported separately; no production or full-generation performance claim.','Graph replays the fused kernel; exact ordered selections and 32 outputs match only for the observed pairs, not universally.','Allocator deltas are PyTorch process allocated/reserved, not device-wide peak; other background processes and temperature may affect timing.','First-token criterion and P3/DS adapter are instrumentation; end-of-generation timestamps include raw output fsync and are not used as performance.','Worker finally restores monkey patches and container exited; no independent per-job post-cleanup hook inventory was archived.','Selected-ID and output equality do not independently reconstruct every prefill or DeepStack tensor.'],'errors':[]}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 print('independent review completed')
if __name__=='__main__':main()
