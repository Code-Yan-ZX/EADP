"""Independent completed latency smoke audit; no references/GT or score."""
import argparse,collections,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;L=ROOT/'e4_c_latency';RUN=L/'latency_001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def supervise(start,end,script,out=None):
 s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def independent_schedule(rows,excluded):
 tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');modes=('C-ON','C-DROP','C-SKIP');seed='20261104';rank=lambda *x:hashlib.sha256('|'.join(map(str,(seed,*x))).encode()).hexdigest()
 assert len(rows)==384 and collections.Counter(r['task'] for r in rows)=={t:128 for t in tasks} and all(r['split']=='DEV' for r in rows)
 out=[];cohort={}
 for task in tasks:
  options=[x for x in rows if x['task']==task and x['group_id'] not in excluded];assert len({x['group_id'] for x in options})==len(options)
  options.sort(key=lambda x:(rank('cohort',task,x['group_id']),x['group_id']));measure,warm=options[:64],options[64:74]
  cohort[task]={'measured':[x['group_id'] for x in measure],'warmup':[x['group_id'] for x in warm]}
  for protocol in ('natural','fixed32'):
   for row in warm:
    for mode in sorted(modes,key=lambda m:rank('warm',task,protocol,row['group_id'],m)):out.append({'task':task,'qid':str(row['qid']),'group_id':row['group_id'],'phase':'warmup','protocol':protocol,'replicate':0,'mode':mode})
   for row in measure:
    for rep in range(5):
     for mode in sorted(modes,key=lambda m:rank('measure',task,protocol,row['group_id'],rep,m)):out.append({'task':task,'qid':str(row['qid']),'group_id':row['group_id'],'phase':'measure','protocol':protocol,'replicate':rep,'mode':mode})
 for i,x in enumerate(out):x['order']=i
 chosen={t:{'measure':set(cohort[t]['measured'][:2]),'warmup':set(cohort[t]['warmup'][:2])} for t in tasks};jobs=[]
 for x in out:
  if x['group_id'] not in chosen[x['task']][x['phase']] or (x['phase']=='measure' and x['replicate']!=0):continue
  jobs.append(dict(x,source_order=x['order'],order=len(jobs)))
 assert len(jobs)==72
 return {'seed':seed,'stage':'smoke','scope':'DEV batch1 warm-cache diagnostic; no GT or scores','cohort':{t:{k:sorted(v) for k,v in chosen[t].items()} for t in tasks},'schedule':jobs,'counts':{'warmup':36,'measured':36},'derivation':'first two frozen measured and warmup groups per task; full schedule relative order preserved'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e4c_latency001_review.json'
 proof=L/'latency_001_verification.json';fs=L/'monitor_verify_latency001/started.json';ft=fs.parent/'terminal.json';rs=L/'monitor_latency001/started.json';rt=rs.parent/'terminal.json';planpath=L/'latency_launch_plan001.json';annotations=L/'latency_001_annotations.jsonl'
 supervise(fs,ft,L/'latency_verify.py',proof);started,terminal=supervise(rs,rt,L/'latency_launch.py');formal=read(proof);plan=read(planpath)
 assert formal['status']=='passed' and formal['stage']=='smoke' and formal['n_jobs']==formal['n_generated']==72 and formal['errors']==[] and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['verifier_sha256']==sha(L/'latency_verify.py')
 for path,digest in formal['evidence_sha256'].items():assert sha(path)==digest
 sources=formal['source_sha256'];assert sources==plan['approval_bundle']['source_sha256'] and set(sources)==set(plan['source_snapshots'])
 for path,digest in sources.items():assert plan['source_snapshots'][path]['sha256']==digest and sha(plan['source_snapshots'][path]['snapshot'])==digest
 launch=read(RUN/'launch.json');ex=read(RUN/'exit.json');frozen=read(RUN/'frozen.json');done=read(RUN/'worker_complete.json')
 assert launch['approval_bundle']==plan['approval_bundle'] and frozen['source_sha256']==sources and frozen['identity']==plan['approval_bundle']['light_identity'] and frozen['stage']=='smoke' and frozen['GT_opened'] is False and frozen['scoring_performed'] is False and done['status']=='completed' and done['jobs']==72
 assert ex['controller']==launch['controller'] and ex['controller']['pid']==terminal['worker_pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['timed_out'] is False and ex['interrupted'] is None and ex['failure_type'] is None and ex['gpu_pids_after']==[]
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==ex['container_identity']['image'] and obj['State']['Running'] is False and obj['State']['Pid']==0 and obj['State']['ExitCode']==0
 sys.path.insert(0,str(L));from input_preflight import load_accepted_e1_references
 from real_worker import references_to_rows
 ep=ROOT/'e1/e1full001_completion.json';es=ROOT/'e1/monitorverifye1full001/started.json';et=es.parent/'terminal.json';rows,refs=load_accepted_e1_references(ep,es,et);rows=references_to_rows(rows,refs)
 excluded={x['group_id'] for x in lines(ROOT/'data_manifest/anchor_exclusion/anchor_groups.jsonl')};assert len(excluded)==16
 schedule=independent_schedule(rows,excluded);assert schedule==frozen['schedule']
 author=read(ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 events=lines(RUN/'journal.jsonl');annotated=lines(annotations);health=lines(rs.parent/'health.jsonl');assert len(events)==len(annotated)==144 and len(health)>=2 and health[-1]['event']=='terminal_snapshot'
 ns=events[0]['result']['boundaries']['time_namespace'];windows=[]
 for h in health:
  w=h['health_snapshot_window_monotonic_ns'];assert h['time_namespace']==ns and len(w)==2 and 0<=w[0]<=w[1];windows.append(w)
 gap=int((started['snapshot_seconds']+2*started['poll_seconds']+30)*1e9);assert all(a[1]<=b[0] and b[0]-a[1]<=gap for a,b in zip(windows,windows[1:]))
 overlaps=0;all_starts=[];all_ends=[]
 for job,g,c,ag,ac in zip(schedule['schedule'],events[::2],events[1::2],annotated[::2],annotated[1::2]):
  assert g['status']=='generated' and c['status']=='completed' and g['job']==c['job']==job and g['worker_pid']==c['worker_pid']==1 and g['measurement_role']==c['measurement_role']=='latency_interleaved' and ac==c
  r=g['result'];ids=r['output_ids'];fixed=job['protocol']=='fixed32';maximum=32 if fixed else 2048;recipe={'do_sample':False,'max_new_tokens':maximum,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  assert r['mode']==job['mode'] and r['protocol']==job['protocol'] and r['generation_scalars']==recipe and r['min_new_tokens']==(32 if fixed else None) and 1<=len(ids)<=maximum and all(type(x) is int for x in ids) and r['output_tokens']==len(ids) and (not fixed or len(ids)==32)
  eos=ids[-1] in r['eos_ids'];trunc=not fixed and not eos and len(ids)==2048;assert r['eos']==eos and r['truncated']==trunc and (fixed or eos or trunc) and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer']
  b=r['boundaries'];ticks=[b[k] for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns')];assert all(type(x) is int and x>=0 for x in ticks) and ticks==sorted(ticks) and b['first_criteria_length']==1 and b['criteria_calls']==len(ids) and b['explicit_sync_boundaries']==['before_t0','first_token','generation_end'] and b['logits_ready_ns'] is None and r['instrumented_logits_ready'] is False
  assert b['time_namespace']==ns and b['clock_implementation']=='clock_gettime(CLOCK_MONOTONIC)';indices=[i for i,(lo,hi) in enumerate(windows) if lo<=ticks[2] and ticks[0]<=hi];assert ag==dict(g,monitor_overlap=bool(indices),monitor_snapshot_indices=indices)
  overlaps+=bool(indices);all_starts.append(ticks[0]);all_ends.append(ticks[2])
 assert windows[0][0]<=min(all_starts) and windows[-1][1]>=max(all_ends) and formal['n_monitor_overlaps']==overlaps
 result={'status':'passed','independent_review':True,'n_jobs':72,'n_generated':72,'n_monitor_overlaps':overlaps,'GT_opened':False,'scoring_performed':False,'errors':[],
         'formal_proof_sha256':sha(proof),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'run_started_sha256':sha(rs),'run_terminal_sha256':sha(rt),'launch_plan_sha256':sha(planpath),'health_sha256':sha(rs.parent/'health.jsonl'),'annotations_sha256':sha(annotations),'reviewer_sha256':sha(__file__),
         'limitations':['Single-process interleaving; all 72 jobs retained, including health overlaps.','Snapshot windows cannot rule out residual interference.','No exact peak VRAM, quality score, or full-stage speed claim.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','jobs':72,'overlaps':overlaps}))
if __name__=='__main__':main()
