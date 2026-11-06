"""Independent reconstruction of full latency records and paired quantiles."""
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];H=R/'e4_c_latency_analysis';F=R/'e4_c_latency_full';I=R/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
report=H/'analysis001/analysis.json'; recpath=H/'analysis001/records.jsonl';p=read(H/'analysis_plan001.json');res=read(report);launch=read(H/'analysis_launch_plan001.json');resolution=read(H/'analysis_launch_resolution001.json');s=mon(H/'monitor_analysis001');mon(I/'monitor_e4c_latency_analysis_prelaunch');mon(I/'monitor_e4c_latency_full001');mon(I/'monitor_e4c_latency_manual_overlap')
planned=launch['command'][launch['command'].index('--')+1:];actual=s['command'];cwd=Path(s['cwd']);assert len(planned)==len(actual)==4 and planned[0]==actual[0] and (cwd/actual[1]).resolve()==Path(planned[1]).resolve() and actual[2]==planned[2]=='--out' and (cwd/actual[3]).resolve()==Path(planned[3]).resolve()
assert resolution['status']=='equivalent_paths' and resolution['actual_started_sha256']==sha(H/'monitor_analysis001/started.json') and resolution['actual_worker_argv']==actual
assert res['status']=='completed' and res['records_sha256']==sha(recpath) and res['n_jobs']==5940 and res['n_warmup']==180 and res['n_measured']==5760 and res['all_samples_retained'] is True
assert res['plan_sha256']==sha(H/'analysis_plan001.json') and res['supplement_sha256']==sha(H/'analysis_plan001_supplement.json') and res['approval']==launch['approval']
assert all(sha(path)==v for path,v in launch['approval']['prelaunch']['source_sha256'].items())
up=launch['approval']['upstream'];journal=F/'full_001/journal.jsonl';fa=F/'full_001_annotations.jsonl';ma=H/'manual_overlap001/annotations.jsonl';assert sha(journal)==up['journal_sha256'] and sha(fa)==up['formal_annotations_sha256'] and sha(ma)==up['manual_annotations_sha256']
records=lines(recpath);events=lines(journal);formal=lines(fa);manual=lines(ma);sched=read(F/'full_001/frozen.json')['schedule']['schedule'];assert len(records)==len(manual)==len(sched)==5940 and len(events)==len(formal)==11880
for i,(r,j,m) in enumerate(zip(records,sched,manual)):
 e=events[2*i];d=events[2*i+1];a=formal[2*i];ad=formal[2*i+1];assert e['status']=='generated' and d['status']=='completed' and all(x['job']==j for x in (e,d,a,ad,m)) and j['order']==i
 for key in ('task','group_id','mode','protocol','phase','replicate'):assert r[key]==j[key]
 assert r['qid']==str(j['qid']);b=e['result']['boundaries'];t0=b['t0_before_image_read_ns'];first=b['first_token_selected_ns'];end=b['generation_complete_ns'];assert 0<=t0<first<=end
 assert math.isclose(r['ttft_ms'],(first-t0)/1e6,rel_tol=0,abs_tol=1e-10) and math.isclose(r['end_to_end_ms'],(end-t0)/1e6,rel_tol=0,abs_tol=1e-10)
 for k in ('output_tokens','eos','truncated'):assert r[k]==e['result'][k]
 for k in ('monitor_overlap','monitor_snapshot_indices'):assert r[k]==a[k]
 for k in ('manual_health_overlap','manual_health_indices','manual_nvml_overlap','manual_nvml_indices'):assert r[k]==m[k]
tasks=['DocVQA_VAL','TextVQA_VAL','AI2D_TEST'];modes=['C-ON','C-DROP','C-SKIP'];protocols=['natural','fixed32'];metrics=['ttft_ms','end_to_end_ms'];pairs=[('C-DROP','C-ON'),('C-SKIP','C-ON'),('C-SKIP','C-DROP')]
def linear_sorted(vals,q):
 a=np.sort(vals,axis=-1);x=q*(a.shape[-1]-1);lo=int(x);hi=math.ceil(x);return a[...,lo]+(a[...,hi]-a[...,lo])*(x-lo)
def close(a,b):assert np.allclose(a,b,atol=1e-8,rtol=1e-11)
rng=np.random.default_rng(p['bootstrap']['seed']);checks=0
for task in tasks:
 rows=[x for x in records if x['task']==task and x['phase']=='measure'];groups=sorted({x['group_id'] for x in rows});assert len(groups)==64 and len(rows)==1920
 lookup={(x['group_id'],x['mode'],x['protocol'],x['replicate']):x for x in rows};assert len(lookup)==1920
 draws=rng.integers(0,64,size=(10000,64));taskres=res['tasks'][task];assert taskres['n_image_groups']==64 and taskres['repeats_per_group']==5 and taskres['measurements_per_mode_protocol']==320
 for protocol in protocols:
  pr=taskres['protocols'][protocol]
  for mode in modes:
   selected=[lookup[(g,mode,protocol,k)] for g in groups for k in range(5)];rr=pr['modes'][mode];tokens=np.array([x['output_tokens'] for x in selected]);assert rr['n']==320 and rr['eos']==sum(x['eos'] for x in selected) and rr['truncated']==sum(x['truncated'] for x in selected);assert rr['output_tokens']['min']==int(tokens.min()) and rr['output_tokens']['max']==int(tokens.max());close(rr['output_tokens']['median'],linear_sorted(tokens,.5));close(rr['output_tokens']['p95'],linear_sorted(tokens,.95))
  for metric in metrics:
   for q,label in ((.5,'median'),(.95,'p95')):
    actual=pr['metrics'][metric+'_'+label];point={};boot={}
    for mode in modes:
     mat=np.array([[lookup[(g,mode,protocol,k)][metric] for k in range(5)] for g in groups]);point[mode]=linear_sorted(mat.reshape(-1),q);boot[mode]=linear_sorted(mat[draws].reshape(10000,320),q);close(actual['mode_quantiles_ms'][mode],point[mode])
    for x,y in pairs:
     c=actual['paired_contrasts'][x+'_minus_'+y];assert point[y]>0 and np.all(boot[y]>0);diff=boot[x]-boot[y];ratio=boot[x]/boot[y];close(c['difference_ms'],point[x]-point[y]);close(c['ratio'],point[x]/point[y]);close(c['relative_change_pct'],100*(point[x]/point[y]-1))
     for field,vals in [('difference_95_ci_ms',diff),('ratio_95_ci',ratio),('relative_change_95_ci_pct',100*(ratio-1))]:close(c[field],[linear_sorted(vals,.025),linear_sorted(vals,.975)])
     checks+=1
for phase in ('warmup','measure'):
 rows=[x for x in records if x['phase']==phase];i=res['interference'][phase];assert i['n']==len(rows)
 for field,name in [('monitor_overlap','supervisor'),('manual_health_overlap','manual_health'),('manual_nvml_overlap','manual_nvml')]:assert i[name]==sum(x[field] for x in rows)
for w in res['interference_by_window']:
 field={'supervisor':'monitor_snapshot_indices','manual_health':'manual_health_indices','manual_nvml':'manual_nvml_indices'}[w['kind']];assert w['n_overlap']==sum(w['index'] in x[field] for x in records) and len(w['counts'])==36
 for c in w['counts']:
  rr=[x for x in records if all(x[k]==c[k] for k in ('phase','task','mode','protocol'))];assert c['n_total']==len(rr) and c['n_overlap']==sum(w['index'] in x[field] for x in rr)
assert checks==72
out=I/'e4c_latency_analysis001_review.json';v={'status':'passed','independent_review':True,'jobs_verified':5940,'measured_verified':5760,'paired_contrasts_verified':checks,'analysis_sha256':sha(report),'records_sha256':sha(recpath),'started_sha256':sha(H/'monitor_analysis001/started.json'),'terminal_sha256':sha(H/'monitor_analysis001/terminal.json'),'launch_plan_sha256':sha(H/'analysis_launch_plan001.json'),'launch_resolution_sha256':sha(H/'analysis_launch_resolution001.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['Plan used absolute paths and actual command used semantically equivalent relative paths; argv bytes differ.','Pooled image-group bootstrap conditions on this run and repeats; host effects and between-run variability remain.','Unknown or unbounded health and project activity cannot be assigned to individual jobs.'],'errors':[]}
with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
