"""Independent review of nine-image multistream v4 completed run; no GT."""
import argparse,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_multistream_expansion352_launch_v4';V=B/'e4_multistream_expansion352_gpu_v4';I=B/'e1_independent_review';P=B/'e4_multistream_expansion352'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()],str(m)
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def ranks(x):
 order=np.argsort(x,kind='stable');v=np.empty(len(x),dtype=np.float64);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[i]]==x[order[j]]:j+=1
  v[order[i:j]]=(i+j+1)/2;i=j
 return (v-1)/1023
def features(n):
 assert n.shape==(4,1024) and n.dtype==np.float64 and np.isfinite(n).all() and (n>=0).all()
 names=('main_cv','ds0_cv','ds1_cv','ds2_cv','main_ds0_rank_corr','main_ds1_rank_corr','main_ds2_rank_corr','four_stream_rank_disagreement');out={}
 for i in range(4):
  avg=float(np.mean(n[i]));out[names[i]]=None if avg==0 else float(np.std(n[i])/avg)
 rr=[ranks(v) for v in n]
 for i in range(3):
  a=rr[0]-np.mean(rr[0]);b=rr[i+1]-np.mean(rr[i+1]);den=math.sqrt(float(np.dot(a,a)*np.dot(b,b)));out[names[i+4]]=None if den==0 else float(np.dot(a,b)/den)
 out[names[7]]=float(np.mean(np.var(np.stack(rr),axis=0)));return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();run=V/'run001';fm=D/'verification001.json';terminal(D/'monitor_run001',D/'launch.py');terminal(D/'monitor_snapshot001',D/'snapshot.py',D/'run001_sources');terminal(D/'monitor_verify001',D/'verify.py',fm)
 f=read(fm);assert f['status']=='passed' and f['images']==9 and f['GT_opened'] is False and f['scoring_performed'] is False and f['errors']==[] and f['verifier_sha256']==sha(D/'verify.py') and f['source_manifest_sha256']==sha(D/'run001_sources/manifest.json') and f['run_started_sha256']==sha(D/'monitor_run001/started.json') and f['run_terminal_sha256']==sha(D/'monitor_run001/terminal.json')
 for p,digest in f['files_sha256'].items():assert sha(p)==digest
 snap=read(D/'run001_sources/manifest.json');assert snap['status']=='passed' and all(sha(v['snapshot'])==sha(k)==v['sha256'] for k,v in snap['sources'].items())
 gp=D/'gpu_gate.py';spec=importlib.util.spec_from_file_location('_independent_multistream_v4_gpu_gate',gp);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);assert sha(gp)==read(I/'e4_multistream_expansion352_v4_guard_gpu_review.json')['source_sha256'][str(gp)]
 guard=g.admit();launch=read(run/'launch.json');ex=read(run/'exit.json');inspect=g.inspect_gpu(launch,ex);assert launch['v4_guard']==guard and ex['gpu_pids_after']==[] and ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['failure_type'] is None
 selection=read(P/'selection001.json')['pilot'];selected={(r['task'],str(r['qid'])):r for r in selection};assert len(selected)==9
 e1=read(B/'e1/e1full001_completion.json');ep=Path(e1['files']['journal']['path']);assert e1['status']=='passed' and sha(ep)==e1['files']['journal']['sha256'];e4p=B/'e4_panel_v2/panel_003/records.jsonl';e4formal=read(B/'e4_panel_v2/panel_003_verification.json');assert e4formal['status']=='passed' and sha(e4p)==e4formal['files_sha256'][str(e4p)]
 e1r={};e4r={}
 with ep.open() as fp:
  for line in fp:
   x=json.loads(line);k=(x['task'],str(x['qid']))
   if k in selected and x.get('arm')=='P3-D1' and x.get('forced') is False:assert k not in e1r;e1r[k]=x
 with e4p.open() as fp:
  for line in fp:
   x=json.loads(line);k=(x['task'],str(x['qid']))
   if k in selected and x.get('budget')==256 and x.get('ds_on') is True:assert k not in e4r;e4r[k]=x
 assert set(e1r)==set(e4r)==set(selected)
 mapping={'processor':'processor','image_start':'image_start','ordered_ids':'ordered_ids','spatial_ids':'spatial_ids','visual_sha256':'main_visual','deepstack_full_sha256':'full_deepstack','prefill_embed_sha256':'prefill_embeds','attention_mask_sha256':'prefill_mask','prefill_position_ids':'prefill_position_ids','rope_delta':'adjusted_delta','injected_mask':'injected_mask','injected_deepstack':'injected_deepstack'}
 journal=[json.loads(l) for l in (run/'journal.jsonl').read_text().splitlines() if l.strip()];status=read(run/'captures/status.json');assert status['status']=='passed' and status['images_planned']==status['images_passed']==9 and len(status['rows'])==9 and not any(x.get('status') in ('image_failed','image_not_evaluated') for x in journal)
 rows=[]
 for i,target in enumerate(selection):
  k=(target['task'],str(target['qid']));a1=e1r[k];a4=e4r[k];assert a4['selection_mode']=='author_greedy' and a4['actual_injection_verified'] is True and all(a1[u]==a4[v] for u,v in mapping.items())
  assert all(str(a4[t])==str(target[t]) for t in ('task','qid','group_id','image_sha256','message_sha256')) and a1['group_id']==target['group_id'] and a1['message_sha256']==target['message_sha256']
  rp=run/'captures'/('image_%03d.json'%i);npth=run/'captures'/('image_%03d_norms.npy'%i);r=read(rp);n=np.load(npth,allow_pickle=False);got=features(n);assert r['status']=='passed' and r['index']==i and r['norms_path']==str(npth) and r['norms_sha256']==sha(npth) and r['hooks_clean'] is True and r['e1_identity']['status']=='passed' and all(str(r[t])==str(target[t]) for t in ('task','qid','group_id','image_sha256','message_sha256'))
  assert set(r['actual_identity'])==set(mapping)-{'image_start'} and all(r['actual_identity'][t]==a1[t] for t in r['actual_identity']);assert set(r['features'])==set(got)
  for t,v in got.items():assert (v is None and r['features'][t] is None) or (v is not None and r['features'][t] is not None and math.isclose(v,r['features'][t],rel_tol=1e-12,abs_tol=1e-12))
  ts=r['timing_seconds'];assert set(ts)=={'processor_preflight','capture_including_processor_visual_selector_and_one_token_generation','features_and_cpu_transfer','norms_write_and_fsync'} and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in ts.values())
  starts=[j for j,x in enumerate(journal) if x.get('status')=='image_started' and x.get('index')==i and (x.get('task'),str(x.get('qid')))==k]
  ends=[j for j,x in enumerate(journal) if x.get('status')=='image_completed' and x.get('index')==i and (x.get('task'),str(x.get('qid')))==k]
  assert len(starts)==len(ends)==1 and starts[0]<ends[0]
  events=journal[starts[0]:ends[0]+1];assert [x.get('status') for x in events]==['image_started','capture_generated','capture_cleanup','image_completed']
  assert (events[1].get('task'),str(events[1].get('qid')))==k and events[2]['passed'] is True and events[2]['capture_completed'] is True and events[3]['norms_sha256']==sha(npth)
  sr=status['rows'][i];assert (sr['task'],str(sr['qid']),sr['status'])==(k[0],k[1],'passed')
  rows.append({'task':k[0],'qid':k[1],'record_sha256':sha(rp),'norms_sha256':sha(npth),'feature_checks':8,'identity_checks':11})
 complete=read(run/'worker_complete.json');assert complete['status']=='completed' and complete['images']==9 and complete['weight_stats_before']==complete['weight_stats_after'] and all(type(complete[t]) in (int,float) and math.isfinite(complete[t]) and complete[t]>=0 for t in ('model_load_seconds','requests_seconds','cleanup_seconds'))
 out={'status':'passed','independent_review':True,'images':9,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'formal_sha256':sha(fm),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'source_manifest_sha256':sha(D/'run001_sources/manifest.json'),'reviewer_sha256':sha(__file__),'journal_sha256':sha(run/'journal.jsonl'),'actual_gpu_container':inspect,'rows':rows,'errors':[],'limitations':['Nine diagnostic captures and their elapsed phases are not TTFT or a representative performance benchmark.','Full visual tensors were not archived: saved norm vectors support offline feature recomputation while tensor association and identity are runtime capture evidence.','Accepted E1/E4 DEV mixed records were parsed for identity only; no ground truth or scoring.']}
 with open(a.out,'x') as fp:json.dump(out,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
