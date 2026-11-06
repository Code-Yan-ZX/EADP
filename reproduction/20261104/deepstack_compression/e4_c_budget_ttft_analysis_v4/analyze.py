"""Prespecified descriptive 3-anchor TTFT analysis; no GT or scoring."""
import argparse,hashlib,json,os,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_budget_ttft_pilot_v3';ACCEPT=BASE/'e4_c_budget_ttft_accept_v3';REV=BASE/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervisor incomplete')
 return s
def admit_source():
 manifest=HERE/'source_manifest001.json';proof=REV/'e4c_budget_ttft_v4_analysis_prelaunch_review.json';reviewer=REV/'review_e4c_budget_ttft_v4_analysis_prelaunch.py'
 if not manifest.exists() or not proof.exists():raise ValueError('independent analysis source review required before reading timings')
 m=read(manifest);r=read(proof);cs=HERE/'monitor_cpu002/started.json';ct=HERE/'monitor_cpu002/terminal.json'
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or
     r.get('source_manifest_sha256')!=sha(manifest) or r.get('source_sha256')!=m['source_sha256'] or r.get('cpu_started_sha256')!=sha(cs) or
     r.get('cpu_terminal_sha256')!=sha(ct) or r.get('reviewer_sha256')!=sha(reviewer)):raise ValueError('analysis source review mismatch')
 if any(sha(path)!=digest for path,digest in m['source_sha256'].items()):raise ValueError('analysis source changed')
 terminal(cs,ct);s=terminal(REV/'monitor_e4c_budget_ttft_v4_analysis_prelaunch/started.json',REV/'monitor_e4c_budget_ttft_v4_analysis_prelaunch/terminal.json')
 cmd=s['command'];cwd=Path(s['cwd']);paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if paths.count(str(reviewer))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(proof):raise ValueError('analysis reviewer process mismatch')
 return sha(proof)
def admit():
 formal=ACCEPT/'verification001.json';review=REV/'e4c_budget_ttft_v3_result_review.json';reviewer=REV/'review_e4c_budget_ttft_v3_result.py'
 if not formal.exists() or not review.exists():raise ValueError('formal and independent timing proof required before reading timings')
 f=read(formal);r=read(review);fs=ACCEPT/'monitor_verify001/started.json';ft=ACCEPT/'monitor_verify001/terminal.json';rs=REV/'monitor_e4c_budget_ttft_v3_result/started.json';rt=REV/'monitor_e4c_budget_ttft_v3_result/terminal.json'
 if (f.get('status')!='passed' or f.get('jobs')!=48 or f.get('warmups')!=12 or f.get('measured')!=36 or f.get('errors')!=[] or
     f.get('GT_opened') is not False or f.get('scoring_performed') is not False or r.get('status')!='passed' or r.get('independent_review') is not True or
     r.get('jobs')!=48 or r.get('warmups')!=12 or r.get('measured')!=36 or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or
     r.get('formal_sha256')!=sha(formal) or r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('reviewer_sha256')!=sha(reviewer) or
     r.get('run_started_sha256')!=sha(RUN/'monitor_run001/started.json') or r.get('run_terminal_sha256')!=sha(RUN/'monitor_run001/terminal.json') or
     r.get('snapshot_manifest_sha256')!=sha(RUN/'run_001_sources/manifest.json') or r.get('health_sha256')!=sha(RUN/'monitor_run001/health.jsonl')):raise ValueError('timing acceptance mismatch')
 terminal(fs,ft);s=terminal(rs,rt);cmd=s['command'];cwd=Path(s['cwd']);paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if paths.count(str(reviewer))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(review):raise ValueError('wrong independent review process')
 evidence={'journal_sha256':RUN/'run_001/journal.jsonl','health_sha256':RUN/'monitor_run001/health.jsonl','snapshot_manifest_sha256':RUN/'run_001_sources/manifest.json',
           'plan_sha256':RUN/'runtime_plan001.json','worker_complete_sha256':RUN/'run_001/worker_complete.json','verifier_sha256':ACCEPT/'verify.py'}
 if any(f.get(k)!=sha(path) for k,path in evidence.items()):raise ValueError('accepted timing evidence changed')
 bg=ACCEPT/'background_overlap001.json';bs=ACCEPT/'monitor_background001/started.json';bt=ACCEPT/'monitor_background001/terminal.json';terminal(bs,bt)
 br=REV/'e4c_budget_ttft_v3_background_review.json';brs=REV/'monitor_e4c_budget_ttft_v3_background/started.json';brt=REV/'monitor_e4c_budget_ttft_v3_background/terminal.json';brcode=REV/'review_e4c_budget_ttft_v3_background.py'
 if not br.exists():raise ValueError('background independent proof missing')
 proof=read(br)
 if (proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('jobs')!=48 or proof.get('errors')!=[] or
     proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or proof.get('background_overlap_sha256')!=sha(bg) or
     proof.get('background_started_sha256')!=sha(bs) or proof.get('background_terminal_sha256')!=sha(bt) or proof.get('main_review_sha256')!=sha(review) or proof.get('reviewer_sha256')!=sha(brcode)):raise ValueError('background independent proof mismatch')
 bsrun=terminal(brs,brt);cmd=bsrun['command'];cwd=Path(bsrun['cwd']);paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if paths.count(str(brcode))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(br):raise ValueError('background reviewer process mismatch')
 b=read(bg)
 if b.get('status')!='passed' or b.get('jobs')!=48 or b.get('formal_sha256')!=sha(formal) or b.get('journal_sha256')!=f['journal_sha256'] or b.get('errors')!=[]:raise ValueError('background overlap proof mismatch')
 return f,r,b,sha(br)
def summarize(jobs,events,annotations,background):
 if len(jobs)!=48 or len(events)!=144 or len(annotations)!=48 or len(background)!=48:raise ValueError('incomplete job evidence')
 samples=[]
 for i,job in enumerate(jobs):
  triple=events[i*3:i*3+3]
  if [x.get('event') for x in triple]!=['started','raw_generation','completed'] or any(x.get('job')!=job for x in triple) or annotations[i].get('order')!=i or background[i].get('order')!=i:raise ValueError('timing event/order mismatch')
  result=triple[2]['payload'];b=result['boundaries'];start=b['t0_before_image_read_ns'];first=b['first_token_selected_ns']
  if type(start) is not int or type(first) is not int or first<=start:raise ValueError('nonpositive TTFT')
  samples.append({'job':job,'ttft_ms':(first-start)/1e6,'output_tokens':len(result['output_ids']),'direct_health_overlap_indices':annotations[i]['ttft_health_overlap_indices'],
                  'direct_ps_query_overlap_indices':background[i]['ttft_ps_query_overlap'],'direct_ps_query_and_first_fsync_overlap_indices':background[i]['ttft_ps_first_fsync_overlap'],'ps_second_fsync_overlap':'unknown'})
 measured=[s for s in samples if s['job']['phase']=='measure'];warm=[s for s in samples if s['job']['phase']=='warmup']
 if len(measured)!=36 or len(warm)!=12:raise ValueError('timing phase denominator')
 bykey={(s['job']['task'],s['job']['qid'],s['job']['replicate'],s['job']['budget']):s for s in measured}
 if len(bykey)!=36:raise ValueError('duplicate measured job')
 cells=[]
 for task,qid in sorted({(s['job']['task'],s['job']['qid']) for s in measured}):
  for k in (128,256,512,1024):
   vals=[bykey[(task,qid,i,k)]['ttft_ms'] for i in range(3)]
   ratios=[bykey[(task,qid,i,k)]['ttft_ms']/bykey[(task,qid,i,256)]['ttft_ms'] for i in range(3)]
   cells.append({'task':task,'qid':qid,'budget':k,'n':3,'median_ttft_ms':statistics.median(vals),'min_ttft_ms':min(vals),'max_ttft_ms':max(vals),'paired_ratios_to_k256':ratios,'median_paired_ratio_to_k256':statistics.median(ratios)})
 return {'samples':samples,'cells':cells}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args();out=args.out_dir.resolve()
 plan=read(HERE/'analysis_plan001.json');supp=read(HERE/'analysis_plan001_supplement.json')
 if plan.get('pilot_plan_sha256')!=sha(RUN/'pilot_plan001.json') or plan.get('runtime_plan_sha256')!=sha(RUN/'runtime_plan001.json'):raise ValueError('analysis plan/source mismatch')
 if supp.get('original_plan_sha256')!=sha(HERE/'analysis_plan001.json') or supp.get('background_overlap_sha256')!=sha(ACCEPT/'background_overlap001.json'):raise ValueError('background supplement mismatch')
 source_review=admit_source();f,r,b,bg_review=admit() # All gates precede performance journal access.
 jobs=read(RUN/'pilot_plan001.json')['jobs'];events=[json.loads(x) for x in (RUN/'run_001/journal.jsonl').read_text().splitlines() if x.strip()]
 result=summarize(jobs,events,f['annotations'],b['annotations']);out.mkdir(exist_ok=False)
 result.update({'status':'passed','GT_opened':False,'scoring_performed':False,'analysis_plan_sha256':sha(HERE/'analysis_plan001.json'),
  'formal_sha256':sha(ACCEPT/'verification001.json'),'review_sha256':sha(REV/'e4c_budget_ttft_v3_result_review.json'),'analysis_source_review_sha256':source_review,
  'analysis_supplement_sha256':sha(HERE/'analysis_plan001_supplement.json'),'background_overlap_sha256':sha(ACCEPT/'background_overlap001.json'),'background_review_sha256':bg_review,
  'source_journal_sha256':f['journal_sha256'],'health_windows':f['health_windows'],'ps_windows':b['watcher_windows'],'limits':supp['limits'],'scope':plan['report']})
 with (out/'analysis.json').open('x') as file:json.dump(result,file,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
if __name__=='__main__':main()
