"""Independent complete sample and paired TTFT recomputation, no GT."""
import argparse,hashlib,json,os
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e4_c_budget_ttft_analysis_v4';R=B/'e4_c_budget_ttft_pilot_v3';A=B/'e4_c_budget_ttft_accept_v3';I=B/'e1_independent_review';OUT=I/'e4c_budget_ttft_v4_analysis_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script,out=None):
 s=read(folder/'started.json');t=read(folder/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve():raise ValueError('bad supervisor')
 if out and ('--out' not in cmd or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
def run():
 result=D/'analysis001/analysis.json';term(D/'monitor_analysis001',D/'analyze.py');term(I/'monitor_e4c_budget_ttft_v4_analysis_prelaunch',I/'review_e4c_budget_ttft_v4_analysis_prelaunch.py',I/'e4c_budget_ttft_v4_analysis_prelaunch_review.json');term(I/'monitor_e4c_budget_ttft_v3_result',I/'review_e4c_budget_ttft_v3_result.py',I/'e4c_budget_ttft_v3_result_review.json');term(I/'monitor_e4c_budget_ttft_v3_background',I/'review_e4c_budget_ttft_v3_background.py',I/'e4c_budget_ttft_v3_background_review.json')
 y=read(result);pre=read(I/'e4c_budget_ttft_v4_analysis_prelaunch_review.json');v=read(A/'verification001.json');bg=read(A/'background_overlap001.json');proof=read(I/'e4c_budget_ttft_v3_result_review.json');background=read(I/'e4c_budget_ttft_v3_background_review.json')
 if any(x.get('status')!='passed' or x.get('GT_opened') is not False or x.get('scoring_performed') is not False for x in (y,pre,v,bg,proof,background)):raise ValueError('upstream status')
 for k,p in {'analysis_plan_sha256':D/'analysis_plan001.json','formal_sha256':A/'verification001.json','review_sha256':I/'e4c_budget_ttft_v3_result_review.json','analysis_source_review_sha256':I/'e4c_budget_ttft_v4_analysis_prelaunch_review.json','analysis_supplement_sha256':D/'analysis_plan001_supplement.json','background_overlap_sha256':A/'background_overlap001.json','background_review_sha256':I/'e4c_budget_ttft_v3_background_review.json','source_journal_sha256':R/'run_001/journal.jsonl'}.items():
  if y.get(k)!=sha(p):raise ValueError('analysis evidence changed '+k)
 if pre.get('reviewer_sha256')!=sha(I/'review_e4c_budget_ttft_v4_analysis_prelaunch.py') or proof.get('reviewer_sha256')!=sha(I/'review_e4c_budget_ttft_v3_result.py') or background.get('reviewer_sha256')!=sha(I/'review_e4c_budget_ttft_v3_background.py'):raise ValueError('reviewer changed')
 for p,digest in pre['source_sha256'].items():
  if sha(p)!=digest:raise ValueError('analysis source changed')
 jobs=read(R/'pilot_plan001.json')['jobs'];events=[json.loads(x) for x in (R/'run_001/journal.jsonl').read_text().splitlines() if x.strip()]
 if len(jobs)!=48 or len(events)!=144 or len(y['samples'])!=48 or len(y['cells'])!=12:raise ValueError('sample denominator')
 expected=[]
 for n,j in enumerate(jobs):
  triple=events[n*3:n*3+3]
  if [q.get('event') for q in triple]!=['started','raw_generation','completed'] or any(q.get('job')!=j for q in triple):raise ValueError('job mismatch')
  payload=triple[2]['payload'];bound=payload['boundaries'];dt=bound['first_token_selected_ns']-bound['t0_before_image_read_ns']
  if dt<=0:raise ValueError('nonpositive')
  expected.append({'job':j,'ttft_ms':dt/1000000,'output_tokens':len(payload['output_ids']),'direct_health_overlap_indices':v['annotations'][n]['ttft_health_overlap_indices'],'direct_ps_query_overlap_indices':bg['annotations'][n]['ttft_ps_query_overlap'],'direct_ps_query_and_first_fsync_overlap_indices':bg['annotations'][n]['ttft_ps_first_fsync_overlap'],'ps_second_fsync_overlap':'unknown'})
 if expected!=y['samples'] or y['health_windows']!=v['health_windows'] or y['ps_windows']!=bg['watcher_windows']:raise ValueError('per-sample or window mismatch')
 measure=[x for x in expected if x['job']['phase']=='measure'];warm=[x for x in expected if x['job']['phase']=='warmup']
 if len(measure)!=36 or len(warm)!=12:raise ValueError('phase')
 cells=[]
 for task,qid in sorted({(s['job']['task'],s['job']['qid']) for s in measure}):
  for k in (128,256,512,1024):
   vals=[];ratios=[]
   for rep in range(3):
    def find(budget):
     matches=[s['ttft_ms'] for s in measure if s['job']['task']==task and s['job']['qid']==qid and s['job']['budget']==budget and s['job']['replicate']==rep]
     if len(matches)!=1:raise ValueError('duplicate or missing paired job')
     return matches[0]
    val=find(k);den=find(256);vals.append(val);ratios.append(val/den)
   cells.append({'task':task,'qid':qid,'budget':k,'n':3,'median_ttft_ms':sorted(vals)[1],'min_ttft_ms':min(vals),'max_ttft_ms':max(vals),'paired_ratios_to_k256':ratios,'median_paired_ratio_to_k256':sorted(ratios)[1]})
 if cells!=y['cells']:raise ValueError('cell or paired ratio mismatch')
 return {'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'samples':48,'warmups':12,'measured':36,'cells':12,'analysis_sha256':sha(result),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'reviewer_sha256':sha(__file__),'cells_recomputed':cells,'errors':[],'limitations':['Three known DEV anchors and three repeats per cell; descriptive exploratory result only.','K1024 is keep-all; all adapter overheads remain; no memory or population conclusion.','All measured samples retained. Direct health/ps windows shown; second ps fsync and other residual background unknown.','TTFT only; completion contains raw-output fsync.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('output')
 o=run()
 with a.out.open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
