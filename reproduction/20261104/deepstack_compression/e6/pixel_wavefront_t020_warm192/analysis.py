"""Full-pair exploratory timing analysis; accepted results and reviews required."""
import argparse,hashlib,json,os
from pathlib import Path
import stats
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
def admit():
 plan=read(D/'analysis_plan001.json');source=plan['source_sha256'];review=I/'pixel_t020_warm192_analysis_prelaunch_review.json';reviewer=I/'review_pixel_t020_warm192_analysis_prelaunch.py';monitor=I/'monitor_pixel_t020_warm192_analysis_prelaunch';terminal(monitor,reviewer,review);q=read(review)
 if plan.get('scope')!='pixel_t020_warm192_paired_analysis' or plan.get('GPU_used') is not False or plan.get('GT_opened') is not False or any(sha(k)!=v for k,v in source.items()) or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GT_opened') is not False or q.get('timing_results_opened') is not False or q.get('errors')!=[] or q.get('plan_sha256')!=sha(D/'analysis_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('analysis prelaunch proof')
 formal=D/'verification001.json';proof=I/'pixel_t020_warm192_result_review.json';pr=I/'review_pixel_t020_warm192_result.py';pm=I/'monitor_pixel_t020_warm192_result';terminal(D/'monitor_verify001',D/'formal.py',formal);terminal(pm,pr,proof);r=read(proof)
 fields={'formal_sha256':formal,'formal_started_sha256':D/'monitor_verify001/started.json','formal_terminal_sha256':D/'monitor_verify001/terminal.json','run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','snapshot_manifest_sha256':D/'run_001_sources/manifest.json','reviewer_sha256':pr}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('processes')!=8 or r.get('measured')!=384 or r.get('warmups')!=8 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('independent accepted result')
 f=read(formal)
 if f.get('status')!='passed' or f.get('processes')!=8 or f.get('measured')!=384 or f.get('warmups')!=8 or f.get('GT_opened') is not False or f.get('errors')!=[] or any(sha(k)!=v for k,v in f['files_sha256'].items()):raise ValueError('formal result or files changed')
 if plan.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json') or plan.get('formal_plan_sha256')!=sha(D/'formal_plan001.json') or f.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json'):raise ValueError('plan binding')
 return plan,f,r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'analysis001.json').resolve():raise ValueError('analysis output')
 plan,formal,review=admit();design=read(D/'design001.json');records=[x['record'] for x in formal['records'] if x['phase']=='measure'];value=stats.calculate(design,records,draws=10000,seed=20261105)
 monitoring={(x['record']['task'],str(x['record']['qid']),x['record']['arm']):{'process_order':x['process_order'],'block':x['block'],'monitoring':x['monitoring']} for x in formal['records'] if x['phase']=='measure'}
 if len(monitoring)!=384:raise ValueError('monitoring inventory')
 for row in value['per_question']:
  key=(row['task'],row['qid']);row['author_observation']=monitoring[(*key,'author')];row['wavefront_observation']=monitoring[(*key,'wavefront')]
 value.update({'GT_opened':False,'scoring_performed':False,'analysis_plan_sha256':sha(D/'analysis_plan001.json'),'formal_sha256':sha(D/'verification001.json'),'independent_result_sha256':sha(I/'pixel_t020_warm192_result_review.json'),'limitations':['post-hoc DEV sample; 3 tasks, single A40 and one paired measurement per image','max_new_tokens=1 differs from historical natural 2048-token setting; full generation/KV equivalence not established','median paired difference is not difference between marginal medians; bootstrap is uncorrected exploratory','prefill KV cache/position and selector observations are in TTFT and identical in both arms','supervisor/observer query windows retained per observation, manual/unsampled background and second fsync remain unknown','model loading excluded; input hashes and warmup establish warm file/model process, not universal steady state']})
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
