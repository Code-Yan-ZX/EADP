"""After source freeze, route already scored DEV rows; no inference or new GT."""
import argparse,collections,hashlib,importlib.util,json,math,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('invalid supervisor')
def prior(tag):
 if tag=='t020':A=B/'e6/pixelprune_t020_dev384_analysis';proof=R/'pixel_t020_dev384_scores_review.json';review=R/'review_pixel_t020_dev384_scores.py';monitor=R/'monitor_pixel_t020_dev384_scores';expected=768
 else:A=B/'e6/pixelprune_dev384_analysis';proof=R/'pixel_dev384_scores_review.json';review=R/'review_pixel_dev384_scores.py';monitor=R/'monitor_pixel_dev384_scores';expected=384
 term(A/'monitor_score001',A/'score.py');term(monitor,review);x=read(proof)
 if x['status']!='passed' or x['independent_review'] is not True or x['questions']!=384 or x['pixel_scores_recomputed']!=expected or x['errors']!=[] or x['reviewer_sha256']!=sha(review) or x['score_started_sha256']!=sha(A/'monitor_score001/started.json') or x['score_terminal_sha256']!=sha(A/'monitor_score001/terminal.json') or x['score_sha256']!=sha(A/'score001/scores.jsonl') or x['analysis_sha256']!=sha(A/'score001/analysis.json') or x['frozen_sha256']!=sha(A/'score001/frozen.json'):raise ValueError('prior proof mismatch')
 return A/'score001/scores.jsonl'
def parse(path):
 rows=[json.loads(s) for s in path.read_text().splitlines()];out={(r['task'],str(r['qid']),r['group_id']):r for r in rows}
 if len(rows)!=384 or len(out)!=384:raise ValueError('source inventory')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();freeze=read(D/'source_freeze001.json');
 for p,h in freeze['source_sha256'].items():
  if sha(p)!=h:raise ValueError('frozen source or upstream changed: '+p)
 plan=read(D/'plan001.json');
 if plan['thresholds']!=[128,256,512] or plan['bootstrap']['group_order']!='ascending group_id':raise ValueError('plan mismatch')
 spec=importlib.util.spec_from_file_location('eadp_fallback_core',D/'core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 p020=prior('t020');p0=prior('t0');new=parse(p020);old=parse(p0)
 if set(new)!=set(old):raise ValueError('source pair mismatch')
 rows=[]
 for key in sorted(new):
  n=new[key];o=old[key];a=n['arms']['pixel_t0'];b=o['arms']['pixel_dynamic'];ea=n['arms']['e4_k256_d1'];eb=o['arms']['e4_k256_d1']
  for x,y in ((a,b),(ea,eb)):
   if x['answer']!=y['answer'] or x['eos']!=y['eos'] or abs(x['score']-y['score'])>1e-12:raise ValueError('shared accepted reference mismatch')
  for x in (n['arms']['pixel_t020'],a,ea):
   if not math.isfinite(x['score']) or not math.isfinite(x['independent_score']) or abs(x['score']-x['independent_score'])>1e-12:raise ValueError('source scoring mismatch')
  rows.append({'task':key[0],'qid':key[1],'group_id':key[2],'k020':n['selected_count'],'k0':o['selected_count'],'s020':n['arms']['pixel_t020']['score'],'s0':a['score'],'e4':ea['score']})
 if collections.Counter(x['task'] for x in rows)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('task denominator')
 reports={task:core.calculate([r for r in rows if r['task']==task]) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')}
 out=args.out;out.mkdir(exist_ok=False)
 with (out/'routes.jsonl').open('x') as f:
  for r in rows:
   d={**r,'routes':{str(t):{'source':'t0' if r['k020']<=t else 't020','selected_k':r['k0'] if r['k020']<=t else r['k020'],'score':r['s0'] if r['k020']<=t else r['s020']} for t in core.THRESHOLDS}};f.write(json.dumps(d,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 result={'status':'completed','questions':384,'tasks':reports,'plan_sha256':sha(D/'plan001.json'),'source_freeze_sha256':sha(D/'source_freeze001.json'),'source_scores_sha256':{'t020':sha(p020),'t0':sha(p0)},'routes_sha256':sha(out/'routes.jsonl'),'GT_opened':False,'generation_performed':False,'score_files_contain_predictions':True,'limitations':plan['limitations']}
 with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
