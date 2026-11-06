"""Score every fixed unsupervised rank and random budget rule on stored DEV scores."""
import argparse,hashlib,json,math
from pathlib import Path

B=Path(__file__).resolve().parent.parent;D=B/'e4_adaptive_features'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def key(r):return (r['task'],str(r['qid']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json')
 done=read(D/'features001/extraction.json');start=read(D/'monitor_extract001/started.json');term=read(D/'monitor_extract001/terminal.json')
 if (term['reason']!='completed' or term['worker_returncode']!=0 or term['remaining_process_group']!=[] or term['worker_pid']!=start['worker_pid']
     or done['status']!='completed' or done['n']!=384 or done['plan_sha256']!=sha(D/'plan001.json') or done['extractor_sha256']!=sha(D/'extract.py')
     or done['score_or_prediction_opened'] is not False or done['GT_opened'] is not False or done['feature_sha256']!=sha(D/'features001/features.jsonl')):raise ValueError('feature gate failed')
 if sha(B/plan['inputs'])!=plan['inputs_sha256']:raise ValueError('input identity')
 for p,h in plan['score_proofs'].items():
  if sha(B/p)!=h or read(B/p).get('status')!='passed':raise ValueError('score proof drift')
 source=B/plan['scores']
 if sha(source)!=plan['scores_sha256']:raise ValueError('score identity')
 f=lines(D/'features001/features.jsonl');raw=lines(B/plan['inputs']);score=lines(source)
 features={key(x):x for x in f};inputs={key(x):x for x in raw};scores={key(x):x for x in score}
 if len(f)!=384 or len(raw)!=384 or len(score)!=384 or len(features)!=384 or set(features)!=set(inputs) or set(features)!=set(scores):raise ValueError('pair inventory')
 for k,x in features.items():
  if (x['group_id']!=inputs[k]['group_id'] or x['group_id']!=scores[k]['group_id'] or x['image_sha256']!=inputs[k]['image_sha256']):raise ValueError('pair identity')
  for q in (128,256,512):
   arm=scores[k]['arms'][f'K{q}-D1'];v=arm['score'];iv=arm['independent_score']
   if not math.isfinite(v) or not (0<=v<=1) or abs(v-iv)>1e-12:raise ValueError('score invalid')
 reports={};assignments=[]
 for task in plan['tasks']:
  taskrows=[x for x in f if x['task']==task]
  if len(taskrows)!=128 or len({x['group_id'] for x in taskrows})!=128:raise ValueError('task inventory')
  rules={}
  for name in ('gray_entropy16','gray_gradient','user_text_chars'):
   order=sorted(taskrows,key=lambda x:(-x[name],digest('20261105|'+x['group_id'])))
   rules[name]={x['group_id'] for x in order[:42]}
  for seed in range(3):
   order=sorted(taskrows,key=lambda x:digest(f"20261105|random|{seed}|"+x['group_id']))
   rules[f'random_{seed}']={x['group_id'] for x in order[:42]}
  baseline=sum(scores[key(x)]['arms']['K256-D1']['score'] for x in taskrows)/128
  taskreport={}
  for name,high in rules.items():
   if len(high)!=42:raise ValueError('budget count')
   values=[];improved=declined=unchanged=0
   for x in taskrows:
    s=scores[key(x)]['arms'];k=512 if x['group_id'] in high else 128
    val=s[f'K{k}-D1']['score'];base=s['K256-D1']['score'];values.append(val)
    improved+=val>base;declined+=val<base;unchanged+=val==base
    assignments.append({'task':task,'qid':x['qid'],'group_id':x['group_id'],'rule':name,'selected_k':k,'score':val,'baseline_score':base})
   mean=sum(values)/128
   taskreport[name]={'mean_pct':100*mean,'minus_fixed256_pp':100*(mean-baseline),'improved':improved,'declined':declined,'unchanged':unchanged,'high_k_count':42,'mean_k':254}
  reports[task]={'n':128,'fixed_k256_mean_pct':100*baseline,'rules':taskreport}
 a.out.mkdir(exist_ok=False)
 with (a.out/'assignments.jsonl').open('x') as out:
  for r in assignments:out.write(json.dumps(r,sort_keys=True)+'\n')
 result={'status':'completed','plan_sha256':sha(D/'plan001.json'),'feature_sha256':sha(D/'features001/features.jsonl'),'scores_sha256':sha(source),'assignments_sha256':sha(a.out/'assignments.jsonl'),'tasks':reports,'GT_opened':False,'new_predictions_opened':False,'limitations':plan['limitations']}
 with (a.out/'analysis.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
 print('completed')
if __name__=='__main__':main()
