"""Independent rank correlation / pair-count AUC of accepted K and features."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixel_k_feature_diagnostic'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def terminal(m,script,out):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def rank(v):
 order=sorted(range(len(v)),key=lambda i:v[i]);out=[0.]*len(v);i=0
 while i<len(v):
  j=i+1
  while j<len(v) and v[order[j]]==v[order[i]]:j+=1
  mean=(i+1+j)/2
  for z in order[i:j]:out[z]=mean
  i=j
 return out
def rho(a,b):
 ra=rank(a);rb=rank(b);ma=sum(ra)/len(ra);mb=sum(rb)/len(rb);top=sum((x-ma)*(y-mb) for x,y in zip(ra,rb));da=sum((x-ma)**2 for x in ra);db=sum((y-mb)**2 for y in rb)
 return top/math.sqrt(da*db) if da and db else None
def close(a,b):return a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12)
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_k_feature_diagnostic_review.json';plan=read(D/'plan001.json');out=read(D/'analysis001.json');fp=B/plan['features'];sp=B/plan['pixel_scores'];frp=B/plan['feature_review'];srp=B/plan['pixel_review'];fr=read(frp);sr=read(srp)
terminal(D/'monitor_analysis001',D/'analyze.py',D/'analysis001.json');terminal(R/'monitor_e4_adaptive_features003',R/'review_e4_adaptive_features.py',frp);terminal(R/'monitor_pixel_t045_dev384_scores002',R/'review_pixel_t045_dev384_scores.py',srp)
assert all(x['status']=='passed' and x['independent_review'] is True and x['questions']==384 for x in (fr,sr)) and fr['reviewer_sha256']==sha(R/'review_e4_adaptive_features.py') and sr['reviewer_sha256']==sha(R/'review_pixel_t045_dev384_scores.py') and fr['features_sha256']==sha(fp) and sr['score_sha256']==sha(sp)
assert out['status']=='completed' and out['plan_sha256']==sha(D/'plan001.json') and out['features_sha256']==sha(fp) and out['pixel_scores_sha256']==sha(sp) and out['GT_opened'] is False and out['new_predictions_opened'] is False and out['existing_mixed_score_prediction_records_parsed_for_identity_and_k'] is True and out['limitations']==plan['limitations']
features=lines(fp);scores=lines(sp);f={key(x):x for x in features};s={key(x):x for x in scores};assert len(features)==len(scores)==len(f)==len(s)==384 and set(f)==set(s) and collections.Counter(k[0] for k in f)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128};counts={}
for task in plan['tasks']:
 keys=[key(x) for x in features if x['task']==task];assert len(keys)==128 and len({x[2] for x in keys})==128;ks=[s[k]['selected_count'] for k in keys];assert all(type(x) is int and 1<=x<=1024 for x in ks);positive=[i for i,x in enumerate(ks) if x<=64];negative=[i for i,x in enumerate(ks) if x>64];counts[task]=len(positive);actual=out['tasks'][task];assert actual['questions']==128 and actual['k_le64']==len(positive)
 for name in plan['features_list']:
  vals=[f[k][name] for k in keys];assert all(type(x) in (int,float) and math.isfinite(x) for x in vals);rr=rho(vals,ks);auc=(sum((vals[i]>vals[j])+0.5*(vals[i]==vals[j]) for i in positive for j in negative)/(len(positive)*len(negative))) if len(positive)>=2 and len(negative)>=2 else None;z=actual['statistics'][name];assert close(z['spearman_with_k'],rr) and close(z['higher_feature_auc_k_le64'],auc)
proof={'status':'passed','independent_review':True,'questions':384,'task_positive_counts':counts,'analysis_sha256':sha(D/'analysis001.json'),'plan_sha256':sha(D/'plan001.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'feature_review_sha256':sha(frp),'pixel_review_sha256':sha(srp),'reviewer_sha256':sha(__file__),'GT_opened':False,'new_predictions_opened':False,'existing_mixed_score_prediction_records_parsed_for_identity_and_k':True,'limitations':plan['limitations'],'errors':[]}
with a.out.open('x') as h:json.dump(proof,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
print('passed')
