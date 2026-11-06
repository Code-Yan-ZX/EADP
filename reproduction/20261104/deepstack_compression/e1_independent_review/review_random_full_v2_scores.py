"""Independent DEV-only random-full score and image-group bootstrap review."""
import argparse,ast,csv,hashlib,importlib.util,io,json,math,os,re,string
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e6/random_full_analysis_v3';P=B/'e6/random_full_exec_v2';SEEDS=(20261104,20261105,20261106);TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');ARMS=('eadp',*[f'random_{s}' for s in SEEDS],'random_three_seed_mean')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def term(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('missing completed supervisor '+str(d))
def distance(a,b):
 prev=list(range(len(b)+1))
 for x in a:
  cur=[prev[0]+1]
  for j,y in enumerate(b):cur.append(min(cur[-1]+1,prev[j+1]+1,prev[j]+(x!=y)))
  prev=cur
 return prev[-1]
def doc(pred,gold):
 out=[]
 for truth in gold:
  n=max(len(truth.upper()),len(pred.upper()));sim=1-distance(' '.join(truth.lower().split()),' '.join(pred.lower().split()))/n if n else 1
  out.append(sim if sim>=.5 else 0.)
 return max(out)
def text_score(pred,answers,normal):
 x=[normal(a) for a in answers];p=normal(pred)
 return sum(min(1,sum(a==p for j,a in enumerate(x) if i!=j)/3) for i in range(len(x)))/len(x)
def option(pred,choices):
 if 'Failed to obtain answer via API' not in pred:
  if any(x in pred for x in ["Sorry, I can't help with images of people yet.","I can't process this file.","I'm sorry, but without the image provided",'Cannot determine the answer']):return 'Z'
  w=re.sub(r'[.()\[\],:;!*#{}]',' ',pred).split();found=[k for k in choices if k in w]
  if len(found)==1 and w.index(found[0])>len(w)-5:return found[0]
  if not found and 'Z' in w:return 'Z'
 if len(pred)>2*sum(len(str(x)) for x in choices.values()):return 'Z'
 found=[k for k,val in choices.items() if str(val).lower() in pred.lower()]
 return found[0] if len(found)==1 else 'Z'
def truths(sources,needed,native_na):
 csv.field_size_limit(100_000_000);out={}
 for task,desc in sources.items():
  if sha(desc['path'])!=desc['sha256']:raise ValueError('dataset hash')
  wanted={q.encode() for t,q in needed if t==task}
  with open(desc['path'],'rb') as file:
   head=next(csv.reader([file.readline().decode()],delimiter='\t'));col=head.index('index')
   if col not in (0,len(head)-1):raise ValueError('index location')
   for raw in file:
    q=(raw.split(b'\t',1)[0] if col==0 else raw.rstrip(b'\r\n').rsplit(b'\t',1)[-1])
    if len(q)>=2 and q[:1]==q[-1:]==b'"':q=q[1:-1]
    if q not in wanted:continue
    fields=next(csv.reader([raw.decode()],delimiter='\t'))
    if len(fields)!=len(head):raise ValueError('selected TSV row width')
    row=dict(zip(head,fields));key=(task,str(row['index']))
    if key in out or key not in needed:raise ValueError('selected qid duplicate')
    if task=='AI2D_TEST':out[key]={'answer':row['answer'],'choices':{k:row[k] for k in string.ascii_uppercase if k in row and row[k] not in native_na}}
    else:
     try:value=ast.literal_eval(row['answer'])
     except (ValueError,SyntaxError):value=row['answer']
     out[key]={'answers':[str(x) for x in value] if isinstance(value,list) else [str(value)]}
 if set(out)!=needed:raise ValueError('missing DEV truths')
 return out

def bootstrap(rows):
 groups=defaultdict(list)
 for x in rows:
  values=[x['arms']['eadp']['score']]+[x['arms'][str(s)]['score'] for s in SEEDS];groups[x['group_id']].append(values+[sum(values[1:])/3])
 ids=sorted(groups);totals=np.asarray([np.asarray(groups[g]).sum(axis=0) for g in ids]);sizes=np.array([len(groups[g]) for g in ids]);mean=100*totals.sum(axis=0)/sizes.sum();draw=np.random.default_rng(20261104).integers(0,len(ids),(10000,len(ids)));boot=100*totals[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
 return ids,mean,boot

def main(out):
 if np.__version__!='2.2.6':raise ValueError('NumPy version')
 dest=D/'score001';term(D/'monitor_score001',D/'score.py',dest);term(P/'monitor_verify001',P/'verify.py',P/'verification001.json');term(I/'monitor_random_full_v2_result',I/'review_random_full_v2_result.py',I/'random_full_v2_result_review.json');term(I/'monitor_random_full_v2_score_prelaunch',I/'review_random_full_v2_score_prelaunch.py',I/'random_full_v2_score_prelaunch_review.json')
 result=read(dest/'analysis.json');frozen=read(dest/'frozen.json');proof=read(I/'random_full_v2_result_review.json');pre=read(I/'random_full_v2_score_prelaunch_review.json');plan=read(D/'score_plan001.json');scorespath=dest/'scores.jsonl'
 if result.get('status')!='completed' or result.get('questions')!=384 or result.get('random_generations')!=1152 or result.get('scores_sha256')!=sha(scorespath) or result.get('plan_sha256')!=sha(D/'score_plan001.json') or pre.get('status')!='passed' or pre.get('score_plan_sha256')!=sha(D/'score_plan001.json') or pre.get('reviewer_sha256')!=sha(I/'review_random_full_v2_score_prelaunch.py') or proof.get('status')!='passed' or proof.get('independent_review') is not True or (proof.get('questions'),proof.get('generations'))!=(384,1152) or proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or proof.get('reviewer_sha256')!=sha(I/'review_random_full_v2_result.py') or proof.get('formal_sha256')!=sha(P/'verification001.json'):raise ValueError('score/result gates')
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()) or frozen['admission']['plan_sha256']!=sha(D/'score_plan001.json') or frozen['admission']['independent_sha256']!=sha(I/'random_full_v2_result_review.json'):raise ValueError('scoring sources changed')
 files=proof['files_sha256']
 if any(sha(k)!=v for k,v in files.items()):raise ValueError('accepted generation changed')
 spec=importlib.util.spec_from_file_location('_independent_author_metrics',B/'e1_analysis/metrics.py');met=importlib.util.module_from_spec(spec);spec.loader.exec_module(met);author=met.author_functions();normal=author['process_answer'];constants=met._author_constants()
 anchors=read(B/'e6/pixelprune_t045_dev384/plan001.json')['anchors'];needed={(x['task'],str(x['qid'])) for x in anchors}
 if len(anchors)!=384 or len(needed)!=384 or Counter(x['task'] for x in anchors)!={t:128 for t in TASKS}:raise ValueError('DEV cohort')
 refs=frozen['references'];gold=truths(refs,needed,met.NATIVE_NA);records=lines(P/'run_001/records.jsonl');scored=lines(scorespath);old=lines(B/'e1/e1full001_score/scores.jsonl');e1raw=lines(B/'e1/e1full001/raw_predictions.jsonl')
 key=lambda r:(r['task'],str(r['qid']),r['group_id']);random={(key(r),r['seed']):r for r in records};before={key(r):r for r in old};ref={(key(r)):r for r in e1raw if r['arm']=='P3-D1' and r['forced'] is False};resultby={key(r):r for r in scored};am={key(r):r for r in anchors}
 if len(records)!=1152 or len(random)!=1152 or len(resultby)!=384 or set(resultby)!=set(am) or set(before)!=set(am) or set(ref)!=set(am) or set(random)!={(k,s) for k in am for s in SEEDS}:raise ValueError('score pair inventory')
 native={}
 for task in TASKS:
  for seed in SEEDS:
   sub=[r for r in records if r['task']==task and r['seed']==seed];f=io.BytesIO();pd.DataFrame({'index':range(len(sub)),'prediction':[r['answer'] for r in sub]}).to_excel(f,index=False,engine='xlsxwriter');f.seek(0);tab=pd.read_excel(f)
   if len(sub)!=128 or len(tab)!=128 or list(tab['index'])!=list(range(128)):raise ValueError('XLSX denominator')
   for r,value in zip(sub,tab['prediction']):native[(key(r),seed)]=str(value)
 for k,a in am.items():
  item=resultby[k];e=ref[k];old_arm=before[k]['arms']['P3-D1']
  if item['arms']['eadp']!=old_arm or e['answer']!=old_arm['raw_answer'] or e['eos']!=old_arm['eos']:raise ValueError('E1 score reference')
  truth=gold[(k[0],k[1])]
  for seed in SEEDS:
   r=random[(k,seed)];sc=item['arms'][str(seed)];pred=native[(k,seed)]
   if sc['answer']!=r['answer'] or sc['eos']!=r['eos'] or sc['truncated']!=r['truncated'] or r['processor']!=a['processor'] or r['generation_scalars']!=e['generation_scalars'] or r['eos_ids']!=e['eos_ids'] or r['message_sha256']!=a['message_sha256'] or r['image_sha256']!=a['image_sha256']:raise ValueError('random score input')
   if k[0]=='AI2D_TEST':official=float((author['can_infer'](pred,dict(truth['choices'])) or 'Z')==truth['answer']);independent=float(option(pred,truth['choices'])==truth['answer'])
   else:
    method='anls' if k[0]=='DocVQA_VAL' else 'vqa_score';x=author['process_line']({'answer':repr(truth['answers']),'prediction':pred},method=method);official=float(author['hit_calculate']([x],k[0])[0]);independent=doc(pred,truth['answers']) if method=='anls' else text_score(pred,truth['answers'],normal)
   if not math.isfinite(official) or not math.isfinite(independent) or abs(official-independent)>1e-12 or abs(official-sc['score'])>1e-12 or abs(independent-sc['independent_score'])>1e-12:raise ValueError('per-question score mismatch')
 for task in TASKS:
  rows=[x for x in scored if x['task']==task];ids,mean,boot=bootstrap(rows);report=result['tasks'][task]
  if len(rows)!=128 or report['questions']!=128 or report['image_groups']!=len(ids) or report['seed_values']!=list(SEEDS) or report['bootstrap']!={'unit':'image_group','group_order':'ascending','weighting':'question','replicates':10000,'seed':20261104,'numpy_version':'2.2.6'}:raise ValueError('bootstrap inventory')
  for idx,name in enumerate(ARMS):
   if not math.isclose(report['arm_mean_pct'][name],float(mean[idx]),abs_tol=1e-10,rel_tol=0) or not np.allclose(report['arm_ci95_pct'][name],np.quantile(boot[:,idx],[.025,.975]),atol=1e-10,rtol=0):raise ValueError('arm mean/CI')
   if idx:
    x=report['contrasts_vs_eadp'][name]
    if not math.isclose(x['difference_pp'],float(mean[idx]-mean[0]),abs_tol=1e-10,rel_tol=0) or not np.allclose(x['ci95_pp'],np.quantile(boot[:,idx]-boot[:,0],[.025,.975]),atol=1e-10,rtol=0):raise ValueError('paired contrast/CI')
  for seed in SEEDS:
   expected={'eos':sum(x['arms'][str(seed)]['eos'] for x in rows),'truncated':sum(x['arms'][str(seed)]['truncated'] for x in rows),'empty':sum(not x['arms'][str(seed)]['answer'].strip() for x in rows)}
   if report['terminal'][str(seed)]!=expected:raise ValueError('terminal counts')
 if any(sha(p)!=h for p,h in plan['source_sha256'].items()) or any(sha(x['path'])!=x['sha256'] for x in refs.values()):raise ValueError('sources changed during review')
 proofout={'status':'passed','independent_review':True,'questions':384,'random_generations':1152,'reviewed_scores':1152,'reference_questions':384,'GT_opened':True,'scoring_performed':True,'errors':[],'score_sha256':sha(scorespath),'analysis_sha256':sha(dest/'analysis.json'),'score_started_sha256':sha(D/'monitor_score001/started.json'),'score_terminal_sha256':sha(D/'monitor_score001/terminal.json'),'random_result_review_sha256':sha(I/'random_full_v2_result_review.json'),'score_plan_sha256':sha(D/'score_plan001.json'),'reviewer_sha256':sha(__file__),'limits':['Posthoc exploratory DEV; intervals uncorrected.','Per-question 3-seed mean; image-group intervals conditional on these seeds.','Task×seed XLSX processed separately.','Discarded author greedy makes timing incomparable.']}
 with out.open('x') as f:json.dump(proofout,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=I/'random_full_v2_scores_review.json' or a.out.exists():raise ValueError('fixed output required')
 main(a.out)
