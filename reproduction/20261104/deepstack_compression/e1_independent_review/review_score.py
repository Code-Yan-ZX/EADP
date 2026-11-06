"""Independent 384-question DEV-only re-score and grouped-bootstrap audit."""
import ast,csv,hashlib,io,json,math,os,re,string,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,pandas as pd
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');E=B/'e1';OUT=B/'e1_independent_review';A=E/'e1full001_score';S=E/'e1full001_score_sources';sys.path.insert(0,str(S));import metrics
ARMS=('P1-D0','P1-D1','P3-D0','P3-D1');TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
CONTRASTS={'DS_at_P1':[-1,1,0,0],'DS_at_P3':[0,0,-1,1],'P3_minus_P1_at_D0':[-1,0,1,0],'P3_minus_P1_at_D1':[0,-1,0,1],'interaction_DS_by_position':[1,-1,-1,1]}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def ls(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('partial journal')
 return [json.loads(x) for x in b.splitlines()]
def distance(a,b):
 row=list(range(len(b)+1))
 for x in a:
  nxt=[row[0]+1]
  for j,y in enumerate(b):nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(x!=y)))
  row=nxt
 return row[-1]
def doc(pred,answers):
 values=[]
 for gt in answers:
  n=max(len(gt.upper()),len(pred.upper()));v=1-distance(' '.join(gt.lower().split()),' '.join(pred.lower().split()))/n if n else 1
  values.append(v if v>=.5 else 0)
 return max(values)
def text_score(pred,answers,normal):
 gold=[normal(x) for x in answers];p=normal(pred);return sum(min(1,sum(v==p for j,v in enumerate(gold) if i!=j)/3) for i in range(len(gold)))/len(gold)
def option(pred,choices):
 if 'Failed to obtain answer via API' not in pred:
  if any(x in pred for x in ["Sorry, I can't help with images of people yet.","I can't process this file.","I'm sorry, but without the image provided",'Cannot determine the answer']):return 'Z'
  words=re.sub(r'[.()\[\],:;!*#{}]',' ',pred).split();found=[x for x in choices if x in words]
  if len(found)==1 and words.index(found[0])>len(words)-5:return found[0]
  if not found and 'Z' in words:return 'Z'
 if len(pred)>2*sum(len(str(x)) for x in choices.values()):return 'Z'
 found=[k for k,v in choices.items() if str(v).lower() in pred.lower()]
 return found[0] if len(found)==1 else 'Z'
def main():
 started=time.time();errors=[];checks=0;verified=0
 def req(v,m):
  nonlocal checks
  checks+=1
  if not bool(v):errors.append(m)
 try:
  proof=rd(E/'e1full001_completion.json');peer=rd(OUT/'e1full001_review.json');peerterm=rd(OUT/'monitor_e1full001/terminal.json');terminal=rd(E/'monitorscoree1full001/terminal.json');runstart=rd(E/'monitorscoree1full001/started.json');f=rd(A/'frozen.json');summary=rd(A/'analysis.json')
  req(peer['status']=='passed' and peer['n_verified']==384 and not peer['errors'] and peerterm.get('reason')=='completed' and peerterm.get('worker_returncode')==0 and peerterm.get('remaining_process_group')==[],'independent E1 review and exit')
  req(terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and terminal.get('remaining_process_group')==[] and terminal.get('worker_pid')==runstart.get('worker_pid'),'score supervisor')
  cmd=runstart['command'];cwd=Path(runstart['cwd']);req(any((cwd/x).resolve()==(B/'e1_analysis/score.py') for x in cmd if Path(x).name=='score.py') and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==A,'score command')
  req(summary['status']=='completed' and summary['n_questions']==384 and summary['scores_sha256']==sha(A/'scores.jsonl') and summary['acceptance_sha256']==sha(E/'e1full001_completion.json'),'score output binding')
  for k,p in [('acceptance_sha256',E/'e1full001_completion.json'),('acceptance_started_sha256',E/'monitorverifye1full001/started.json'),('acceptance_terminal_sha256',E/'monitorverifye1full001/terminal.json'),('dataset_summary_sha256',B/'data_manifest/manifest_summary.json')]:req(f[k]==sha(p),'frozen '+k)
  for path,h in f['code_sha256'].items():
   req(sha(path)==h,'metric source '+path)
   if Path(path).name in ('metrics.py','paired_stats.py','score.py'):req(sha(S/Path(path).name)==h,'snapshot '+path)
  for name,obj in f['inputs'].items():req(sha(obj['path'])==obj['sha256'],'accepted input '+name)
  panel=[r for r in ls(proof['files']['panel']['path']) if r['split']=='DEV'];journal=ls(proof['files']['journal']['path']);scored=ls(A/'scores.jsonl');req(len(panel)==len(scored)==384 and len(journal)==1536,'inventory')
  req(Counter(r['task'] for r in panel)=={x:128 for x in TASKS} and len({r['group_id'] for r in panel})==384,'DEV panel')
  need={(r['task'],str(r['qid'])) for r in panel};meta=rd(B/'data_manifest/manifest_summary.json');csv.field_size_limit(100_000_000);truths={}
  for task in TASKS:
   desc=meta['sources'][task];req(sha(desc['path'])==desc['sha256'],'dataset source '+task)
   with open(desc['path'],encoding='utf8',newline='') as file:
    for row in csv.DictReader(file,delimiter='\t'):
     key=(task,str(row['index']))
     if key not in need:continue
     if key in truths:raise ValueError('duplicate DEV qid')
     if task=='AI2D_TEST':truths[key]={'answer':row['answer'],'choices':{k:row[k] for k in string.ascii_uppercase if k in row and row[k] not in metrics.NATIVE_NA}}
     else:
      try: parsed=ast.literal_eval(row['answer'])
      except (ValueError,SyntaxError):parsed=row['answer']
      truths[key]={'answers':[str(x) for x in parsed] if isinstance(parsed,list) else [str(parsed)]}
  req(set(truths)==need,'DEV-only reference inventory')
  author=metrics.author_functions();norm=author['process_answer'];by={(r['task'],str(r['qid']),r['arm']):r for r in journal};req(len(by)==1536,'journal inventory')
  native={}
  for task in TASKS:
   for arm in ARMS:
    rows=[r for r in journal if r['task']==task and r['arm']==arm];stream=io.BytesIO();pd.DataFrame({'index':range(len(rows)),'prediction':[r['answer'] for r in rows]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);tab=pd.read_excel(stream)
    req(len(tab)==len(rows)==128 and list(tab['index'])==list(range(128)),'native spreadsheet denominator '+task+arm)
    for r,value in zip(rows,tab['prediction']):native[(task,str(r['qid']),arm)]=str(value)
  for i,(row,result) in enumerate(zip(panel,scored)):
   before=len(errors);key=(row['task'],str(row['qid']));label=':'.join(key)
   req(all(str(row[k])==str(result[k]) for k in ('task','qid','group_id')) and set(result['arms'])==set(ARMS),label+' identity')
   for arm in ARMS:
    j=by[(*key,arm)];x=result['arms'][arm];pred=native[(*key,arm)];g=truths[key]
    req(x['raw_answer']==j['answer'] and x['eos']==j['eos'] and x['truncated']==j['truncated'] and x['normalized_prediction']==pred,label+arm+' raw')
    if key[0]=='AI2D_TEST':
     official=float((author['can_infer'](pred,dict(g['choices'])) or 'Z')==g['answer']);independent=float(option(pred,g['choices'])==g['answer']);literal=float(option(j['answer'],g['choices'])==g['answer'])
    else:
     method='anls' if key[0]=='DocVQA_VAL' else 'vqa_score';v=author['process_line']({'answer':repr(g['answers']),'prediction':pred},method=method);official=float(author['hit_calculate']([v],key[0])[0])
     independent=doc(pred,g['answers']) if method=='anls' else text_score(pred,g['answers'],norm)
     literal=doc(j['answer'],g['answers']) if method=='anls' else text_score(j['answer'],g['answers'],norm)
    req(all(math.isclose(v,x[k],abs_tol=1e-12) for v,k in ((official,'score'),(independent,'independent_score'),(literal,'literal_answer_diagnostic'))) and abs(official-independent)<1e-12,label+arm+' scores')
   if len(errors)==before:verified+=1
  for task in TASKS:
   rows=[r for r in scored if r['task']==task];rep=summary['tasks'][task];groups=sorted({r['group_id'] for r in rows});req(len(groups)==128 and rep['n_questions']==128 and rep['n_image_groups']==128 and rep['replicates']==10000 and rep['seed']==20261104,task+' group counts')
   matrix=np.array([[r['arms'][a]['score'] for a in ARMS] for r in rows]);grouped={g:[r for r in rows if r['group_id']==g] for g in groups}
   totals=np.array([[sum(r['arms'][a]['score'] for r in grouped[g]) for a in ARMS] for g in groups]);sizes=np.array([len(grouped[g]) for g in groups]);draw=np.random.default_rng(20261104).integers(0,len(groups),size=(10000,len(groups)));boot=totals[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
   for i,a in enumerate(ARMS):req(math.isclose(rep['arm_means_pct'][a],100*matrix[:,i].mean(),abs_tol=1e-10),task+a+' mean')
   for name,c in CONTRASTS.items():
    d=matrix@np.array(c);ci=np.quantile(100*(boot@np.array(c)),[.025,.975]);z=rep['contrasts'][name]
    req(math.isclose(z['difference_pp'],100*d.mean(),abs_tol=1e-10) and np.allclose(z['image_cluster_percentile_95_ci_pp'],ci,atol=1e-10,rtol=0) and z['improved']==int((d>1e-12).sum()) and z['declined']==int((d< -1e-12).sum()) and z['unchanged']==int((abs(d)<=1e-12).sum()),task+name+' contrast')
  for path,h in f['code_sha256'].items():req(sha(path)==h,'source after scoring '+path)
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('exception: '+repr(exc))
 result={'status':'passed' if not errors and verified==384 else 'failed','n_expected':384,'n_verified':verified,'arms_verified':verified*4,'checks':checks,'errors':errors,'score_sha256':sha(A/'scores.jsonl'),'analysis_sha256':sha(A/'analysis.json'),'score_frozen_sha256':sha(A/'frozen.json'),'score_terminal_sha256':sha(E/'monitorscoree1full001/terminal.json'),'e1_review_sha256':sha(OUT/'e1full001_review.json'),'reviewer_sha256':sha(__file__),'scope':'384 DEV only; reread only needed qid reference fields; all 1536 arms, independent metrics and image-group bootstrap','start_unix':started,'end_unix':time.time()}
 with (OUT/'e1full001_score_review.json').open('x') as file:json.dump(result,file,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
 print(json.dumps(result));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
