"""Accepted full E4 DEV panel only. Gate runs before parsing predictions or GT."""
import argparse,importlib.metadata,io,json,math,os,sys,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path[:0]=[str(BASE/'e4_panel_v2'),str(BASE/'e4'),str(BASE/'e1_analysis')]
import completion,metrics,score as e1_score
from panel_v2_imports import load_local
panel_gate=load_local('panel_gate');panel_verify=load_local('panel_verify')
validate_supervisor=panel_gate.validate_supervisor;validated_dev=panel_gate.validated_dev
import paired
V2=BASE/'e4_panel_v2';E4=BASE/'e4'
V2_SOURCES=('panel_worker.py','panel_launch.py','panel_gate.py','panel_verify.py','native_panel_gate.py','panel_v2_imports.py','budget_plan.json')
E4_SOURCES=('smoke_contract.py','budget_runner.py','budget_adapter.py','decode_observer.py','off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py')

TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
PLAN_SHA256='f49fe73e99a8ce2a5fda72126d6d0bfcc382cc6bfc390847409a62df635d0e74'
def sha(p):return completion.sha(p)
def read(path):return json.loads(Path(path).read_text())
def config(k,on):
 if k not in (128,256,512,1024) or type(on) is not bool:raise ValueError('unsupported budget/DS')
 return 'K'+str(k)+'-D'+str(int(on))

def validate_independent_review(review_path,review_started,review_terminal,panel_proof,formal_started,formal_terminal):
    """Require a separately supervised reviewer, bound to this exact acceptance."""
    review=read(review_path);reviewer=BASE/'e1_independent_review/review_e4panel.py'
    if (review.get('status')!='passed' or review.get('n_expected')!=384 or review.get('n_verified')!=384 or
        review.get('arms_expected')!=3072 or review.get('arms_verified')!=3072 or review.get('errors')!=[] or
        review.get('GT_opened') is not False or review.get('scoring_performed') is not False or
        review.get('formal_proof_sha256')!=sha(panel_proof) or review.get('formal_verifier_started_sha256')!=sha(formal_started) or
        review.get('formal_verifier_terminal_sha256')!=sha(formal_terminal) or review.get('reviewer_sha256')!=sha(reviewer)):
        raise ValueError('independent E4 panel review not passed or mismatched')
    validate_supervisor(review_started,review_terminal,reviewer,review_path)
    return review

def validate_panel_sources(frozen, v2=V2, e4=E4):
 """Bind the complete runtime source inventory to the approved v2 implementation."""
 bindings=frozen.get('code_sha256',{})
 for path in [v2/x for x in V2_SOURCES]+[e4/x for x in E4_SOURCES]:
  if bindings.get(str(path.resolve()))!=sha(path):raise ValueError('panel runtime source missing or changed: '+str(path))
 for path,digest in bindings.items():
  if sha(path)!=digest:raise ValueError('panel runtime source changed: '+path)

def gate(args):
 """Fail closed before JSONL prediction decoding or any reference answers."""
 run=args.run.resolve();proof_path=args.panel_proof.resolve();started=args.verify_started.resolve();terminal=args.verify_terminal.resolve()
 validate_supervisor(started,terminal,V2/'panel_verify.py',proof_path)
 proof=read(proof_path)
 if (proof.get('status')!='passed' or proof.get('stage')!='full' or proof.get('n_questions')!=384 or proof.get('all_arms_validated')!=3072 or
     proof.get('k256_reuse') is not False or proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or
     proof.get('verifier_sha256')!=sha(V2/'panel_verify.py')):raise ValueError('E4 full-panel acceptance not passed')
 validate_independent_review(args.review_proof,args.review_started,args.review_terminal,proof_path,started,terminal)
 # The accepted verifier inputs are hashed without interpreting any predictions.
 accepted=proof.get('files_sha256',{});required=[args.e1_acceptance,args.e1_started,args.e1_terminal,args.run_started,args.run_terminal,run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'generated.jsonl']
 for path in required:
  p=Path(path).resolve()
  if accepted.get(str(p))!=sha(p):raise ValueError('accepted artifact missing or changed: '+str(p))
 for path,digest in accepted.items():
  if sha(path)!=digest:raise ValueError('accepted evidence changed: '+path)
 validate_panel_sources(read(run/'frozen.json'))
 panel_verify.validate_terminal(run,read(args.run_started),read(args.run_terminal),read(run/'launch.json'),read(run/'exit.json'))
 e1,rows=validated_dev(args.e1_acceptance,args.e1_started,args.e1_terminal)
 if len(rows)!=384 or e1.get('GT_opened') is not False or e1.get('scoring_performed') is not False:raise ValueError('upstream E1 proof invalid')
 return proof,e1,rows

def inventory(panel,records):
 """Bind all eight records per DEV sample, preserving author question order."""
 keys=('task','qid','group_id','message_sha256','image_sha256');grouped={};expected=[]
 for row in panel:
  q=(row['task'],str(row['qid']))
  if q in grouped:raise ValueError('duplicate panel question')
  grouped[q]={}
  for k in (128,256,512,1024):
   for on in (False,True):expected.append((q[0],q[1],k,on))
 actual=[(r.get('task'),str(r.get('qid')),r.get('budget'),r.get('ds_on')) for r in records]
 if actual!=expected:raise ValueError('full eight-arm inventory/order mismatch')
 for row_index,row in enumerate(panel):
  q=(row['task'],str(row['qid']))
  for k in (128,256,512,1024):
   for on in (False,True):
    r=records[len(grouped[q])+8*row_index]
    if any(str(r.get(x))!=str(row.get(x)) for x in keys) or r.get('forced') is not False:raise ValueError('record mismatch')
    grouped[q][config(k,on)]=r
 if len(panel)!=384 or Counter(r['task'] for r in panel)!={t:128 for t in TASKS} or len({r['group_id'] for r in panel})!=384:raise ValueError('DEV denominator mismatch')
 return grouped

def native_predictions(records,configs):
 """Match E1 author xlsxwriter -> pandas round trip for every task/config."""
 import pandas as pd
 result={}
 for task in TASKS:
  for arm in configs:
   sub=[r for r in records if r['task']==task and config(r['budget'],r['ds_on'])==arm]
   if len(sub)!=128:raise ValueError('xlsx denominator mismatch')
   stream=io.BytesIO();pd.DataFrame({'index':list(range(len(sub))),'prediction':[r['answer'] for r in sub]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);roundtrip=pd.read_excel(stream)
   if len(roundtrip)!=len(sub) or list(roundtrip['index'])!=list(range(len(sub))):raise ValueError('xlsx roundtrip changed rows')
   for r,x in zip(sub,roundtrip['prediction']):result[(task,str(r['qid']),arm)]=str(x)
 if len(result)!=len(records):raise ValueError('xlsx inventory mismatch')
 return result

def main():
 p=argparse.ArgumentParser()
 for name in ('run','panel-proof','verify-started','verify-terminal','run-started','run-terminal','e1-acceptance','e1-started','e1-terminal','review-proof','review-started','review-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if sha(HERE/'plan.json')!=PLAN_SHA256:raise ValueError('predeclared scoring plan changed')
 plan=read(HERE/'plan.json');pre={str(x):sha(x) for x in (HERE/'plan.json',HERE/'paired.py',Path(__file__),BASE/'e1_analysis/score.py',BASE/'e1_analysis/metrics.py',BASE/'e1_analysis/completion.py',*[V2/x for x in V2_SOURCES],*[E4/x for x in E4_SOURCES],BASE/'e1_independent_review/review_e4panel.py',metrics.VQA,metrics.MISC,metrics.MATCH)}
 proof,e1,panel=gate(a) # may hash evidence; no prediction JSON or GT before this gate
 records=completion.read_jsonl(a.run/'records.jsonl');groups=inventory(panel,records)
 dataset_path=Path(e1['files']['dataset_summary']['path']);meta=read(dataset_path)
 if sha(dataset_path)!=e1['files']['dataset_summary']['sha256'] or set(meta['sources'])!=set(TASKS):raise ValueError('dataset source proof')
 needed={(r['task'],str(r['qid'])) for r in panel}
 if len(needed)!=384:raise ValueError('bad DEV qids')
 a.out.mkdir(parents=True,exist_ok=False)
 frozen={'stage':'E4 DEV scoring','plan_sha256':sha(HERE/'plan.json'),'panel_proof_sha256':sha(a.panel_proof),'verifier_started_sha256':sha(a.verify_started),'verifier_terminal_sha256':sha(a.verify_terminal),'e1_acceptance_sha256':sha(a.e1_acceptance),'review_proof_sha256':sha(a.review_proof),'review_started_sha256':sha(a.review_started),'review_terminal_sha256':sha(a.review_terminal),'dataset_summary_sha256':sha(dataset_path),'source_code_sha256':pre,'reference_tsv':meta['sources'],'scope':'384 DEV only; non-DEV answer values not parsed','runtime_versions':{n:importlib.metadata.version(n) for n in ('numpy','pandas','xlsxwriter','openpyxl')}}
 with (a.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 # E1 implementation hashes whole TSVs but decodes/parses only selected DEV qids.
 truth=e1_score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(records,plan['configs']);scored=[]
 for row in panel:
  key=(row['task'],str(row['qid']));out={'task':key[0],'qid':key[1],'group_id':row['group_id'],'arms':{}}
  for arm,r in groups[key].items():
   value=metrics.score(key[0],r['answer'],truth[key],author,constants,scoring_prediction=native[(key[0],key[1],arm)])
   if not math.isfinite(value['official']) or abs(value['official']-value['independent'])>1e-12:raise ValueError('author/independent scorer disagree')
   out['arms'][arm]={'answer':r['answer'],'score':value['official'],'independent_score':value['independent'],'eos':r['eos'],'truncated':r['truncated']}
  scored.append(out)
 tasks={}
 for task in TASKS:
  subset=[x for x in scored if x['task']==task]
  rows=[{'group_id':x['group_id'],'scores':{arm:x['arms'][arm]['score'] for arm in plan['configs']},'answers':{arm:x['arms'][arm]['answer'] for arm in plan['configs']}} for x in subset]
  tasks[task]=paired.analyze(rows,plan['configs'],plan['contrasts'],replicates=plan['bootstrap']['replicates'],seed=plan['bootstrap']['seed'])
  tasks[task]['terminal']={arm:{'eos':sum(x['arms'][arm]['eos'] for x in subset),'truncated':sum(x['arms'][arm]['truncated'] for x in subset),'empty':sum(not x['arms'][arm]['answer'].strip() for x in subset)} for arm in plan['configs']}
 for p,h in pre.items():
  if sha(p)!=h:raise ValueError('analysis source changed')
 for task,desc in meta['sources'].items():
  if sha(desc['path'])!=desc['sha256']:raise ValueError('source TSV changed')
 for path,digest in proof['files_sha256'].items():
  if sha(path)!=digest:raise ValueError('accepted panel artifact changed')
 score_path=a.out/'scores.jsonl'
 with score_path.open('x') as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','n_questions':384,'n_predictions':len(records),'tasks':tasks,'scores_sha256':sha(score_path),'plan_sha256':sha(HERE/'plan.json'),'panel_proof_sha256':sha(a.panel_proof),'limitations':['DEV exploratory only; intervals per contrast are not adjusted for multiple comparisons.','AI2D uses local exact matching, no external judge.','Instrumented panel is not a performance benchmark.']}
 with (a.out/'analysis.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'completed','n_questions':384,'n_predictions':len(records)}))
if __name__=='__main__':main()
