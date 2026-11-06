"""Independently rescore all 384 paired DEV questions; no CONFIRM/FINAL answers."""
import argparse,collections,copy,importlib.util,io,json,math,os,sys
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/ziprerank_full_dev_v2_analysis';O=D/'score001';N=B/'e6/ziprerank_full_dev_v2/run_smoke001/observations';A=B/'e1_analysis'
spec=importlib.util.spec_from_file_location('_zip_review_gate',D/'admission.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
sys.path.insert(0,str(A));import metrics,score as e1score
spec=importlib.util.spec_from_file_location('_e1_independent_math',R/'review_score.py');ind=importlib.util.module_from_spec(spec);spec.loader.exec_module(ind)
ARMS=('zip_prefix_rope','zip_user_text_rope','eadp_p3_d1');METHOD={'zip_prefix_rope':'author_prefix_256','zip_user_text_rope':'user_text_256'};TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');CONTRASTS={'prefix_minus_eadp':(0,2),'user_text_minus_eadp':(1,2),'user_text_minus_prefix':(1,0)}
def req(x,msg):
 if not bool(x):raise ValueError(msg)
def rows(p):
 with open(p) as h:return [json.loads(s) for s in h if s.strip()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();g.supervisor(D/'monitor_score001',D/'score.py',O);ev=g.admission();plan=g.read(g.D/'plan001.json');anchors=plan['anchors'];report=g.read(O/'analysis.json');frozen=g.read(O/'frozen.json')
 req(report['status']=='completed' and report['questions']==384 and report['zip_generations']==768 and report['scores_sha256']==g.sha(O/'scores.jsonl') and report['plan_sha256']==g.sha(g.PLAN),'completed score output')
 req(frozen['scorer_prelaunch']['zip_formal_sha256']==ev['zip_formal_sha256'] and frozen['scorer_prelaunch']['zip_review_sha256']==ev['zip_review_sha256'] and frozen['scorer_prelaunch']['prelaunch_review_sha256']==ev['prelaunch_review_sha256'] and frozen['plan_sha256']==g.sha(g.PLAN),'score provenance')
 scored=rows(O/'scores.jsonl');req(len(anchors)==len(scored)==384 and collections.Counter(x['task'] for x in anchors)=={t:128 for t in TASKS} and len({key(x) for x in anchors})==384,'panel')
 req([key(x) for x in anchors]==[key(x) for x in scored],'score order')
 zip_rows={};base_scores={key(x):x for x in rows(B/'e1/e1full001_score/scores.jsonl')};base_predictions={(x['task'],str(x['qid']),x['group_id']):x for x in rows(B/'e1/e1full001/raw_predictions.jsonl') if x['arm']=='P3-D1' and x['forced'] is False}
 req(set(base_scores)==set(base_predictions)=={key(x) for x in anchors},'E1 inventory')
 for x in anchors:
  k=key(x);task,qid,_=k;processor=g.read(N/(task+'_'+qid+'_processor.json'));base=base_predictions[k];req(base['message_sha256']==x['message_sha256'] and processor['processor']==base['processor'],'input identity')
  for arm,name in METHOD.items():
   row=g.read(N/(task+'_'+qid+'_'+name+'.json'));req(key(row)==k and row['arm']==name and row['status']=='completed' and row['local_rope_adapter'] is True and row['full_user_text_adapter'] is (name=='user_text_256'),'zip identity');ctrl=row['generation_controls'];eos=ctrl.get('eos_token_id');eos=[eos] if isinstance(eos,int) else eos or []
   req(len(base['generation_scalars'])==6 and all(ctrl.get(k)==v for k,v in base['generation_scalars'].items()) and eos==base['eos_ids'],'controls');zip_rows[(arm,k)]=row
 summary=g.read(B/'e1/e1full001_completion.json');meta_path=Path(summary['files']['dataset_summary']['path']);req(g.sha(meta_path)==summary['files']['dataset_summary']['sha256'],'reference manifest');meta=g.read(meta_path);needed={(x['task'],str(x['qid'])) for x in anchors};truth=e1score.read_truths(meta['sources'],needed);author=metrics.author_functions();normal=author['process_answer']
 native={}
 for task in TASKS:
  for arm in METHOD:
   subset=[x for x in anchors if x['task']==task];f=io.BytesIO();pd.DataFrame({'index':range(len(subset)),'prediction':[zip_rows[(arm,key(x))]['answer'] for x in subset]}).to_excel(f,index=False,engine='xlsxwriter');f.seek(0);tab=pd.read_excel(f);req(len(tab)==128 and list(tab['index'])==list(range(128)),'xlsx')
   for x,v in zip(subset,tab['prediction']):native[(arm,key(x))]=str(v)
 checked=0
 for x,result in zip(anchors,scored):
  k=key(x);task,qid,_=k;truth_item=truth[(task,qid)];req(set(result['arms'])==set(ARMS),'arms')
  for arm in ARMS:
   out=result['arms'][arm]
   if arm in METHOD:
    source=zip_rows[(arm,k)];pred=native[(arm,k)];raw=source['answer'];req(out['answer']==raw and out['eos']==source['eos'] and out['truncated']==source['truncated'],'zip output')
   else:
    source=base_predictions[k];baseline=base_scores[k]['arms']['P3-D1'];req(out==baseline and baseline['raw_answer']==source['answer'] and baseline['eos']==source['eos'] and baseline['truncated']==source['truncated'],'E1 output');pred=baseline['normalized_prediction'];raw=source['answer']
   if task=='AI2D_TEST':
    choices=truth_item['choices'];correct=truth_item['answer'];official=float((author['can_infer'](pred,copy.deepcopy(choices)) or 'Z')==correct);independent=float(ind.option(pred,choices)==correct)
   else:
    answers=truth_item['answers'];method='anls' if task=='DocVQA_VAL' else 'vqa_score';v=author['process_line']({'answer':repr(answers),'prediction':pred},method=method);official=float(author['hit_calculate']([v],task)[0]);independent=ind.doc(pred,answers) if method=='anls' else ind.text_score(pred,answers,normal)
   req(all(math.isfinite(z) for z in (official,independent,out['score'],out['independent_score'])) and abs(official-independent)<=1e-12 and abs(out['score']-official)<=1e-12 and abs(out['independent_score']-independent)<=1e-12,'score mismatch');checked+=1
 for task in TASKS:
  items=[x for x in scored if x['task']==task];r=report['tasks'][task];groups=sorted({x['group_id'] for x in items});grouped={v:[x for x in items if x['group_id']==v] for v in groups};m=np.array([[x['arms'][a]['score'] for a in ARMS] for x in items]);totals=np.array([[sum(x['arms'][a]['score'] for x in grouped[v]) for a in ARMS] for v in groups]);sizes=np.array([len(grouped[v]) for v in groups]);draw=np.random.default_rng(20261104).integers(0,len(groups),(10000,len(groups)));boot=totals[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
  req(r['n_questions']==128 and r['n_image_groups']==len(groups),'group denominator')
  for i,arm in enumerate(ARMS):req(abs(r['arm_means_pct'][arm]-100*m[:,i].mean())<1e-10 and np.allclose(r['arm_percentile_95_ci_pct'][arm],np.quantile(100*boot[:,i],[.025,.975]),atol=1e-10,rtol=0),'arm statistic')
  for name,(i,j) in CONTRASTS.items():
   z=r['contrasts'][name];d=m[:,i]-m[:,j];ci=np.quantile(100*(boot[:,i]-boot[:,j]),[.025,.975]);answers=lambda x,arm:x['arms'][arm].get('answer',x['arms'][arm].get('raw_answer'))
   req(abs(z['difference_pp']-100*d.mean())<1e-10 and np.allclose(z['image_cluster_percentile_95_ci_pp'],ci,atol=1e-10,rtol=0) and z['improved']==int((d>1e-12).sum()) and z['declined']==int((d< -1e-12).sum()) and z['unchanged']==int((abs(d)<=1e-12).sum()) and z['answer_changed']==sum(answers(x,ARMS[i])!=answers(x,ARMS[j]) for x in items),'paired statistic')
  for arm in METHOD:
   z=r['terminal'][arm];req(z['eos']==sum(x['arms'][arm]['eos'] for x in items) and z['truncated']==sum(x['arms'][arm]['truncated'] for x in items) and z['empty']==sum(not x['arms'][arm]['answer'].strip() for x in items),'empty/eos count')
 req(checked==1152 and all(g.sha(p)==h for p,h in ev['source_sha256'].items()),'source after verification')
 out={'status':'passed','independent_review':True,'questions':384,'arms_verified':checked,'GT_scope':'384 DEV qids only','scoring_performed':True,'score_sha256':g.sha(O/'scores.jsonl'),'analysis_sha256':g.sha(O/'analysis.json'),'frozen_sha256':g.sha(O/'frozen.json'),'score_started_sha256':g.sha(D/'monitor_score001/started.json'),'score_terminal_sha256':g.sha(D/'monitor_score001/terminal.json'),'reviewer_sha256':g.sha(__file__),'prelaunch_review_sha256':g.sha(R/'zip_full_dev_v2_score_prelaunch_review.json'),'limitations':['DEV posthoc and unadjusted; neither equal computation nor latency established','only six recorded generation scalars and EOS establish comparability','local RoPE repair in both Zip arms, user-text selection includes instructions/options'],'errors':[]}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 print('independent scores passed')
if __name__=='__main__':main()
