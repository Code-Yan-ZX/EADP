"""DEV-only Pixel scoring after independent source and complete-panel admission."""
import argparse,collections,importlib.util,io,json,math,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path.insert(0,str(HERE))
import admission
ARMS=('pixel_dynamic','e1_p3_d1','e4_k256_d1','e4_k1024_d1')
CONTRASTS={'pixel_minus_e1':['pixel_dynamic','e1_p3_d1'],'pixel_minus_e4_k256':['pixel_dynamic','e4_k256_d1'],'pixel_minus_e4_k1024':['pixel_dynamic','e4_k1024_d1']}
def key(row):return (row['task'],str(row['qid']),row['group_id'])
def unique(rows):
 out={}
 for row in rows:
  k=key(row)
  if k in out:raise ValueError('duplicate question')
  out[k]=row
 return out
def lines(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def validate_pair(anchor,pixel,e1,e4,es,bs):
 for obj in (pixel,e1,e4,es):
  if key(obj)!=key(anchor):raise ValueError('paired identity')
 for obj in (pixel,e1,e4):
  if obj.get('message_sha256')!=anchor['message_sha256']:raise ValueError('message mismatch')
 for obj in (pixel,e4):
  if obj.get('image_sha256')!=anchor['image_sha256']:raise ValueError('image mismatch')
 if pixel['processor_identity']['processor']!=anchor['processor'] or e1['processor']!=anchor['processor'] or e4['processor']!=anchor['processor']:raise ValueError('processor identity mismatch')
 common=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty');pk=pixel['generation_kwargs'];shared={k:pk[k] for k in common}
 if set(e1['generation_scalars'])!=set(common) or set(e4['generation_scalars'])!=set(common) or shared!=e1['generation_scalars'] or shared!=e4['generation_scalars'] or pk.get('num_beams')!=1 or pk.get('use_cache') is not True or pixel['eos_token_ids']!=e1['eos_ids'] or pixel['eos_token_ids']!=e4['eos_ids']:raise ValueError('recipe mismatch')
 if e1['answer']!=es['arms']['P3-D1']['raw_answer'] or e4['answer']!=bs['answer'] or e1['eos']!=es['arms']['P3-D1']['eos'] or e4['eos']!=bs['eos']:raise ValueError('reference score/answer mismatch')
 for a in (es['arms']['P3-D1'],bs):
  if not math.isfinite(a['score']) or not math.isfinite(a['independent_score']) or abs(a['score']-a['independent_score'])>1e-12:raise ValueError('reference scoring mismatch')
def native_predictions(rows):
 import pandas as pd
 out={}
 for task in sorted({r['task'] for r in rows}):
  subset=[r for r in rows if r['task']==task];stream=io.BytesIO();pd.DataFrame({'index':list(range(len(subset))),'prediction':[r['decoded'] for r in subset]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);frame=pd.read_excel(stream)
  if len(frame)!=len(subset) or list(frame['index'])!=list(range(len(subset))):raise ValueError('native XLSX row identity')
  for row,value in zip(subset,frame['prediction']):out[key(row)]=str(value)
 if len(out)!=len(rows):raise ValueError('native XLSX duplicate')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();evidence=admission.admission()
 # No prediction data or label is parsed before the source and full-panel gate above.
 if any(x in sys.modules for x in ('paired','completion','metrics','paired_stats','score')):raise ValueError('preloaded scoring module rejected')
 import paired
 sys.path.insert(0,str(BASE/'e1_analysis'));import completion,metrics,score as e1score
 if any(Path(m.__file__).resolve()!=BASE/'e1_analysis'/f'{name}.py' for name,m in [('completion',completion),('metrics',metrics),('score',e1score)]):raise ValueError('scorer import origin')
 if Path(paired.__file__).resolve()!=HERE/'paired.py':raise ValueError('paired import origin')
 plan=admission.read(BASE/'e6/pixelprune_dev384/plan001.json');anchors=plan['anchors'];indexed=unique(anchors)
 if len(indexed)!=384 or collections.Counter(a['task'] for a in anchors)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('anchor inventory')
 obs=BASE/'e6/pixelprune_dev384/run_smoke001/observations';pixel=unique([admission.read(obs/(r['task']+'_'+str(r['qid'])+'.json')) for r in anchors]);e1=unique([r for r in lines(BASE/'e1/e1full001/raw_predictions.jsonl') if r['arm']=='P3-D1' and r['forced'] is False]);es=unique(lines(BASE/'e1/e1full001_score/scores.jsonl'));bs=unique(lines(BASE/'e4_analysis/score_panel003/scores.jsonl'))
 e4={}
 for r in lines(BASE/'e4_panel_v2/panel_003/records.jsonl'):
  if r['forced'] is False and r['ds_on'] is True and str(r['budget']) in ('256','1024'):
   k=(str(r['budget']),key(r))
   if k in e4:raise ValueError('duplicate E4 arm')
   e4[k]=r
 if not (set(indexed)==set(pixel)==set(e1)==set(es)==set(bs) and set(e4)=={(k,i) for i in indexed for k in ('256','1024')}):raise ValueError('paired inventory')
 for k,a in indexed.items():
  for budget in ('256','1024'):validate_pair(a,pixel[k],e1[k],e4[(budget,k)],es[k],bs[k]['arms']['K'+budget+'-D1'])
 ep=admission.read(BASE/'e1/e1full001_completion.json');summary=Path(ep['files']['dataset_summary']['path'])
 if admission.sha(summary)!=ep['files']['dataset_summary']['sha256']:raise ValueError('dataset summary mismatch')
 meta=admission.read(summary);needed={(x['task'],str(x['qid'])) for x in anchors}
 if len(needed)!=384 or set(meta['sources'])!={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'}:raise ValueError('DEV inventory')
 args.out.mkdir(parents=True,exist_ok=False)
 frozen={'scope':'DEV selected qids only','scorer_prelaunch':evidence,'plan_sha256':admission.sha(admission.PLAN),'references':meta['sources']}
 with (args.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 truth=e1score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(list(pixel.values()));results=[]
 for a in anchors:
  k=key(a);r=pixel[k];v=metrics.score(a['task'],r['decoded'],truth[(a['task'],str(a['qid']))],author,constants,scoring_prediction=native[k])
  if not math.isfinite(v['official']) or not math.isfinite(v['independent']) or abs(v['official']-v['independent'])>1e-12:raise ValueError('Pixel author/independent disagreement')
  result={'task':a['task'],'qid':str(a['qid']),'group_id':a['group_id'],'selected_count':r['selected_count'],'arms':{'pixel_dynamic':{'answer':r['decoded'],'score':v['official'],'independent_score':v['independent'],'eos':r['ends_eos'],'truncated':not r['ends_eos']},'e1_p3_d1':es[k]['arms']['P3-D1'],'e4_k256_d1':bs[k]['arms']['K256-D1'],'e4_k1024_d1':bs[k]['arms']['K1024-D1']}}
  results.append(result)
 reports={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  sub=[r for r in results if r['task']==task];rows=[]
  for r in sub:
   arms=r['arms'];rows.append({'group_id':r['group_id'],'scores':{x:arms[x]['score'] for x in ARMS},'answers':{x:arms[x].get('answer',arms[x].get('raw_answer')) for x in ARMS}})
  reports[task]=paired.analyze(rows,list(ARMS),CONTRASTS,replicates=10000,seed=20261104);ks=sorted(r['selected_count'] for r in sub);reports[task]['dynamic_k']={'min':ks[0],'median':(ks[63]+ks[64])/2,'max':ks[-1]};reports[task]['terminal']={'eos':sum(r['arms']['pixel_dynamic']['eos'] for r in sub),'truncated':sum(r['arms']['pixel_dynamic']['truncated'] for r in sub),'empty':sum(not r['arms']['pixel_dynamic']['answer'].strip() for r in sub)}
 for p,h in evidence['source_sha256'].items():
  if admission.sha(p)!=h:raise ValueError('scoring source changed')
 for d in meta['sources'].values():
  if admission.sha(d['path'])!=d['sha256']:raise ValueError('DEV source changed')
 with (args.out/'scores.jsonl').open('x') as f:
  for r in results:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','questions':384,'pixel_generations':384,'tasks':reports,'scores_sha256':admission.sha(args.out/'scores.jsonl'),'plan_sha256':admission.sha(admission.PLAN),'limitations':['DEV post-hoc exploratory; comparisons unadjusted','dynamic K differs from fixed-K compute; no latency claim','AI2D deterministic local scorer','full visual/DS/KV arrays not archived']}
 with (args.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('completed')
if __name__=='__main__':main()
