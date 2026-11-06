"""Complete DEV-only StackTok scoring; source and full-panel gate first."""
import argparse,collections,importlib.util,io,json,math,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];sys.path.insert(0,str(D));import admission
ARMS=('stacktok_k256','eadp_k256_d1','pixel_t0','pixel_t045')
CONTRASTS={'stacktok_minus_eadp':['stacktok_k256','eadp_k256_d1'],'stacktok_minus_pixel_t0':['stacktok_k256','pixel_t0'],'stacktok_minus_pixel_t045':['stacktok_k256','pixel_t045']}
def key(r):return (r['task'],str(r['qid']),r['group_id'])
def unique(rows):
 result={}
 for row in rows:
  k=key(row)
  if k in result:raise ValueError('duplicate question identity')
  result[k]=row
 return result
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def validate_pairs(anchors,records,old,refs):
 if not (len(anchors)==len(records)==len(old)==384) or set(anchors)!=set(records) or set(anchors)!=set(old):raise ValueError('full paired inventory mismatch')
 if collections.Counter(x['task'] for x in anchors.values())!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('task denominators')
 for k,a in anchors.items():
  r=records[k];o=old[k];ref=refs[(a['task'],str(a['qid']))]
  if any(r.get(x)!=a.get(x) for x in ('group_id','message_sha256','image_sha256')) or r.get('processor')!=a.get('processor') or r.get('processor')!=ref['processor'] or r.get('generation_scalars')!=ref['generation_scalars'] or r.get('eos_ids')!=ref['eos_ids'] or r.get('path')!='local_stacktok' or r.get('budget')!=256:raise ValueError('same-input/processor/recipe identity mismatch')
  for name in ('pixel_t0','pixel_t045','e4_k256_d1'):
   arm=o['arms'][name]
   if not math.isfinite(arm['score']) or not math.isfinite(arm['independent_score']) or abs(arm['score']-arm['independent_score'])>1e-12:raise ValueError('prior score disagreement')
 return True
def native_predictions(rows):
 import pandas as pd
 result={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  sub=[r for r in rows if r['task']==task];buffer=io.BytesIO();pd.DataFrame({'index':list(range(len(sub))),'prediction':[r['answer'] for r in sub]}).to_excel(buffer,index=False,engine='xlsxwriter');buffer.seek(0);frame=pd.read_excel(buffer)
  if len(frame)!=len(sub) or list(frame['index'])!=list(range(len(sub))):raise ValueError('native XLSX identity')
  for row,pred in zip(sub,frame['prediction']):result[key(row)]=str(pred)
 return result
def analyze_task(rows,paired):
 return paired.analyze(rows,list(ARMS),CONTRASTS,replicates=10000,seed=20261104)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 evidence=admission.require() # No target predictions, previously scored answers or GT before this.
 if any(x in sys.modules for x in ('paired','completion','metrics','score','full_refs')):raise ValueError('scoring module preloaded')
 import paired
 sys.path.insert(0,str(B/'e1_analysis'));import completion,metrics,score as e1score
 for name,module in (('completion',completion),('metrics',metrics),('score',e1score)):
  if Path(module.__file__).resolve()!=B/'e1_analysis'/f'{name}.py':raise ValueError('scorer origin mismatch')
 if Path(paired.__file__).resolve()!=D/'paired.py':raise ValueError('paired origin mismatch')
 path=B/'e6/stacktok_full/full_refs.py';spec=importlib.util.spec_from_file_location('_stacktok_full_score_refs',path);rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
 if Path(rf.__file__).resolve()!=path:raise ValueError('reference origin mismatch')
 anchors0=admission.read(B/'e6/pixelprune_t045_dev384/plan001.json')['anchors'];anchors=unique(anchors0);records=unique(lines(B/'e6/stacktok_full/run_001/records.jsonl'));old=unique(lines(B/'e6/pixelprune_t045_dev384_analysis/score001/scores.jsonl'))
 refs=rf.e4_records(anchors0);validate_pairs(anchors,records,old,refs)
 for k,r in records.items():rf.validate_reference(r,refs[(r['task'],str(r['qid']))])
 completion_proof=admission.read(B/'e1/e1full001_completion.json');summary=Path(completion_proof['files']['dataset_summary']['path'])
 if admission.sha(summary)!=completion_proof['files']['dataset_summary']['sha256']:raise ValueError('dataset summary changed')
 meta=admission.read(summary);needed={(x['task'],str(x['qid'])) for x in anchors0}
 if len(needed)!=384 or set(meta['sources'])!={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'}:raise ValueError('DEV truth inventory')
 a.out.mkdir(parents=True,exist_ok=False);frozen={'scope':'selected DEV qids only','prelaunch':evidence,'plan_sha256':admission.sha(D/'score_plan001.json'),'references':meta['sources']}
 with (a.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 truths=e1score.read_truths(meta['sources'],needed);authors=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(list(records.values()));out=[]
 for anchor in anchors0:
  k=key(anchor);r=records[k];prior=old[k]['arms'];v=metrics.score(anchor['task'],r['answer'],truths[(anchor['task'],str(anchor['qid']))],authors,constants,scoring_prediction=native[k])
  if not math.isfinite(v['official']) or not math.isfinite(v['independent']) or abs(v['official']-v['independent'])>1e-12:raise ValueError('StackTok author/independent scoring disagreement')
  arm={'answer':r['answer'],'score':v['official'],'independent_score':v['independent'],'eos':r['eos'],'truncated':r['truncated']}
  out.append({'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],'arms':{'stacktok_k256':arm,'eadp_k256_d1':prior['e4_k256_d1'],'pixel_t0':prior['pixel_t0'],'pixel_t045':prior['pixel_t045']}})
 reports={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  subset=[r for r in out if r['task']==task];rows=[]
  for r in subset:
   arms=r['arms'];rows.append({'group_id':r['group_id'],'scores':{x:arms[x]['score'] for x in ARMS},'answers':{x:arms[x].get('answer',arms[x].get('raw_answer')) for x in ARMS}})
  reports[task]=analyze_task(rows,paired);reports[task]['terminal']={'eos':sum(r['arms']['stacktok_k256']['eos'] for r in subset),'truncated':sum(r['arms']['stacktok_k256']['truncated'] for r in subset),'empty':sum(not r['arms']['stacktok_k256']['answer'].strip() for r in subset)}
 for p,h in evidence['source_sha256'].items():
  if admission.sha(p)!=h:raise ValueError('scoring sources changed')
 for x in meta['sources'].values():
  if admission.sha(x['path'])!=x['sha256']:raise ValueError('truth source changed')
 with (a.out/'scores.jsonl').open('x') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','questions':384,'tasks':reports,'scores_sha256':admission.sha(a.out/'scores.jsonl'),'plan_sha256':admission.sha(D/'score_plan001.json'),'limitations':['DEV exploration after prior experiments; three uncorrected contrasts per task','same K does not imply equal compute; Pixel dynamic K','tensor summaries do not permit full offline reconstruction']}
 with (a.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('completed')
if __name__=='__main__':main()
