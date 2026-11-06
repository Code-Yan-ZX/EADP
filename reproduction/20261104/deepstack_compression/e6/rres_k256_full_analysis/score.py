"""Score only a completely accepted R-res DEV panel, never partial outputs."""
import argparse,collections,hashlib,importlib.util,io,json,math,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');SEEDS=(20261104,20261105,20261106)
ADMISSION_SHA='08157287a1aa0521f1847ffbdee93e9ff0cf3f91b2a98c11961794eed5b872db'
def load_gate():
 path=D/'admission.py';name='_rres_score_admission'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=ADMISSION_SHA or name in sys.modules:raise ValueError('unreviewed or preloaded admission')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def key(row):return row['task'],str(row['qid']),row['group_id']
def unique(rows,seeds=False):
 out={}
 for x in rows:
  k=key(x)+((x['seed'],) if seeds else ())
  if k in out:raise ValueError('duplicate identity')
  out[k]=x
 return out
def finite_score(arm):
 a=arm.get('score');b=arm.get('independent_score')
 if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=1 or abs(a-b)>1e-12:raise ValueError('invalid accepted scoring')
def validate_pairs(anchors,target,e1,eadp,random_records,random_scores,e1_full):
 if len(anchors)!=384 or collections.Counter(k[0] for k in anchors)!={t:128 for t in TASKS}:raise ValueError('full DEV denominator')
 if set(anchors)!=set(target) or set(anchors)!=set(e1) or set(anchors)!=set(e1_full) or set(anchors)!=set(eadp) or set(anchors)!=set(random_scores) or set(random_records)!={(*k,s) for k in anchors for s in SEEDS}:raise ValueError('full matched inventory')
 for k,ref in anchors.items():
  x=target[k];e=e1[k];old=eadp[k]['arms']['P3-D1'];rs=random_scores[k]['arms'];finite_score(old)
  if e.get('message_sha256')!=ref['message_sha256'] or e.get('processor')!=ref['processor'] or e.get('answer')!=old['raw_answer'] or e.get('eos')!=old['eos']:raise ValueError('E1 score or processor identity')
  full=e1_full[k];full_score=eadp[k]['arms']['P1-D1'];finite_score(full_score)
  if full.get('message_sha256')!=ref['message_sha256'] or full.get('processor')!=ref['processor'] or full.get('answer')!=full_score['raw_answer'] or full.get('eos')!=full_score['eos'] or full.get('generation_scalars')!=e.get('generation_scalars') or full.get('eos_ids')!=e.get('eos_ids'):raise ValueError('full E1 score or identity')
  if any(x.get(v)!=ref.get(v) for v in ('image_sha256','message_sha256')) or x.get('resolution')!=512 or x.get('visual_tokens')!=256 or x.get('generation_scalars')!=e.get('generation_scalars') or x.get('eos_ids')!=e.get('eos_ids') or x.get('hooks_and_methods_restored') is not True:raise ValueError('R-res input or recipe')
  proc=x.get('processor',{});grid=proc.get('image_grid_thw',{});pixels=proc.get('pixel_values',{});ids=proc.get('input_ids',{})
  if grid.get('shape')!=[1,3] or pixels.get('shape')!=[1024,1536] or not isinstance(ids.get('shape'),list) or len(ids['shape'])!=2 or ids['shape'][0]!=1:raise ValueError('R-res processor shape')
  for seed in SEEDS:
   r=random_records[(*k,seed)];s=rs[str(seed)];finite_score(s)
   if (r.get('seed')!=seed or r.get('image_sha256')!=ref['image_sha256'] or r.get('message_sha256')!=ref['message_sha256'] or r.get('processor')!=ref['processor'] or r.get('generation_scalars')!=e['generation_scalars'] or r.get('eos_ids')!=e['eos_ids'] or r.get('budget')!=256 or r.get('ds_on') is not True or r.get('protocol')!='natural' or r.get('forced') is not False or r.get('answer')!=s['answer'] or r.get('eos')!=s['eos'] or r.get('truncated')!=s['truncated']):raise ValueError('random record or score mismatch')
  finite_score(rs['eadp'])
  if abs(rs['eadp']['score']-old['score'])>1e-12 or rs['eadp']['raw_answer']!=old['raw_answer']:raise ValueError('random score E1 binding')
 return True
def native_predictions(rows):
 import pandas as pd
 out={}
 for task in TASKS:
  subset=[r for r in rows if r['task']==task];stream=io.BytesIO()
  pd.DataFrame({'index':list(range(len(subset))),'prediction':[r['answer'] for r in subset]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);frame=pd.read_excel(stream)
  if len(frame)!=len(subset) or list(frame['index'])!=list(range(len(subset))):raise ValueError('native XLSX identity')
  for r,v in zip(subset,frame['prediction']):out[key(r)]=str(v)
 if len(out)!=len(rows):raise ValueError('native XLSX duplicate')
 return out
def lines(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);a=parser.parse_args()
 if a.out.resolve()!=D/'score001' or a.out.exists():raise ValueError('unexpected destination')
 admission=load_gate();evidence=admission.require()
 if any(x in sys.modules for x in ('paired','completion','metrics','score')):raise ValueError('preloaded scoring module')
 sys.path.insert(0,str(D));import paired
 sys.path.insert(0,str(B/'e1_analysis'));import completion,metrics,score as e1score
 if Path(paired.__file__).resolve()!=D/'paired.py' or any(Path(m.__file__).resolve()!=B/'e1_analysis'/(n+'.py') for n,m in (('completion',completion),('metrics',metrics),('score',e1score))):raise ValueError('scorer path')
 anchors=unique(admission.read(B/'e6/pixelprune_t020_dev384/plan001.json')['anchors']);directory=B/'e6/rres_k256_dev384/run001/observations'
 target=unique([admission.read(directory/(x['task']+'_'+str(x['qid'])+'.json')) for x in anchors.values()])
 journal=lines(B/'e1/e1full001/raw_predictions.jsonl');e1=unique([x for x in journal if x['arm']=='P3-D1' and x['forced'] is False]);e1_full=unique([x for x in journal if x['arm']=='P1-D1' and x['forced'] is False]);eadp=unique(lines(B/'e1/e1full001_score/scores.jsonl'))
 random_records=unique(lines(B/'e6/random_full_exec_v2/run_001/records.jsonl'),seeds=True);random_scores=unique(lines(B/'e6/random_full_analysis_v3/score001/scores.jsonl'))
 validate_pairs(anchors,target,e1,eadp,random_records,random_scores,e1_full)
 proof=admission.read(B/'e1/e1full001_completion.json');summary=Path(proof['files']['dataset_summary']['path'])
 if admission.sha(summary)!=proof['files']['dataset_summary']['sha256']:raise ValueError('dataset metadata')
 meta=admission.read(summary);needed={(r['task'],str(r['qid'])) for r in anchors.values()}
 if len(needed)!=384 or set(meta['sources'])!=set(TASKS):raise ValueError('truth scope')
 a.out.mkdir(parents=True,exist_ok=False)
 with (a.out/'frozen.json').open('x') as f:json.dump({'scope':'selected DEV qids only','admission':evidence,'references':meta['sources']},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 truth=e1score.read_truths(meta['sources'],needed);authors=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(list(target.values()));result=[]
 for k,anchor in anchors.items():
  r=target[k];v=metrics.score(anchor['task'],r['answer'],truth[(anchor['task'],str(anchor['qid']))],authors,constants,scoring_prediction=native[k])
  if not math.isfinite(v['official']) or not math.isfinite(v['independent']) or abs(v['official']-v['independent'])>1e-12:raise ValueError('target official/independent metric')
  result.append({'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],
                 'rres':{'answer':r['answer'],'score':v['official'],'independent_score':v['independent'],'eos':r['eos'],'truncated':r['truncated']},
                 'eadp':eadp[k]['arms']['P3-D1'],'e1_full':eadp[k]['arms']['P1-D1'],'random':{str(s):random_scores[k]['arms'][str(s)] for s in SEEDS}})
 reports={}
 for task in TASKS:
  subset=[r for r in result if r['task']==task];rows=[{'qid':r['qid'],'group_id':r['group_id'],'scores':{'rres':r['rres']['score'],'eadp':r['eadp']['score'],'e1_full':r['e1_full']['score'],**{s:r['random'][str(s)]['score'] for s in SEEDS}}} for r in subset]
  reports[task]=paired.analyze(rows);reports[task]['rres_terminal']={'eos':sum(x['rres']['eos'] for x in subset),'truncated':sum(x['rres']['truncated'] for x in subset),'empty':sum(not x['rres']['answer'].strip() for x in subset)}
 if any(admission.sha(p)!=h for p,h in evidence['source_sha256'].items()) or any(admission.sha(v['path'])!=v['sha256'] for v in meta['sources'].values()):raise ValueError('source or truth changed')
 with (a.out/'scores.jsonl').open('x') as f:
  for r in result:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','questions':384,'tasks':reports,'scores_sha256':admission.sha(a.out/'scores.jsonl'),'plan_sha256':evidence['plan_sha256'],
         'limitations':['posthoc DEV exploration; bootstrap intervals uncorrected','same K does not imply equal compute or same image resolution','three fixed random seeds averaged within each question; no seed-population uncertainty','random adapter performs discarded author greedy; its timing is not comparable']}
 with (a.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
