"""DEV-only scoring of two local ZipRerank adaptations after full admission."""
import argparse,collections,hashlib,importlib.util,io,json,math,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path.insert(0,str(HERE))
def load_admission(path,expected):
 path=Path(path).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('admission source hash mismatch')
 if '_zip_full_score_admission' in sys.modules:raise ValueError('preloaded admission rejected')
 spec=importlib.util.spec_from_file_location('_zip_full_score_admission',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 if Path(module.__file__).resolve()!=path:raise ValueError('admission path mismatch')
 return module

ADMISSION_SHA='c18f8fde56fbe4dbfc283cda3e23be28ae6546e1dbe0432d22fd31eacfab43bd'
admission=load_admission(HERE/'admission.py',ADMISSION_SHA)
ARMS=('zip_prefix_rope','zip_user_text_rope','eadp_p3_d1')
CONTRASTS={'prefix_minus_eadp':['zip_prefix_rope','eadp_p3_d1'],'user_text_minus_eadp':['zip_user_text_rope','eadp_p3_d1'],'user_text_minus_prefix':['zip_user_text_rope','zip_prefix_rope']}
METHODS={'zip_prefix_rope':'author_prefix_256','zip_user_text_rope':'user_text_256'}
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
def validate_pair(anchor,zip_row,processor,e1,e1score,method):
 for row in (zip_row,e1,e1score):
  if key(row)!=key(anchor):raise ValueError('paired identity')
 if e1.get('message_sha256')!=anchor['message_sha256'] or processor['processor']!=e1['processor']:raise ValueError('E1 message/processor mismatch')
 if zip_row.get('arm')!=method or zip_row.get('status')!='completed' or zip_row.get('local_rope_adapter') is not True or zip_row.get('full_user_text_adapter') is not (method=='user_text_256'):raise ValueError('local method mismatch')
 ctrl=zip_row['generation_controls'];shared=e1['generation_scalars'];eos=ctrl.get('eos_token_id');eos=[eos] if isinstance(eos,int) else eos or []
 if set(shared)!={'do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty'} or any(ctrl.get(k)!=v for k,v in shared.items()) or eos!=e1['eos_ids']:raise ValueError('E1 generation scalar or EOS mismatch')
 arm=e1score['arms']['P3-D1']
 if e1['answer']!=arm['raw_answer'] or e1['eos']!=arm['eos'] or not math.isfinite(arm['score']) or not math.isfinite(arm['independent_score']) or abs(arm['score']-arm['independent_score'])>1e-12:raise ValueError('reference scoring mismatch')
def native_predictions(rows):
 import pandas as pd
 out={}
 for task in sorted({r['task'] for r in rows}):
  subset=[r for r in rows if r['task']==task];stream=io.BytesIO();pd.DataFrame({'index':list(range(len(subset))),'prediction':[r['answer'] for r in subset]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);frame=pd.read_excel(stream)
  if len(frame)!=len(subset) or list(frame['index'])!=list(range(len(subset))):raise ValueError('XLSX identity mismatch')
  for row,value in zip(subset,frame['prediction']):out[key(row)]=str(value)
 if len(out)!=len(rows):raise ValueError('XLSX duplicate')
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();evidence=admission.admission()
 if any(x in sys.modules for x in ('paired','completion','metrics','paired_stats','score')):raise ValueError('preloaded scoring module rejected')
 import paired
 sys.path.insert(0,str(BASE/'e1_analysis'));import completion,metrics,score as e1score
 if any(Path(m.__file__).resolve()!=BASE/'e1_analysis'/f'{name}.py' for name,m in [('completion',completion),('metrics',metrics),('score',e1score)]):raise ValueError('scorer import origin')
 if Path(paired.__file__).resolve()!=HERE/'paired.py':raise ValueError('paired import origin')
 plan=admission.read(BASE/'e6/ziprerank_full_dev_v2/plan001.json');anchors=plan['anchors'];indexed=unique(anchors)
 if len(indexed)!=384 or collections.Counter(a['task'] for a in anchors)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('anchor inventory')
 obs=BASE/'e6/ziprerank_full_dev_v2/run_smoke001/observations';ziprows={};processors={}
 for a in anchors:
  k=key(a);prefix=a['task']+'_'+str(a['qid']);processors[k]=admission.read(obs/(prefix+'_processor.json'))
  for label,name in METHODS.items():ziprows[(label,k)]=admission.read(obs/(prefix+'_'+name+'.json'))
 e1=unique([r for r in lines(BASE/'e1/e1full001/raw_predictions.jsonl') if r['arm']=='P3-D1' and r['forced'] is False]);scores=unique(lines(BASE/'e1/e1full001_score/scores.jsonl'))
 if set(indexed)!=set(e1) or set(indexed)!=set(scores) or set(ziprows)!={(name,k) for name in METHODS for k in indexed}:raise ValueError('paired inventory mismatch')
 for k,a in indexed.items():
  for label,name in METHODS.items():validate_pair(a,ziprows[(label,k)],processors[k],e1[k],scores[k],name)
 ep=admission.read(BASE/'e1/e1full001_completion.json');summary=Path(ep['files']['dataset_summary']['path'])
 if admission.sha(summary)!=ep['files']['dataset_summary']['sha256']:raise ValueError('dataset summary mismatch')
 meta=admission.read(summary);needed={(x['task'],str(x['qid'])) for x in anchors}
 if len(needed)!=384 or set(meta['sources'])!={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'}:raise ValueError('DEV inventory')
 args.out.mkdir(parents=True,exist_ok=False)
 with (args.out/'frozen.json').open('x') as f:json.dump({'scope':'DEV selected qids only','scorer_prelaunch':evidence,'plan_sha256':admission.sha(admission.PLAN),'references':meta['sources']},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 truth=e1score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants()
 native={label:native_predictions([ziprows[(label,key(a))] for a in anchors]) for label in METHODS};results=[]
 for a in anchors:
  k=key(a);values={}
  for label in METHODS:
   row=ziprows[(label,k)];v=metrics.score(a['task'],row['answer'],truth[(a['task'],str(a['qid']))],author,constants,scoring_prediction=native[label][k])
   if not math.isfinite(v['official']) or not math.isfinite(v['independent']) or abs(v['official']-v['independent'])>1e-12:raise ValueError('Zip author/independent scoring mismatch')
   values[label]={'answer':row['answer'],'score':v['official'],'independent_score':v['independent'],'eos':row['eos'],'truncated':row['truncated']}
  values['eadp_p3_d1']=scores[k]['arms']['P3-D1'];results.append({'task':a['task'],'qid':str(a['qid']),'group_id':a['group_id'],'arms':values})
 reports={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  subset=[r for r in results if r['task']==task];rows=[]
  for r in subset:
   arms=r['arms'];rows.append({'group_id':r['group_id'],'scores':{x:arms[x]['score'] for x in ARMS},'answers':{x:arms[x].get('answer',arms[x].get('raw_answer')) for x in ARMS}})
  reports[task]=paired.analyze(rows,list(ARMS),CONTRASTS,replicates=10000,seed=20261104)
  reports[task]['terminal']={x:{'eos':sum(r['arms'][x]['eos'] for r in subset),'truncated':sum(r['arms'][x]['truncated'] for r in subset),'empty':sum(not r['arms'][x]['answer'].strip() for r in subset)} for x in METHODS}
 for p,h in evidence['source_sha256'].items():
  if admission.sha(p)!=h:raise ValueError('scoring source changed')
 for d in meta['sources'].values():
  if admission.sha(d['path'])!=d['sha256']:raise ValueError('DEV source changed')
 with (args.out/'scores.jsonl').open('x') as f:
  for r in results:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','questions':384,'zip_generations':768,'tasks':reports,'scores_sha256':admission.sha(args.out/'scores.jsonl'),'plan_sha256':admission.sha(admission.PLAN),'limitations':['DEV post-hoc exploratory; comparisons unadjusted','both Zip paths use local RoPE repair; user-text path also changes selection scope','fixed K does not prove equal compute; no latency claim','AI2D deterministic local scorer']}
 with (args.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('completed')
if __name__=='__main__':main()
