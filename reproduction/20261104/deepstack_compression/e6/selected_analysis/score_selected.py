"""Complete, independently accepted Div/CD DEV scoring only.

Do not parse prediction records or decode ground truth until gate() returns.
E1's reader locates selected DEV qids in bytes before decoding any answer.
"""
import argparse,hashlib,importlib.metadata,importlib.util,io,json,math,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];E1=BASE/'e1_analysis'
ADMISSION_SHA256='da8b36b1b8a484acc9ca9f21b19d48222f4c680fd746904416cca71b5ac40524'
def load_exact(name,path,expected):
 path=Path(path).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('trusted source changed: '+str(path))
 if name in sys.modules:
  raise ValueError('preloaded module name rejected: '+name)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 if Path(module.__file__).resolve()!=path:raise ValueError('import path mismatch')
 return module
admission=load_exact('eadp_selected_admission',HERE/'admission.py',ADMISSION_SHA256)
def load_scoring_modules(sources):
 global gate,inventory,paired,completion,metrics,e1_score
 def load(name,path):return load_exact(name,path,sources[str(path.resolve())])
 gate=load('eadp_selected_gate',HERE/'gate.py');inventory=load('eadp_selected_inventory',HERE/'inventory.py');paired=load('eadp_selected_paired',HERE/'paired.py')
 completion=load('completion',E1/'completion.py');metrics=load('metrics',E1/'metrics.py');load('paired_stats',E1/'paired_stats.py');e1_score=load('eadp_selected_e1_score',E1/'score.py')
PLAN_SHA256='f397d895e27b143e1dab1ada05f8f039e8913fcc7a651eacfac9cd13c6832b43'

def native_predictions(records,arms):
 """Match author XLSX/pandas per-task/per-arm conversion exactly."""
 import pandas as pd
 result={}
 for task in inventory.TASKS:
  for arm in arms:
   sub=[r for r in records if r['task']==task and r['method']+':'+r['path']==arm]
   if len(sub)!=128:raise ValueError('native roundtrip denominator mismatch')
   stream=io.BytesIO();pd.DataFrame({'index':list(range(len(sub))),'prediction':[r['answer'] for r in sub]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);frame=pd.read_excel(stream)
   if len(frame)!=len(sub) or list(frame['index'])!=list(range(len(sub))):raise ValueError('native roundtrip changed rows')
   for row,value in zip(sub,frame['prediction']):result[(task,str(row['qid']),arm)]=str(value)
 if len(result)!=len(records):raise ValueError('native roundtrip duplicated record')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 approved=admission.admit() # Independent scorer review precedes all scorer imports/data.
 load_scoring_modules(approved['source_sha256'])
 if gate.sha(HERE/'plan.json')!=PLAN_SHA256:raise ValueError('predeclared plan changed')
 plan=gate.read(HERE/'plan.json');evidence=gate.gate() # Full two-review gate before predictions and GT.
 code=[Path(__file__),HERE/'gate.py',HERE/'inventory.py',HERE/'paired.py',HERE/'plan.json',E1/'score.py',E1/'completion.py',E1/'metrics.py',metrics.VQA,metrics.MISC,metrics.MATCH]
 frozen_code={str(path):gate.sha(path) for path in code}
 e1path=BASE/'e1/e1full001_completion.json';e1s=BASE/'e1/monitorverifye1full001/started.json';e1t=BASE/'e1/monitorverifye1full001/terminal.json'
 e1_score.validate_acceptance_supervisor(e1path,e1s,e1t);proof,files,panel,_natural=e1_score.recheck_acceptance(e1path)
 panel=inventory.build_dev_panel(panel,completion.read_jsonl(files['inputs']))
 records=completion.read_jsonl(gate.RUN/'records.jsonl');groups=inventory.inventory(panel,records)
 summary_path=Path(proof['files']['dataset_summary']['path']);meta=gate.read(summary_path)
 if gate.sha(summary_path)!=proof['files']['dataset_summary']['sha256'] or set(meta['sources'])!=set(inventory.TASKS):raise ValueError('dataset source identity')
 needed={(row['task'],str(row['qid'])) for row in panel}
 if len(needed)!=384:raise ValueError('DEV qid inventory')
 args.out.mkdir(parents=True,exist_ok=False)
 frozen={'stage':'selected full DEV scoring','scorer_prelaunch':approved,'plan_sha256':PLAN_SHA256,'formal_proof_sha256':gate.sha(gate.PROOF),'review_proof_sha256':gate.sha(gate.RPROOF),'formal_terminal_sha256':gate.sha(gate.FT),'review_terminal_sha256':gate.sha(gate.REND),
         'e1_acceptance_sha256':gate.sha(e1path),'dataset_summary_sha256':gate.sha(summary_path),'source_code_sha256':frozen_code,'reference_tsv':meta['sources'],
         'runtime_versions':{n:importlib.metadata.version(n) for n in ('numpy','pandas','xlsxwriter','openpyxl')},'scope':'complete DEV only; non-DEV answer values not decoded'}
 with (args.out/'frozen.json').open('x') as f:json.dump(frozen,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 truths=e1_score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants();native=native_predictions(records,plan['configurations']);scored=[]
 for row in panel:
  key=(row['task'],str(row['qid']));result={'task':key[0],'qid':key[1],'group_id':row['group_id'],'arms':{}}
  for arm,record in groups[key].items():
   value=metrics.score(key[0],record['answer'],truths[key],author,constants,scoring_prediction=native[(key[0],key[1],arm)])
   if not math.isfinite(value['official']) or abs(value['official']-value['independent'])>1e-12:raise ValueError('author and independent score mismatch')
   result['arms'][arm]={'answer':record['answer'],'score':value['official'],'independent_score':value['independent'],'literal_answer_diagnostic':value['literal_answer_diagnostic'],
                        'normalized_prediction':value['normalized_prediction'],'eos':record['eos'],'truncated':record['truncated'],
                        **({'official_option':value['official_option'],'independent_option':value['independent_option']} if key[0]=='AI2D_TEST' else {})}
  scored.append(result)
 tasks={}
 for task in inventory.TASKS:
  sub=[r for r in scored if r['task']==task];rows=[{'group_id':r['group_id'],'scores':{a:r['arms'][a]['score'] for a in plan['configurations']},'answers':{a:r['arms'][a]['answer'] for a in plan['configurations']}} for r in sub]
  tasks[task]=paired.analyze(rows,plan['configurations'],plan['contrasts'],replicates=plan['bootstrap']['replicates'],seed=plan['bootstrap']['seed'])
  tasks[task]['terminal']={a:{'eos':sum(x['arms'][a]['eos'] for x in sub),'truncated':sum(x['arms'][a]['truncated'] for x in sub),'empty':sum(not x['arms'][a]['answer'].strip() for x in sub)} for a in plan['configurations']}
 for path,digest in approved['source_sha256'].items():
  if gate.sha(path)!=digest:raise ValueError('approved scoring source changed')
 for path,digest in frozen_code.items():
  if gate.sha(path)!=digest:raise ValueError('analysis source changed')
 for desc in meta['sources'].values():
  if gate.sha(desc['path'])!=desc['sha256']:raise ValueError('dataset source changed')
 for mapping in (evidence['formal']['files_sha256'],evidence['smoke']['files_sha256'],evidence['smoke_arrays']):
  if any(gate.sha(path)!=digest for path,digest in mapping.items()):raise ValueError('accepted evidence changed during scoring')
 scores=args.out/'scores.jsonl'
 with scores.open('x') as f:
  for row in scored:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 report={'status':'completed','n_questions':len(scored),'n_predictions':len(records),'tasks':tasks,'scores_sha256':gate.sha(scores),'plan_sha256':PLAN_SHA256,
         'limitations':['DEV exploratory; task and contrast intervals are not adjusted for multiplicity','AI2D uses deterministic local scoring without an API judge','full panel saves tensor summaries, not reconstructable full tensors','quality panel is not a timing benchmark']}
 with (args.out/'analysis.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'completed','questions':len(scored),'predictions':len(records)}))
if __name__=='__main__':main()
