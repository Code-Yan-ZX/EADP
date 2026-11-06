import argparse,collections,datetime,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');END=1794065414
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(D/'core.py')!='82ca6da3e3959d0193ef24e04453559952c92b3c897e0aaeddad8d3dc2e83832' or 'core' in sys.modules:raise ValueError('core source')
import core
if Path(core.__file__).resolve()!=(D/'core.py').resolve():raise ValueError('core path')
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script):
 a=read(mon/'started.json');b=read(mon/'terminal.json');cwd=Path(a['cwd'])
 if a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in a['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor')
def main():
 if datetime.datetime.now(datetime.timezone.utc).timestamp()>=END:raise ValueError('budget expired')
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=(D/'result001.json').resolve() or a.out.exists() or (D/'rows001.jsonl').exists():raise ValueError('fixed output')
 plan=read(D/'plan001.json');cpu=read(D/'cpu_proof001.json')
 if read(B/'initial_plan.json')['budget_end_unix']!=END or plan['status']!='frozen' or plan['GT_opened'] is not False or plan['scores_parsed'] is not False or plan['individual_dimensions_read'] is not False or importlib.metadata.version('Pillow')!=plan['pillow_version'] or cpu.get('status')!='passed' or cpu.get('plan_sha256')!=sha(D/'plan001.json') or cpu.get('source_sha256')!=plan['source_sha256'] or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan or source')
 terminal(D/'monitor_cpu001',D/'cpu_check.py');terminal(B/'e1/monitorverifye1full001',B/'e1_analysis/completion.py');terminal(I/'monitor_e1full001',I/'review_full.py');terminal(I/'monitor_rres_k256_dev384_result',I/'review_rres_k256_dev384_result.py');terminal(I/'monitor_rres_k256_dev384_scores',I/'review_rres_k256_dev384_scores.py');terminal(B/'e6/rres_k256_full_analysis_v2/monitor_score001',B/'e6/rres_k256_full_analysis_v2/score.py')
 ef=read(B/'e1/e1full001_completion.json');er=read(I/'e1full001_review.json');rf=read(B/'e6/rres_k256_dev384/verification001.json');rr=read(I/'rres_k256_dev384_result_review.json');rs=read(I/'rres_k256_dev384_scores_review.json');inputs=Path(ef['files']['inputs']['path']);scores=B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl'
 if ef.get('status')!='passed' or ef.get('natural_questions')!=384 or ef.get('GT_opened') is not False or er.get('status')!='passed' or er.get('n_verified')!=384 or er.get('errors')!=[] or er.get('completion_proof_sha256')!=sha(B/'e1/e1full001_completion.json') or er.get('reviewer_sha256')!=sha(I/'review_full.py') or ef['files']['inputs']['sha256']!=sha(inputs) or ef['files']['input_proof']['sha256']!=sha(ef['files']['input_proof']['path']):raise ValueError('E1 input source')
 if rf.get('status')!='passed' or rf.get('images')!=384 or rf.get('GT_opened') is not False or rr.get('status')!='passed' or rr.get('independent_review') is not True or rr.get('images')!=384 or rr.get('errors')!=[] or rr.get('formal_sha256')!=sha(B/'e6/rres_k256_dev384/verification001.json') or rr.get('reviewer_sha256')!=sha(I/'review_rres_k256_dev384_result.py') or rs.get('status')!='passed' or rs.get('independent_review') is not True or rs.get('questions')!=384 or rs.get('errors')!=[] or rs.get('score_sha256')!=sha(scores) or rs.get('result_review_sha256')!=sha(I/'rres_k256_dev384_result_review.json') or rs.get('reviewer_sha256')!=sha(I/'review_rres_k256_dev384_scores.py'):raise ValueError('Rres proofs')
 design=read(B/'e6/rres_k256_dev384/design001.json')['anchors'];items=lines(inputs);scored=lines(scores);di=core.unique(design);ii=core.unique(items);si=core.unique(scored)
 if any(len(x)!=384 for x in (design,items,scored,di,ii,si)) or not set(di)==set(ii)==set(si) or collections.Counter(k[0] for k in di)!={t:128 for t in TASKS}:raise ValueError('384 matched input/score inventory')
 output=[]
 for k,a in di.items():
  source=ii[k];s=si[k]
  if any(source.get(n)!=a.get(n) for n in ('image_sha256','message_sha256')):raise ValueError('input hash binding')
  w,h=core.dimensions(Path(source['image_path']).read_bytes(),a['image_sha256']);aspect,area=core.groups(w,h);r=s['rres'];e=s['eadp']
  if abs(r['score']-r['independent_score'])>1e-12 or abs(e['score']-e['independent_score'])>1e-12:raise ValueError('score identity')
  output.append({'task':k[0],'qid':k[1],'group_id':k[2],'image_sha256':a['image_sha256'],'message_sha256':a['message_sha256'],'width':w,'height':h,'aspect':aspect,'area':area,'rres':r['score'],'eadp':e['score']})
 task={t:core.summary([x for x in output if x['task']==t]) for t in TASKS}
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source changed')
 path=D/'rows001.jsonl'
 with path.open('x') as f:
  for x in output:f.write(json.dumps(x,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 result={'status':'passed','questions':384,'tasks':task,'rows_sha256':sha(path),'plan_sha256':sha(D/'plan001.json'),'GT_opened':False,'existing_predictions_opened':True,'errors':[],'limits':['Posthoc descriptive fixed bins; multiple comparisons; no causal or routing claims.','Encoded raw image dimensions, EXIF orientation not applied.','Different input resolution; same K is not equal compute.','Six runtime kwargs/EOS checked online; full effective generation config not separately archived.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
