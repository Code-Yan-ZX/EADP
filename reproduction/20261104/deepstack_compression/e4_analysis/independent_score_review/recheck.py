"""Independent DEV-only recomputation of E4 384x8 author and check scores."""
import ast,copy,csv,hashlib,io,json,math,os,string,sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent
sys.path.insert(0,str(BASE/'e1_analysis'))
import metrics
SCORE=BASE/'e4_analysis/score_panel003';RUN=BASE/'e4_panel_v2/panel_003';PANEL=BASE/'e4_panel_v2/panel_003_verification.json';REVIEW=BASE/'e1_independent_review/e4panel003_review.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f]
def exited(path):
 t=read(path)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('supervisor incomplete: '+str(path))
def truth_by_qid(sources,needed):
 """Filter bytes by frozen DEV qid before decoding/parsing answer fields."""
 csv.field_size_limit(100_000_000);out={}
 for task,meta in sources.items():
  path=Path(meta['path'])
  if sha(path)!=meta['sha256']:raise ValueError('reference source changed')
  wanted={qid.encode() for t,qid in needed if t==task}
  with path.open('rb') as f:
   header=next(csv.reader([f.readline().decode()],delimiter='\t'));idx=header.index('index')
   if idx not in (0,len(header)-1):raise ValueError('qid not on edge')
   for raw in f:
    key=(raw.split(b'\t',1)[0] if idx==0 else raw.rstrip(b'\r\n').rsplit(b'\t',1)[-1]).strip(b'"')
    if key not in wanted:continue
    fields=next(csv.reader([raw.decode()],delimiter='\t'))
    if len(fields)!=len(header):raise ValueError('invalid DEV TSV row')
    row=dict(zip(header,fields));k=(task,row['index'])
    if k in out or k not in needed:raise ValueError('duplicate or foreign reference')
    if task=='AI2D_TEST':out[k]={'answer':row['answer'],'choices':{a:row[a] for a in string.ascii_uppercase if a in row and row[a] not in metrics.NATIVE_NA}}
    else:
     try:value=ast.literal_eval(row['answer'])
     except (ValueError,SyntaxError):value=row['answer']
     out[k]={'answers':[str(x) for x in value] if isinstance(value,list) else [str(value)]}
 if set(out)!=needed:raise ValueError('DEV GT inventory mismatch')
 return out

def main():
 panel=read(PANEL);review=read(REVIEW);frozen=read(SCORE/'frozen.json');analysis=read(SCORE/'analysis.json')
 if panel.get('status')!='passed' or panel.get('all_arms_validated')!=3072 or panel.get('GT_opened') is not False:raise ValueError('panel proof failed')
 if review.get('status')!='passed' or review.get('n_verified')!=384 or review.get('arms_verified')!=3072 or review.get('formal_proof_sha256')!=sha(PANEL) or review.get('GT_opened') is not False:raise ValueError('panel independent review failed')
 if frozen['panel_proof_sha256']!=sha(PANEL) or frozen['review_proof_sha256']!=sha(REVIEW) or analysis.get('status')!='completed' or analysis.get('scores_sha256')!=sha(SCORE/'scores.jsonl'):raise ValueError('score proof mismatch')
 for path in (BASE/'e4_panel_v2/monitor_verify_panel003/terminal.json',BASE/'e1_independent_review/monitor_e4panel003/terminal.json',BASE/'e4_analysis/monitor_score_panel003/terminal.json'):exited(path)
 for path,digest in panel['files_sha256'].items():
  if sha(path)!=digest:raise ValueError('panel input changed')
 for path,digest in frozen['source_code_sha256'].items():
  if sha(path)!=digest:raise ValueError('score source changed')
 records=lines(RUN/'records.jsonl');scored=lines(SCORE/'scores.jsonl')
 if len(records)!=3072 or len(scored)!=384:raise ValueError('wrong inventory')
 byqid={(r['task'],str(r['qid'])):r for r in scored}
 if len(byqid)!=384:raise ValueError('duplicate scored question')
 needed=set(byqid);truth=truth_by_qid(frozen['reference_tsv'],needed)
 grouped={}
 for r in records:
  key=(r['task'],str(r['qid']),'K'+str(r['budget'])+'-D'+str(int(r['ds_on'])))
  if key in grouped or key[:2] not in needed or r.get('forced') is not False:raise ValueError('unexpected arm')
  grouped[key]=r
 if len(grouped)!=3072 or any(sum(k[:2]==q for k in grouped)!=8 for q in needed):raise ValueError('missing arm')
 author=metrics.author_functions();constants=metrics._author_constants();checked=0;counts={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  for arm in ('K128-D0','K128-D1','K256-D0','K256-D1','K512-D0','K512-D1','K1024-D0','K1024-D1'):
   subset=[r for r in records if r['task']==task and 'K'+str(r['budget'])+'-D'+str(int(r['ds_on']))==arm]
   if len(subset)!=128:raise ValueError('task-arm denominator')
   stream=io.BytesIO();pd.DataFrame({'index':range(len(subset)),'prediction':[r['answer'] for r in subset]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);preds=pd.read_excel(stream)['prediction']
   for r,value in zip(subset,preds):
    q=(task,str(r['qid']));expected=byqid[q]['arms'][arm];raw=r['answer'];pred=str(value);gt=truth[q]
    if expected['answer']!=raw or expected['eos']!=r['eos'] or expected['truncated']!=r['truncated'] or byqid[q]['group_id']!=r['group_id']:raise ValueError('score record differs from prediction')
    if task=='AI2D_TEST':
     official=float((author['can_infer'](pred,copy.deepcopy(gt['choices'])) or 'Z')==gt['answer'])
     independent=float((metrics.independent_option(pred,gt['choices']) or 'Z')==gt['answer'])
    else:
     method='anls' if task=='DocVQA_VAL' else 'vqa_score';line={'answer':repr(gt['answers']),'prediction':pred}
     official=float(author['hit_calculate']([author['process_line'](line,method=method)],task)[0])
     independent=metrics.independent_doc(pred,gt['answers']) if task=='DocVQA_VAL' else metrics.independent_text(pred,gt['answers'],constants)
    if not math.isfinite(official) or abs(official-independent)>1e-12 or abs(official-expected['score'])>1e-12 or abs(independent-expected['independent_score'])>1e-12:raise ValueError('score mismatch '+task+' '+q[1]+' '+arm)
    checked+=1;counts[task]=counts.get(task,0)+1
 proof={'status':'passed','questions':len(needed),'arms_recomputed':checked,'per_task_arms':counts,'errors':[],
        'scope':'384 DEV only; qid byte filter before answer decoding; no CONFIRM/FINAL predictions or answer parsing',
        'score_sha256':sha(SCORE/'scores.jsonl'),'analysis_sha256':sha(SCORE/'analysis.json'),'panel_proof_sha256':sha(PANEL),'review_proof_sha256':sha(REVIEW),'source_sha256':sha(__file__),
        'limitation':'author source functions and previously independent E1 metric implementations are reused; paired/bootstrap audit performed separately'}
 out=HERE/'verification001.json'
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','arms_recomputed':checked}))
if __name__=='__main__':main()
