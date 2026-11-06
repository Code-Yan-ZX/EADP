"""Independently reconstruct matched DEV index overlap; no GT or scoring."""
import argparse,collections,hashlib,json,math,os,statistics
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixel_t010_t020_selection'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor identity')
def panel(tag,plan):
 analysis=B/f'e6/pixelprune_{tag}_accept';review_path=R/f'pixel_{tag}_dev384_result_review.json';review=read(review_path);formal=read(analysis/'verification001.json');reviewer=R/f'review_pixel_{tag}_dev384_result.py'
 term(analysis/'monitor_verify001',analysis/'verify.py');term(R/f'monitor_pixel_{tag}_dev384_result',reviewer)
 if sha(review_path)!=plan['reviews_sha256'][str(review_path)] or review['status']!='passed' or review['independent_review'] is not True or review['questions']!=384 or review['formal_sha256']!=sha(analysis/'verification001.json') or review['reviewer_sha256']!=sha(reviewer) or formal['status']!='passed' or formal['questions']!=384 or formal['GT_opened'] is not False:raise ValueError('formal or independent evidence')
 obs=B/f'e6/pixelprune_{tag}_dev384/run_smoke001/observations';result={}
 for q in formal['per_question']:
  path=obs/(q['task']+'_'+str(q['qid'])+'.json');digest=sha(path)
  if formal['files_sha256'].get(str(path))!=digest or review['per_question_files_sha256'].get(str(path))!=digest:raise ValueError('observation hash')
  x=read(path);ids=x['selected_indices'];key=(x['task'],str(x['qid']),x['group_id'],x['image_sha256'],x['message_sha256'])
  if x['task']!=q['task'] or str(x['qid'])!=str(q['qid']) or key in result or x['status']!='completed' or not ids or ids!=sorted(set(ids)) or len(ids)!=x['selected_count'] or any(type(i) is not int or not 0<=i<1024 for i in ids):raise ValueError('indices or identities invalid')
  result[key]=frozenset(ids)
 if len(result)!=384:raise ValueError('panel incomplete')
 return result
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_t010_t020_selection_review.json' and not a.out.exists()
term(D/'monitor_analysis001',D/'analyze.py');plan=read(D/'plan001.json');freeze=read(D/'source_freeze001.json');result=read(D/'result001.json')
assert freeze['plan_sha256']==result['plan_sha256']==sha(D/'plan001.json') and freeze['script_sha256']==result['script_sha256']==sha(D/'analyze.py') and result['status']=='completed' and result['GT_opened'] is False and result['scoring_performed'] is False and result['mixed_prediction_files_opened'] is True
one=panel('t010',plan);two=panel('t020',plan);assert set(one)==set(two)
for task in plan['tasks']:
 keys=[k for k in one if k[0]==task];assert len(keys)==128 and len({k[2] for k in keys})==128
 diff=[len(one[k])-len(two[k]) for k in keys];missing=sum(bool(two[k]-one[k]) for k in keys);jac=[len(one[k]&two[k])/len(one[k]|two[k]) for k in keys];r=result['results'][task]
 assert r['questions']==128 and r['k010_minus_k020_min_median_max']==[min(diff),float(statistics.median(diff)),max(diff)] and r['k010_less_than_k020']==sum(x<0 for x in diff) and r['t020_not_subset_of_t010']==missing
 assert all(math.isclose(a,b,abs_tol=1e-15) for a,b in zip(r['jaccard_min_median_max'],[min(jac),statistics.median(jac),max(jac)]))
proof={'status':'passed','independent_review':True,'questions':384,'tasks':3,'GT_opened':False,'scoring_performed':False,'mixed_prediction_files_opened':True,'prediction_scope':'parsed previously accepted DEV observations only for input identities and selected token indices','plan_sha256':sha(D/'plan001.json'),'script_sha256':sha(D/'analyze.py'),'result_sha256':sha(D/'result001.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'reviewer_sha256':sha(__file__),'limitations':plan['limitations'],'errors':[]}
with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
