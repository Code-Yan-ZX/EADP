"""Rebuild all geometry groups from accepted inputs and scores without GT."""
import argparse,collections,hashlib,io,json,math,os
from pathlib import Path
from PIL import Image
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_image_geometry_v2';TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def clean(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervision '+str(mon))
def index(items):
 result={}
 for x in items:
  k=(x['task'],str(x['qid']),x['group_id'])
  if k in result:raise ValueError('duplicate')
  result[k]=x
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_eadp_image_geometry_v2_review.json'
 if a.out.resolve()!=dest or dest.exists():raise ValueError('output')
 plan=read(D/'plan001.json');result=read(D/'result001.json');clean(D/'monitor_freeze001',D/'freeze.py');clean(D/'monitor_cpu001',D/'cpu_check.py');clean(D/'monitor_run001',D/'run.py',D/'result001.json')
 if plan['status']!='frozen' or plan['GT_opened'] is not False or plan['scores_parsed'] is not False or plan['individual_dimensions_read'] is not False or result['status']!='passed' or result['questions']!=384 or result['GT_opened'] is not False or result['plan_sha256']!=sha(D/'plan001.json') or result['rows_sha256']!=sha(D/'rows001.jsonl') or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan/results')
 ef=B/'e1/e1full001_completion.json';er=I/'e1full001_review.json';rf=B/'e6/rres_k256_dev384/verification001.json';rr=I/'rres_k256_dev384_result_review.json';sr=I/'rres_k256_dev384_scores_review.json';score=B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl'
 for mon,script in ((I/'monitor_e1full001',I/'review_full.py'),(I/'monitor_rres_k256_dev384_result',I/'review_rres_k256_dev384_result.py'),(I/'monitor_rres_k256_dev384_scores',I/'review_rres_k256_dev384_scores.py')):clean(mon,script)
 ep=read(ef);eq=read(er);rq=read(rr);sq=read(sr);inp=Path(ep['files']['inputs']['path'])
 if ep['status']!='passed' or ep['files']['inputs']['sha256']!=sha(inp) or eq['status']!='passed' or eq['completion_proof_sha256']!=sha(ef) or eq['reviewer_sha256']!=sha(I/'review_full.py') or rq['status']!='passed' or rq['formal_sha256']!=sha(rf) or rq['reviewer_sha256']!=sha(I/'review_rres_k256_dev384_result.py') or sq['status']!='passed' or sq['score_sha256']!=sha(score) or sq['result_review_sha256']!=sha(rr) or sq['reviewer_sha256']!=sha(I/'review_rres_k256_dev384_scores.py'):raise ValueError('accepted sources')
 design=index(read(B/'e6/rres_k256_dev384/design001.json')['anchors']);original=index(lines(inp));scored=index(lines(score));recorded=index(lines(D/'rows001.jsonl'))
 if not(len(design)==len(original)==len(scored)==len(recorded)==384 and set(design)==set(original)==set(scored)==set(recorded)) or collections.Counter(k[0] for k in design)!={t:128 for t in TASKS}:raise ValueError('input inventory')
 groups=collections.defaultdict(list)
 for k in sorted(design):
  d=design[k];o=original[k];s=scored[k];r=recorded[k]
  if any(d[n]!=o[n] or d[n]!=r[n] for n in ('image_sha256','message_sha256')):raise ValueError('identity')
  data=Path(o['image_path']).read_bytes()
  if sha_bytes(data)!=d['image_sha256']:raise ValueError('actual image bytes')
  with Image.open(io.BytesIO(data)) as im:w,h=im.size
  aspect='portrait' if w/h<.8 else ('landscape' if w/h>1.25 else 'middle');area='small' if w*h<1000000 else 'large'
  if (r['width'],r['height'],r['aspect'],r['area'])!=(w,h,aspect,area):raise ValueError('image geometry')
  values=(s['rres']['score'],s['eadp']['score'])
  if not all(math.isfinite(v) and 0<=v<=1 for v in values) or any(abs(s[z]['score']-s[z]['independent_score'])>1e-12 for z in ('rres','eadp')) or (r['rres'],r['eadp'])!=values:raise ValueError('score mismatch')
  groups[(k[0],'all','all')].append((k,values));groups[(k[0],'aspect',aspect)].append((k,values));groups[(k[0],'area',area)].append((k,values))
 for task in TASKS:
  for typ,labels in (('all',['all']),('aspect',['portrait','middle','landscape']),('area',['small','large'])):
   for label in labels:
    data=groups[(task,typ,label)];out=result['tasks'][task]['all'] if typ=='all' else result['tasks'][task][typ][label];n=len(data)
    if out['questions']!=n or out['image_groups']!=len({x[0][2] for x in data}):raise ValueError('group denominator')
    for field,ix in (('rres_mean_pct',0),('eadp_mean_pct',1)):
     expected=100*sum(x[1][ix] for x in data)/n if n else None
     if (expected is None and out[field] is not None) or (expected is not None and not math.isclose(out[field],expected,abs_tol=1e-10,rel_tol=0)):raise ValueError('group score')
    diff=100*sum(x[1][0]-x[1][1] for x in data)/n if n else None
    if (diff is None and out['difference_pp'] is not None) or (diff is not None and not math.isclose(out['difference_pp'],diff,abs_tol=1e-10,rel_tol=0)):raise ValueError('difference')
 proof={'status':'passed','independent_review':True,'questions':384,'GT_opened':False,'existing_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'plan001.json'),'result_sha256':sha(D/'result001.json'),'rows_sha256':sha(D/'rows001.jsonl'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'reviewer_sha256':sha(__file__),'limitations':['Posthoc DEV descriptive subgrouping; no causal, routing, equal-compute or speed claim.','EXIF orientation is not applied.','v2 code fix occurred after v1 processed data and failed writing output; original bins remain as preregistered.','R-res six generation kwargs/EOS checked online; full effective config not archived.']}
 with dest.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
