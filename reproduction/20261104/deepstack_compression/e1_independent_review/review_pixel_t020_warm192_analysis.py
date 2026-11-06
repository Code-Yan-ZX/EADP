"""Recalculate all paired one-token timings and group bootstrap after acceptance."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';E=B/'e6/pixel_wavefront_t020_warm192_extra_checks';I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 if a.out.resolve()!=(I/'pixel_t020_warm192_analysis_review.json').resolve():raise ValueError('output')
 formal=D/'verification001.json';proof=I/'pixel_t020_warm192_result_review.json';analysis=D/'analysis001.json'
 terminal(D/'monitor_verify001',E/'formal_v3.py',formal);terminal(I/'monitor_pixel_t020_warm192_result',I/'review_pixel_t020_warm192_result.py',proof);terminal(D/'monitor_analysis001',E/'analysis_v3.py',analysis)
 p=read(D/'analysis_plan001.json');f=read(formal);r=read(proof);value=read(analysis)
 if p.get('draws')!=10000 or p.get('bootstrap_seed')!=20261105 or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('formal_sha256')!=sha(formal) or f.get('status')!='passed' or f.get('measured')!=384 or value.get('analysis_plan_sha256')!=sha(D/'analysis_plan001.json') or value.get('formal_sha256')!=sha(formal) or value.get('independent_result_sha256')!=sha(proof):raise ValueError('source proof')
 pairs={}
 for item in f['records']:
  if item['phase']!='measure':continue
  x=item['record'];k=(x['task'],str(x['qid']),x['arm'])
  if k in pairs:raise ValueError('duplicate')
  pairs[k]=(x,item)
 tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');expected_rows=[];rng=np.random.default_rng(20261105)
 for task in tasks:
  keys=sorted({(k[0],k[1]) for k in pairs if k[0]==task},key=lambda k:(pairs[(k[0],k[1],'author')][0]['group_id'],k[1]))
  if len(keys)!=64 or len({pairs[(t,q,'author')][0]['group_id'] for t,q in keys})!=64:raise ValueError('task denominator')
  rows=[]
  for t,q in keys:
   aa,aa_meta=pairs[(t,q,'author')];bb,bb_meta=pairs[(t,q,'wavefront')]
   if aa['group_id']!=bb['group_id']:raise ValueError('pair group')
   av=aa['boundaries']['first_token_selected_ns']-aa['boundaries']['t0_before_image_read_ns'];bv=bb['boundaries']['first_token_selected_ns']-bb['boundaries']['t0_before_image_read_ns']
   if not (0<av and 0<bv):raise ValueError('duration')
   row={'task':t,'qid':q,'group_id':aa['group_id'],'author_ns':av,'wavefront_ns':bv,'difference_ns':bv-av,'ratio':bv/av,
        'first_token_equal':aa['first_token_id']==bb['first_token_id'],'eos_ids_equal':aa['eos_ids']==bb['eos_ids'],
        'first_eos_equal':aa['first_token_is_eos']==bb['first_token_is_eos'],'ordered_selection_equal':aa['selected_indices']==bb['selected_indices'],
        'author_selected_count':len(aa['selected_indices']),'wavefront_selected_count':len(bb['selected_indices']),
        'author_observation':{'process_order':aa_meta['process_order'],'block':aa_meta['block'],'monitoring':aa_meta['monitoring']},
        'wavefront_observation':{'process_order':bb_meta['process_order'],'block':bb_meta['block'],'monitoring':bb_meta['monitoring']}}
   rows.append(row)
  expected_rows+=rows;diff=np.array([x['difference_ns'] for x in rows],dtype=float);rat=np.array([x['ratio'] for x in rows],dtype=float)
  draws=rng.integers(0,64,size=(10000,64));actual=value['tasks'][task]
  expected={'images':64,'author_median_ns':float(np.median([x['author_ns'] for x in rows])),
            'wavefront_median_ns':float(np.median([x['wavefront_ns'] for x in rows])),
            'paired_difference_median_ns':float(np.median(diff)),
            'paired_difference_bootstrap_95_ns':[float(x) for x in np.quantile(np.median(diff[draws],axis=1),[.025,.975])],
            'paired_ratio_median':float(np.median(rat)),
            'paired_ratio_bootstrap_95':[float(x) for x in np.quantile(np.median(rat[draws],axis=1),[.025,.975])],
            'first_token_mismatches':sum(not x['first_token_equal'] for x in rows),'eos_ids_mismatches':sum(not x['eos_ids_equal'] for x in rows),
            'first_eos_mismatches':sum(not x['first_eos_equal'] for x in rows),'ordered_selection_mismatches':sum(not x['ordered_selection_equal'] for x in rows)}
  if actual!=expected:raise ValueError('paired/bootstrap mismatch '+task)
 if value.get('per_question')!=expected_rows or value.get('pairs')!=192 or value.get('observations')!=384 or value.get('draws')!=10000 or value.get('seed')!=20261105:raise ValueError('full output inventory')
 output={'status':'passed','independent_review':True,'pairs':192,'observations':384,'questions_per_task':64,'GT_opened':False,'scoring_performed':False,'errors':[],
         'analysis_sha256':sha(analysis),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),
         'formal_sha256':sha(formal),'independent_result_sha256':sha(proof),'reviewer_sha256':sha(__file__),'numpy_version':np.__version__,
         'limitations':['post-hoc DEV paired one-token timing on one GPU; no independent confirmatory quality conclusion','paired bootstrap uncorrected; no universal steady-state inference','manual/unsampled background and second fsync remain unknown']}
 with a.out.open('x') as dest:json.dump(output,dest,indent=2);dest.write('\n');dest.flush();os.fsync(dest.fileno())
if __name__=='__main__':main()
