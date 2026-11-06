"""Freeze already-selected E4 native DEV anchors; never open predictions or GT."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def plan():
 native=ROOT/'e4_native/native_001/frozen.json';native_proof=ROOT/'e4_native/native_001_verification.json'
 review=ROOT/'e1_independent_review/e4native001_review.json';terminal=ROOT/'e1_independent_review/monitor_e4native001/terminal.json'
 acceptance=ROOT/'e1/e1full001_completion.json';budget=ROOT/'initial_plan.json'
 n=json.loads(native.read_text());e=json.loads(acceptance.read_text());t=json.loads(terminal.read_text())
 if (json.loads(native_proof.read_text()).get('status')!='passed' or json.loads(review.read_text()).get('status')!='passed' or
     t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     e.get('status')!='passed' or e.get('GT_opened') is not False):raise ValueError('prior acceptance incomplete')
 inputs=Path(e['files']['inputs']['path'])
 if sha(inputs)!=e['files']['inputs']['sha256']:raise ValueError('input inventory changed')
 anchors=n['anchors'];wanted=['DocVQA_VAL','TextVQA_VAL','AI2D_TEST']
 if [x['task'] for x in anchors]!=wanted or len({x['group_id'] for x in anchors})!=3:raise ValueError('wrong native anchor inventory')
 rows=[json.loads(s) for s in inputs.read_text().splitlines()]
 for anchor in anchors:
  matches=[r for r in rows if r.get('task')==anchor['task'] and str(r.get('qid'))==str(anchor['qid'])]
  if len(matches)!=1 or any(str(matches[0].get(k))!=str(v) for k,v in anchor.items()):raise ValueError('anchor not in E1 inputs')
  if sha(matches[0]['image_path'])!=anchor['image_sha256']:raise ValueError('anchor image changed')
 initial=json.loads(budget.read_text())
 if (initial['budget_start_unix'],initial['budget_end_unix'])!=(1793806214,1794065414):raise ValueError('wrong user budget')
 return {'purpose':'Div/CD stock selector and P3/DS adapter diagnostic; no GT or scoring',
   'anchors':anchors,'methods':['div','cd'],'natural_generations':12,'forced_generations':4,
   'fixed_prefix_task':'DocVQA_VAL','fixed_decode_steps':32,'max_natural_tokens':2048,
   'stock_vs_adapter':'processor, visual streams, author-selected rows and generation scalars; no cross-position output identity claim',
   'inner_timeout_seconds':7200,'outer_timeout_seconds':7380,'progress_stale_seconds':1800,
   'poll_seconds':15,'grace_seconds':90,'snapshot_seconds':3600,'minimum_free_bytes':32212254720,
   'budget_end_unix':initial['budget_end_unix'],'budget_source_sha256':sha(budget),
   'inputs_sha256':sha(inputs),'upstream_sha256':{str(p):sha(p) for p in (native,native_proof,review,terminal,acceptance)},
   'gt_read':False,'gpu_authorized':False}
if __name__=='__main__':
 out=HERE/'plan.json'
 with out.open('x') as f:json.dump(plan(),f,indent=2);f.write('\n')
