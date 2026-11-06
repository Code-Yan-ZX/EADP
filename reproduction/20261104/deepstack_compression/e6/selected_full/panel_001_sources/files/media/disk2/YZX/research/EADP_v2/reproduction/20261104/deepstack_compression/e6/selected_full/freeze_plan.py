"""Freeze all admitted E1 DEV inputs and prior Div/CD identity evidence; no GT."""
import collections,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];SMOKE=HERE.parent/'selected_smoke';REVIEW=ROOT/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def build():
 paths=[ROOT/'e1/e1full001_completion.json',SMOKE/'smoke_001_verification.json',SMOKE/'monitor_verify001/started.json',
        SMOKE/'monitor_verify001/terminal.json',REVIEW/'selected_smoke001_review.json',
        REVIEW/'monitor_selected_smoke001/started.json',REVIEW/'monitor_selected_smoke001/terminal.json',
        SMOKE/'smoke_001_sources/manifest.json',ROOT/'initial_plan.json']
 e,sp,ss,st,rev,rs,rt,manifest,initial=(json.loads(p.read_text()) for p in paths)
 if (e.get('status')!='passed' or e.get('natural_questions')!=384 or e.get('GT_opened') is not False or
     sp.get('status')!='passed' or sp.get('questions')!=3 or sp.get('generations')!=16 or sp.get('pairs')!=8 or
     sp.get('GT_opened') is not False or sp.get('scoring') is not False or rev.get('status')!='passed' or
     any(x.get('reason')!='completed' or x.get('worker_returncode')!=0 or x.get('remaining_process_group')!=[] or
         x.get('worker_pid')!=y.get('worker_pid') for x,y in ((st,ss),(rt,rs)))):
  raise ValueError('upstream identity not independently complete')
 inputs=Path(e['files']['inputs']['path'])
 if sha(inputs)!=e['files']['inputs']['sha256']:raise ValueError('E1 input manifest changed')
 rows=[json.loads(s) for s in inputs.read_text().splitlines()]
 if (len(rows)!=384 or collections.Counter(x['task'] for x in rows)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} or
     len({x['group_id'] for x in rows})!=384 or len({(x['task'],str(x['qid'])) for x in rows})!=384):raise ValueError('unexpected DEV inventory')
 if initial.get('budget_end_unix')!=1794065414:raise ValueError('deadline changed')
 return {'stage':'prospective Div/CD full DEV panel, approval absent','inputs_path':str(inputs),'inputs_sha256':sha(inputs),
         'questions':384,'groups':384,'tasks':{'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128},
         'methods':['div','cd'],'paths':['author_stock','adapted_p3_ds'],'generation_count':1536,
         'recipe':'E1 frozen natural greedy; original Div/CD K256 selection',
         'generation_controls_reference':'per-method and per-task accepted smoke natural stock/adapter settings; checked before each generate',
         'previous_pre_controls_plan_sha256':sha(HERE/'plan_001_pre_controls.json'),
         'storage':'actual IDs, positions, runtime selector/DS checks, tensor hashes and outputs; full tensors only in prior smoke',
         'inner_timeout_seconds':10800,'outer_timeout_seconds':10980,'progress_stale_seconds':1800,
         'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90,'minimum_free_bytes':32212254720,
         'budget_end_unix':initial['budget_end_unix'],'upstream_sha256':{str(p):sha(p) for p in paths},
         'gt_opened':False,'scoring':False,'gpu_authorized':False}
if __name__=='__main__':
 with (HERE/'plan.json').open('x') as f:json.dump(build(),f,indent=2);f.write('\n')
