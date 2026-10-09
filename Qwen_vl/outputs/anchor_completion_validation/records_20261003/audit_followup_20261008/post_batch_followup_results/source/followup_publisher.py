"""Separate CPU publisher for three followup groups, using the existing Git lock.
The frozen original publisher is imported unchanged for archive/Git operations.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=Path('/media/disk2/YZX/research/EADP_amp');BASE=ROOT/'Qwen_vl/outputs/audit_followup_20261008'
ORIGINAL=ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/publish_repair_results.py'
EXPECTED='ae9fc06bcb69dc20c2e2987badda79a95474019fa38c1ef58f9a7eed7a477d57'
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Original publisher SHA drift')
pub=module(ORIGINAL,'frozen_original_publication_helpers');ctrl=module(OUT/'followup_controller.py','followup_cpu_controller')
LOCAL=OUT/'group_publish';LOCAL.mkdir(exist_ok=True);pub.LOCAL=LOCAL;pub.STATE=LOCAL/'publish_state.json'
pub.DEST=Path('Qwen_vl/outputs/anchor_completion_validation/records_20261003/audit_followup_20261008/post_batch_followup_results')
pub.ALLOWLIST=[]

def metadata_snapshot(plan,state):
 history=dict(state.get('published_groups',{}))
 for p in plan['available_groups']:history[p['group_id']]=dict(identity=p['identity'],scores=p['manifest']['scores'])
 files={};sources={}
 for p in sorted(OUT.glob('*.py'))+sorted(OUT.glob('*.json')):
  if any(w in p.name for w in ['state','finished','loaded_model','registration']):continue
  data=p.read_bytes();relative=str(pub.DEST/'source'/p.name);files[relative]=data;sources[relative]=pub.digest(data)
 for p in sorted((BASE/'submission_tables_20261009').glob('latest_tables.*')):
  data=p.read_bytes();relative=str(pub.DEST/'tables'/p.name);files[relative]=data;sources[relative]=pub.digest(data)
 for p in [OUT.parent/'three_row_audit.json',OUT.parent/'three_row_audit.md']:
  if p.exists():data=p.read_bytes();relative=str(pub.DEST/'audit'/p.name);files[relative]=data;sources[relative]=pub.digest(data)
 files[str(pub.DEST/'README.md')]=('Post-batch predeclared K128 controls. Original 75 arms/55 groups are unchanged.\nNeXT GQA pairs a new EADP arm with the completed original AZ128 under the same beta2 recipe.\nVizWiz uses two new full arms per model under predeclared public defaults alpha0/beta1; not verified paper argv. Official leave-one-out remains primary; no score-driven selection.\n'+json.dumps(history,indent=2)).encode()
 return pub.digest(json.dumps(sources,sort_keys=True).encode()),files,history
pub.metadata_snapshot=metadata_snapshot

def packages(plan,state):
 queue=pub.read(ctrl.STATE);available=[];pending=[]
 for group in plan['groups']:
  key=group['group_id']
  if key in state.get('published_groups',{}):continue
  if not all(queue.get('jobs',{}).get(k,{}).get('status')=='complete' for k in group['job_ids']):pending.append(key);continue
  verified=[ctrl.validate_job(plan['jobs'][k],plan)for k in group['job_ids']]
  if group['kind']=='reused_pair':
   original_worker=module(BASE/'streamwait_continuation_20261008/legacy_streamwait_worker.py','original_reuse_validator')
   verified.append(original_worker.validate_finished(pub.read(plan['reuse']['job']['control_protocol'])))
  ctrl.fairness(group,plan['jobs']);files={};scores=[]
  for report in verified:
   for filename,expected in report['files'].items():
    path=Path(filename)
    if pub.sha(path)!=expected:raise ValueError('Publication source drift '+filename)
    # No weights, images, private state, user drivers or account files.
    if path.suffix not in ['.py','.json','.jsonl'] or path.name in ['watch_state.json','publish_state.json']:continue
    relative=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else 'external/'+pub.digest(filename.encode())[:12]+'/'+path.name
    files[relative]=path.read_bytes()
   s=report['score'];scores.append(dict(model=s['model'],task=s['task'],method=s['method'],budget=s['budget'],n=s['n'],metric=s['metric'],accuracy=s['value'],prediction_sha256=s['prediction_sha256'],protocol_sha256=s['control_protocol_sha256']))
  statistics=module(OUT/'pair_statistics.py','paired_cpu_statistics').compare(verified)
  files['PAIRED_STATISTICS.json']=json.dumps(statistics,indent=2).encode()
  (LOCAL/(key+'.paired_statistics.json')).write_text(json.dumps(statistics,indent=2))
  files['FOLLOWUP_POLICY.json']=(OUT/'POLICY.json').read_bytes();files['PAIR_IDENTITY.json']=json.dumps(ctrl.fairness(group,plan['jobs']),indent=2).encode()
  if group['task']=='vizwiz':
   auxiliary=vizwiz_released_audit(verified);files['RELEASED_SCORER_AUDIT.json']=json.dumps(auxiliary,indent=2).encode()
  available.append(pub.bundle(key,files,scores))
 return dict(available_groups=available,pending_groups=pending,common_audit=None,expected_groups=[g['group_id']for g in plan['groups']],all_experiments_complete=queue.get('status')=='complete' and not ctrl.process(queue.get('supervisor_pid'))['alive'])

def vizwiz_released_audit(reports):
 import ast,contextlib,io
 release=Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official/LLaVA/scripts/eval_vizwiz.py')
 if pub.sha(release)!='6d92bfefff9fbea9cb49ea85b63ce6ec6d1fd57dba4eed08d26cceef86179b26':raise ValueError('Released scorer drift')
 evaluator=module(ROOT/'LLaVA/llava/eval/m4c_evaluator.py','cpu_release_normalizer')
 tree=ast.parse(release.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
 rows=[]
 for r in reports:
  protocol=r['protocol'];pred=Path(protocol['output_paths']['prediction']);score=r['score'];gt_path=protocol['origin']['score_data'][0]['path']
  from types import SimpleNamespace
  args=SimpleNamespace(annotation_file=gt_path,result_file=str(pred),visual_token_num=None,beta=None,alpha=None)
  ns=dict(parse_args=lambda:args,json=json,os=os,EvalAIAnswerProcessor=evaluator.EvalAIAnswerProcessor)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(release),'exec'),ns)
  capture=io.StringIO()
  with contextlib.redirect_stdout(capture):ns['main']()
  processor=evaluator.EvalAIAnswerProcessor();answers={r['question_id']:r['text'] for r in (json.loads(line)for line in pred.read_text().splitlines())};gt=pub.read(gt_path)
  if len(answers)!=4319 or set(answers)!=set(range(4319))or len(gt)!=4319:raise ValueError('Full auxiliary denominator differs')
  vals=[min(1.,sum(processor(answers[i])==processor(a['answer'])for a in item['answers'])/3.) for i,item in enumerate(gt)]
  value=sum(vals)/4319*100
  if 'Accuracy: '+format(value,'.2f')+'%' not in capture.getvalue()or 'Missing predictions: 0' not in capture.getvalue():raise ValueError('Release main mismatch')
  rows.append(dict(method=protocol['method'],n=4319,official_loo=score['value'],released_min_matches_over3=value,released_main_stdout=capture.getvalue(),prediction_sha256=pub.sha(pred),normalizer_sha256=pub.sha(ROOT/'LLaVA/llava/eval/m4c_evaluator.py')))
 return dict(source=str(release),source_sha256=pub.sha(release),rows=rows,note='Auxiliary released formula, not proven paper scorer; official LOO remains primary.')

def run(once=False):
 own=(LOCAL/'watch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=pub.read(pub.STATE)if pub.STATE.exists()else dict(created_utc=pub.now(),published_groups={},remote_head=pub.remote_head())
 while True:
  try:
   plan=pub.read(ctrl.PLAN);ctrl.verify(plan)
   shared=(BASE/'rerun_batch/group_publish/publish.lock').open('a')
   try:
    fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=pub.finish_pending(state);package=packages(plan,state)
    if package['available_groups']:
     subprocess.run([sys.executable,str(OUT/'refresh_tables.py')],check=True,cwd=ROOT)
    state['expected_groups']=package['expected_groups']
    if package['available_groups'] or state.get('published_groups'):state=pub.publish(package,state)
    else:state.update(status='waiting_for_complete_pair',pending_groups=package['pending_groups'],updated_utc=pub.now());pub.atomic(pub.STATE,state)
    if pub.publication_complete(package,state):state.update(status='complete',finished_utc=pub.now());pub.atomic(pub.STATE,state);return
   finally:shared.close()
  except BlockingIOError:
   state.update(status='waiting_for_original_publisher_lock',updated_utc=pub.now());pub.atomic(pub.STATE,state)
  except Exception as exc:
   state.update(status='failed_retryable',error=str(exc),updated_utc=pub.now());pub.atomic(pub.STATE,state);traceback.print_exc()
  if once:return
  time.sleep(30)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--once',action='store_true');a.add_argument('--watch',action='store_true');args=a.parse_args();run(args.once)
