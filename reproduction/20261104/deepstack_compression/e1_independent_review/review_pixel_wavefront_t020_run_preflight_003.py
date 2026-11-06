"""Independent review admitting only t020 final run import preflight."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_wavefront_model_t020_run';ROOT=B.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def terminal(d,script,out):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=script.resolve() or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervisor identity')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
 if a.out.resolve()!=D/'independent_preflight_review_003.json':raise ValueError('output path')
 plan=D/'preflight_plan005.json';cpu=D/'cpu_check007/verification.json';mon=D/'monitor_cpu007';p=read(plan);c=read(cpu)
 terminal(mon,D/'cpu_validation.py',cpu.parent);terminal(D/'monitor_freeze005',D/'freeze_preflight.py',plan)
 actual={name:sha(name) for name in p['source_sha256']}
 if p['source_sha256']!=actual or c['source_sha256_before']!=actual or c['source_sha256_after']!=actual or c['status']!='passed' or c['tests_returncode']!=0 or 'Ran 31 tests' not in c['tests_stderr'] or c['GT_opened'] is not False or c['gpu_used'] is not False or c['model_loaded'] is not False or c['existing_DEV_predictions_opened'] is not True:raise ValueError('source/CPU evidence')
 if p['stage']!='six-arm identity execution no-GPU import and container preflight only' or p['threshold']!=0.2 or p['GT_opened'] is not False or p['gpu_authorized'] is not False or p['existing_DEV_predictions_opened'] is not True or p['cpu_sha256']!=sha(cpu) or p['cpu_started_sha256']!=sha(mon/'started.json') or p['cpu_terminal_sha256']!=sha(mon/'terminal.json') or p['budget_source_sha256']!=sha(B/'initial_plan.json'):raise ValueError('plan contract')
 upstream=R/'pixel_wavefront_t020_execution_preflight_result_review.json';u=read(upstream)
 if u.get('status')!='passed' or u.get('independent_review') is not True or u.get('gpu_authorized') is not False or p['upstream_review_sha256']!=sha(upstream):raise ValueError('upstream evidence')
 jobs=p['jobs'];anchors=read(B/'e6/pixel_wavefront_model_t020/preflight_plan002.json')['images'];bykey={(x['anchor']['task'],str(x['anchor']['qid'])):x['anchor'] for x in anchors}
 if len(jobs)!=6 or [x['order'] for x in jobs]!=list(range(6)) or {(x['task'],str(x['qid']),x['arm']) for x in jobs}!={(t,q,arm) for t,q in bykey for arm in ('author','wavefront')}:raise ValueError('six-job inventory')
 for j in jobs:
  ref=bykey[(j['task'],str(j['qid']))]
  if any(j[k]!=ref[k] for k in ('group_id','image_sha256','message_sha256')):raise ValueError('anchor identity')
 required=[D/x for x in ('preflight_admission.py','preflight_entry.py','preflight_launch.py','gpu_admission.py','source_inventory.py','upstream_identity.py','execution_upstream.py','worker_entry.py','single_arm.py','controller.py','bounded_runner_draft.py','schedule_contract.py','snapshot_sources.py','weight_guard.py','supervision_contract.py','imports_guard.py')]
 if any(actual.get(str(x))!=sha(x) for x in required):raise ValueError('unbound execution module')
 sys.path.insert(0,str(D));import gpu_admission
 try:gpu_admission.require_gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU not locked')
 source=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py';spec=importlib.util.spec_from_file_location('independent_t020_execution_runtime',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cmd=m.container_command(read(source.parent/'run_plan.json'),[str(D/'preflight_entry.py'),'--out',str(D/'run_preflight001/worker_preflight.json')],False,'eadp-review-cpu')
 if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or cmd.count('--rm')!=1:raise ValueError('no-GPU container')
 proof={'status':'passed','independent_review':True,'preflight_only':True,'gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'reviewer_sha256':sha(__file__),'source_sha256':actual,'plan_sha256':sha(plan),'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(mon/'started.json'),'cpu_terminal_sha256':sha(mon/'terminal.json'),'errors':[],'limitations':['Only execution import/no-GPU container preflight is admitted; no model identity or GPU work authorized.','Reviewer authored wavefront candidate but not this execution preflight.','Prior DEV prediction evidence was read for identity.']}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
