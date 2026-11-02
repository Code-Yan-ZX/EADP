"""Source/experiment guards for fixed16 native DeepStack toggle candidate."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
Q=HERE.parent
PREV=Q/'full1024_two_path_panel'
OLD=Q/'gpu_formula_facility_panel_run02'
FULL=Q/'doc_full_two_routes'
PLAN=HERE/'plan.json'
RUN_ID='Qwen_fixed16_plain_EADP1024_prefill_deepstack_toggle'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def check(plan,namespace='host'):
 assert plan['run_id']==RUN_ID and len(plan['panel'])==16
 for path,digest in plan['files'].items():assert sha(path)==digest,path
 sys.path.insert(0,str(OLD));import diagnose
 prior=read(OLD/'run_plan.json');diagnose.validate(prior,namespace)
 assert plan['panel']==read(PREV/'plan.json')['panel']
 proof=read(PREV/'root_terminal_process_proof.json');report=read(PREV/'execution/independent_verification.json')
 assert proof['root_owned_exec_actual_exit_code']==0 and proof['all_original_processes_exited']
 assert report['success'] and report['n_questions']==16 and report['n_native_generations']==32
 assert 'torch' not in sys.modules
