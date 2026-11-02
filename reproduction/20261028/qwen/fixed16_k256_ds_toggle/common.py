"""Frozen source and artifact guards for K256 fixed16 DS diagnostic."""
import hashlib,json,sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;Q=HERE.parent;OLD=Q/'gpu_formula_facility_panel_run02';FACTOR=Q/'fixed16_ds_factorial';AUDIT=Q/'independent_audit'
PLAN=HERE/'plan.json';RUN_ID='Qwen_fixed16_K256_native_selection_prefill_DS_toggle'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def identity(pid):
 try:ticks=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:return None
 return dict(pid=pid,start_ticks=ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def check(plan,namespace='host'):
 assert plan['run_id']==RUN_ID and len(plan['panel'])==16 and plan['model_instances']==1
 for path,digest in plan['files'].items():assert sha(path)==digest,path
 sys.path.insert(0,str(OLD));import diagnose
 diagnose.validate(read(OLD/'run_plan.json'),namespace)
 assert plan['panel']==read(OLD/'run_plan.json')['panel']
 r=read(OLD/'root_terminal_process_proof.json');assert r['root_owned_exec_actual_exit_code']==0 and r['all_original_proc_paths_absent'] and r['container_absent']
 r=read(FACTOR/'root_terminal_process_proof.json');v=read(FACTOR/'execution/independent_verification.json')
 assert r['root_owned_exec_actual_exit_code']==0 and r['all_original_processes_exited'] and r['independent_verification_sha256']==sha(FACTOR/'execution/independent_verification.json') and v['success']
 a=read(AUDIT/'public_fa2_full_run02.terminal_audit.report.json');assert a['audit_terminated'] and a['coverage_complete'] and a['generation_valid'] and a['public_launcher_exited_successfully']
 assert a['existing_CPU_verification']['datasets']['DocVQA_VAL']['per_question']['sha256']==sha(AUDIT/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl')
 assert 'torch' not in sys.modules
