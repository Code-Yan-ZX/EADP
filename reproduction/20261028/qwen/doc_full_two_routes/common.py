"""Full DocVQA two native routes: bounded guards and process metadata."""
import hashlib,json,os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
Q=HERE.parent
PANEL=Q/'full1024_two_path_panel'
OLD=Q/'gpu_formula_facility_panel_run02'
PLAN=HERE/'plan.json'
RUN_ID='Qwen_Doc5349_plain_and_EADP1024_full_routes'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def rows(path):
 with Path(path).open() as f:return [json.loads(s) for s in f]
def identity(pid):
 try:ticks=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:return None
 return dict(pid=pid,start_ticks=ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def check(plan,namespace='host'):
 assert plan['run_id']==RUN_ID and plan['rows']==5349 and plan['model_instances']==1
 for path,digest in plan['file_guards'].items():assert sha(path)==digest,path
 sys.path.insert(0,str(OLD));import diagnose
 old=read(OLD/'run_plan.json');diagnose.validate(old,namespace)
 panel=read(PANEL/'execution/independent_verification.json');root=read(PANEL/'root_terminal_process_proof.json')
 assert panel['success'] and panel['n_native_generations']==32 and root['root_owned_exec_actual_exit_code']==0 and root['all_original_processes_exited']
 items=rows(HERE/'input_manifest.jsonl');assert len(items)==5349 and len({r['index'] for r in items})==5349
 images={r['image_path']:r['image_sha256'] for r in items};assert len(images)==1286
 for path,digest in images.items():assert sha(path)==digest,path
 assert 'torch' not in sys.modules
