"""No-label guards for full Doc K256 paired generation."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;Q=HERE.parent;OLD=Q/'gpu_formula_facility_panel_run02';FIXED=Q/'fixed16_k256_ds_toggle';INPUT=Q/'doc_full_two_routes/input_manifest.jsonl';PLAN=HERE/'plan.json'
RUN_ID='Qwen_Doc5349_K256_native_selection_prefill_DS_pair'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):
 with Path(path).open() as f:return [json.loads(s) for s in f]
def save(path,data):
 p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def identity(pid):
 try:ticks=int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
 except FileNotFoundError:return None
 return dict(pid=pid,start_ticks=ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def check(plan,namespace='host'):
 assert plan['run_id']==RUN_ID and plan['rows']==5349 and plan['model_instances']==1
 for path,digest in plan['files'].items():assert sha(path)==digest,path
 old=read(OLD/'run_plan.json')
 # Only code/weights and unlabeled images: reference_sha256 may contain
 # historical score-bearing data and is deliberately not read in this task.
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in old[group].items():assert sha(path)==digest,path
 sys.path.insert(0,str(OLD));from namespace_guards import check_model_stats
 from diagnose import stat
 from formula_helpers import parts
 check_model_stats(old,stat,namespace);assert parts(old['pruner_path'])[-1]==old['source_AST_proof']
 assert old['recipe']['visual_token_num']==256 and old['recipe']['alpha']==.5 and old['recipe']['beta']==2.
 p=read(FIXED/'root_exit_proof.json');v=read(FIXED/'execution/independent_verification.json');r=read(FIXED/'execution/release.json')
 assert p['root_pty_actual_exit_code']==0 and p['original_container_stopped'] and all(x['original_process_absent'] for x in p['roles'])
 assert p['independent_verification_sha256']==sha(FIXED/'execution/independent_verification.json') and p['release_sha256']==sha(FIXED/'execution/release.json')
 assert v['success'] and v['n_questions']==16 and v['n_generations']==32 and r['success']
 items=rows(INPUT);assert len(items)==5349 and [x['position'] for x in items]==list(range(5349)) and len({x['index'] for x in items})==5349
 images={x['image_path']:x['image_sha256'] for x in items};assert len(images)==1286
 for path,digest in images.items():assert sha(path)==digest,path
 assert 'torch' not in sys.modules
