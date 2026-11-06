"""Freeze random K256 image/seed manifest; no predictions, GT, model or GPU."""
import collections,hashlib,importlib.util,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];RB=B/'e6/random_baselines';E=B/'e1';R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d):
 s=read(d/'started.json');t=read(d/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('unclean supervisor: '+str(d))
for d in (RB/'monitor_cpu001',D/'monitor_source_audit001',E/'monitore1full001',E/'monitorverifye1full001',R/'monitor_e1full001',B/'e4_native/monitor_native001',B/'e4_native/monitor_verify_native001',R/'monitor_e4native001'):terminal(d)
cpu=read(RB/'cpu_check_001/verification.json');formal=read(E/'e1full001_completion.json');ind=read(R/'e1full001_review.json');e1f=read(E/'e1full001/frozen.json');inp=E/'input_preparation_002/inputs.jsonl';iv=read(E/'input_preparation_002/independent_verification.json');native=read(B/'e4_native/native_001_verification.json');ni=read(R/'e4native001_review.json')
if cpu['status']!='passed' or cpu['source_sha256_before']!=cpu['source_sha256_after'] or any(sha(p)!=h for p,h in cpu['source_sha256_after'].items()):raise ValueError('random source CPU proof mismatch')
if formal['status']!='passed' or formal['natural_questions']!=384 or formal['GT_opened'] is not False or ind['status']!='passed' or ind['n_verified']!=384 or ind['completion_proof_sha256']!=sha(E/'e1full001_completion.json') or iv['status']!='passed' or iv['n_verified']!=384 or iv['inputs_sha256']!=sha(inp) or e1f['inputs_sha256']!=sha(inp):raise ValueError('E1 input acceptance mismatch')
if native['status']!='passed' or native['GT_opened'] is not False or ni['status']!='passed' or ni['formal_proof_sha256']!=sha(B/'e4_native/native_001_verification.json'):raise ValueError('native accepted identity missing')
sel_path=RB/'selector.py';spec=importlib.util.spec_from_file_location('eadp_random_manifest_selector',sel_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[json.loads(s) for s in inp.read_text().splitlines()];index={(r['task'],str(r['qid'])):r for r in rows}
if len(rows)!=384 or len(index)!=384 or len({r['group_id'] for r in rows})!=384 or collections.Counter(r['task'] for r in rows)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('DEV inventory mismatch')
anchors=read(B/'e4_c_budget_ttft_pilot_v3/pilot_plan001.json')['anchors'];seeds=read(RB/'plan.json')['seeds']
if seeds!=[20261104,20261105,20261106] or len(anchors)!=3 or len({x['task'] for x in anchors})!=3:raise ValueError('anchor/seed mismatch')
for a in anchors:
 r=index[(a['task'],str(a['qid']))]
 if any(a[k]!=r[k] for k in ('group_id','image_sha256','message_sha256')):raise ValueError('anchor input mismatch')
manifest=D/'selection_manifest001.jsonl'
with manifest.open('x') as f:
 for r in rows:
  for seed in seeds:
   choice=mod.selection(r['image_sha256'],seed)
   f.write(json.dumps({'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'image_sha256':r['image_sha256'],'message_sha256':r['message_sha256'],**choice},sort_keys=True,separators=(',',':'))+'\n')
 f.flush();os.fsync(f.fileno())
sources=[Path(__file__),RB/'plan.json',RB/'cpu_check_001/verification.json',inp,E/'e1full001/frozen.json',E/'e1full001_completion.json',E/'input_preparation_002/independent_verification.json',R/'e1full001_review.json',B/'e4_native/native_001_verification.json',R/'e4native001_review.json',B/'e4_c_budget_ttft_pilot_v3/pilot_plan001.json',B/'initial_plan.json']+[Path(x) for x in cpu['source_sha256_after']]
for d in (RB/'monitor_cpu001',D/'monitor_source_audit001',E/'monitore1full001',E/'monitorverifye1full001',R/'monitor_e1full001',B/'e4_native/monitor_native001',B/'e4_native/monitor_verify_native001',R/'monitor_e4native001'):sources.extend([d/'started.json',d/'terminal.json'])
plan={'status':'planning_only','scope':'posthoc random K256 baseline after prior DEV studies; no GT or predictions opened','seeds':seeds,'selection_manifest_sha256':sha(manifest),'n_questions':384,'n_seed_question_pairs':1152,'anchors':anchors,'smoke':{'natural_arms':9,'forced_arms':9,'forced_output_ids':33,'forced_decode_steps':32,'forced_token_id':100,'no_scoring':True},'full':{'natural_arms':1152,'complete_before_scoring':True,'include_all_seeds':True},'source_sha256':{str(p.resolve()):sha(p) for p in sorted(set(sources))},'GT_opened':False,'predictions_opened':False,'GPU_authorized':False,'limits':['random seed and plan selected after previous DEV observations','diagnostic adapter computes and discards EADP greedy; never use for timing','E1 full reference journal must be verified before actual execution and contains DEV predictions','formal and independent GPU smoke/no-GPU preflight not yet implemented']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('prepared 384 x 3 choices, 18 planned smoke arms')
