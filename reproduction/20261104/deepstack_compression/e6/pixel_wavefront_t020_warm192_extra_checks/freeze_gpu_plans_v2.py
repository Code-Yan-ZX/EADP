"""Freeze plans only after the independently reviewed real no-device preflight."""
import hashlib,importlib.util,json,os,time
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=B/'e6/pixel_wavefront_t020_warm192';E=Path(__file__).resolve().parent;I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def add(source,p):
 p=Path(p).resolve();v=sha(p)
 if str(p) in source and source[str(p)]!=v:raise ValueError('source changed')
 source[str(p)]=v
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
spec=importlib.util.spec_from_file_location('warm192_frozen_gate_for_plan',D/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
gate.admit_cpu()
pre=read(D/'preflight_plan001.json');source=pre['source_sha256'].copy();gate.source_check(source)
reviewer=I/'review_pixel_t020_warm192_preflight_result.py';proof=I/'pixel_t020_warm192_preflight_result_review.json';monitor=I/'monitor_pixel_t020_warm192_preflight_result'
gate.terminal(monitor,reviewer,proof);gate.terminal(D/'monitor_preflight001',D/'preflight_launch.py',D/'run_preflight001')
q=read(proof);fields={'plan_sha256':D/'preflight_plan001.json','worker_preflight_sha256':D/'run_preflight001/worker_preflight.json','launch_sha256':D/'run_preflight001/launch.json','identity_sha256':D/'run_preflight001/container_identity.json','exit_sha256':D/'run_preflight001/exit.json','started_sha256':D/'monitor_preflight001/started.json','terminal_sha256':D/'monitor_preflight001/terminal.json','reviewer_sha256':reviewer}
if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('real noGPU result proof')
for p in [*fields.values(),proof,reviewer,monitor/'started.json',monitor/'terminal.json',I/'pixel_t020_warm192_cpu_preflight_review.json',I/'review_pixel_t020_warm192_cpu_preflight.py',I/'monitor_pixel_t020_warm192_cpu_preflight/started.json',I/'monitor_pixel_t020_warm192_cpu_preflight/terminal.json',E/'host_entry_smoke002.py',E/'host_entry002.json',E/'monitor_host_entry002/started.json',E/'monitor_host_entry002/terminal.json',E/'gpu_launch_v2.py',E/'formal_v2.py',E/'analysis_v2.py',E/'observer_guard_v2.py',E/'test_observer_guard_v2.py',E/'monitor_observer_guard_cpu001/started.json',E/'monitor_observer_guard_cpu001/terminal.json',E/'observer_guard_real_cpu.py',E/'observer_guard_real_cpu001.json',E/'monitor_observer_guard_real_cpu001/started.json',E/'monitor_observer_guard_real_cpu001/terminal.json',E/'host_entry_smoke003.py',E/'host_entry003.json',E/'monitor_host_entry003/started.json',E/'monitor_host_entry003/terminal.json',E/'one_token_generate.py',E/'one_token_generate001.json',E/'monitor_one_token_generate001/started.json',E/'monitor_one_token_generate001/terminal.json',Path(__file__),B/'initial_plan.json']:
 add(source,p)
for p in (E/'postrun_entry_smoke_v2.py',E/'postrun_entry_smoke_v2_001.json',E/'monitor_postrun_entry_smoke_v2_001/started.json',E/'monitor_postrun_entry_smoke_v2_001/terminal.json'):add(source,p)
gate.terminal(E/'monitor_postrun_entry_smoke_v2_001',E/'postrun_entry_smoke_v2.py');gate.terminal(E/'monitor_host_entry003',E/'host_entry_smoke003.py');gate.terminal(E/'monitor_observer_guard_real_cpu001',E/'observer_guard_real_cpu.py');gate.terminal(E/'monitor_host_entry002',E/'host_entry_smoke002.py');gate.terminal(E/'monitor_one_token_generate001',E/'one_token_generate.py')
check=read(E/'one_token_generate001.json')
if check.get('status')!='passed' or check.get('actual_transformers_generate') is not True or check.get('cuda_initialized') is not False or check.get('script_sha256')!=sha(E/'one_token_generate.py'):raise ValueError('CPU first token check')
initial=read(B/'initial_plan.json');design=read(D/'design001.json');py=str(B.parents[2]/'_reproduction/envs/qwen_author/bin/python')
if not Path(py).is_file():raise ValueError('interpreter')
if time.time()+21600+300>=initial['budget_end_unix']:raise ValueError('insufficient hard deadline')
gpu={'scope':'pixel_t020_warm192_eight_process_ttft','jobs':design['processes'],'design_sha256':sha(D/'design001.json'),'source_sha256':source,'budget_end_unix':initial['budget_end_unix'],'budget_source_sha256':sha(B/'initial_plan.json'),'outer_timeout_seconds':21600,'grace_seconds':300,'snapshot_seconds':3600,'poll_seconds':5,'inner_timeout_seconds':2400,'cleanup_seconds':300,'minimum_disk_bytes':10737418240,'run_supervisor_command':[py,str(E/'gpu_launch_v2.py'),'--run-id','001'],'run_supervisor_cwd':str(B),'GT_opened':False,'scoring_performed':False,'GPU_used_at_freeze':False,'predictions_opened':True,'prediction_scope':'accepted E1 and previous six-arm records parsed for identity; four accepted earlier generated files hashed as part of source inventory; no new predictions','contract':'4 blocks, AB/BA/BA/AB, 8 independent model processes; 1 warmup and 48 distinct one-token requests per process; all 192 pairs retained','limitations':['post-hoc DEV sample after three-anchor results; paired implementation comparison only','one-token maximum differs from earlier natural 2048-token generation','fresh KV and selector observation includes synchronization within both TTFT arms','one warmup per independent process does not prove steady state','observer guard uses a conservative Linux proc start lower bound and checks supervisor/liveness before GPU query','endpoint stat cannot exclude transient replacement','observer windows and unknown background retained; no filtering']}
write(D/'gpu_plan001.json',gpu)
formal={'scope':'pixel_t020_warm192_formal_verification','gpu_plan_sha256':sha(D/'gpu_plan001.json'),'GT_opened':False,'scoring_performed':False,'GPU_used_at_freeze':False,'source_sha256':source,'records':'8 excluded warmups and all 384 measured requests, actual container exit and complete processor/selection/first-token and cache checks; record all mismatches'}
write(D/'formal_plan001.json',formal)
analysis={'scope':'pixel_t020_warm192_paired_analysis','gpu_plan_sha256':sha(D/'gpu_plan001.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),'source_sha256':source,'GPU_used':False,'GT_opened':False,'draws':10000,'bootstrap_seed':20261105,'denominator':'192 fixed unique images; 64 per task, all paired observations','comparison':'wavefront minus author TTFT and ratio, paired by image; bootstrap within task by image','limitations':gpu['limitations']}
write(D/'analysis_plan001.json',analysis)
print(json.dumps({'status':'passed','source_count':len(source),'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'formal_plan_sha256':sha(D/'formal_plan001.json'),'analysis_plan_sha256':sha(D/'analysis_plan001.json')}))
