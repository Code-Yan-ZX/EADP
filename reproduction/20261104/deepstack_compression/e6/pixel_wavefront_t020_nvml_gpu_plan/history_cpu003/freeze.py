"""Freeze nine-job GPU plan only after independently reviewed real noGPU preflight.

No model, GPU or NVML query is performed. All earlier evidence remains intact.
"""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';P=B/'e6/pixel_wavefront_t020_nvml_execution_preflight_v2';I=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path,expected):
 path=Path(path).resolve()
 if name in sys.modules or sha(path)!=expected:raise ValueError('source changed or cached')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 out=E/'gpu_plan001.json'
 if out.exists():raise ValueError('GPU plan already exists')
 p=read(P/'plan001.json');source=dict(p['source_sha256'])
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('preflight sources changed')
 pre=I/'pixel_t020_nvml_execution_preflight_v2_result_review.json';reviewer=I/'review_pixel_t020_nvml_execution_preflight_v2_result.py';monitor=I/'monitor_pixel_t020_nvml_execution_preflight_v2_result';g=load('_pixel_nvml_plan_gate',E/'gate.py',source[str((E/'gate.py').resolve())]);g.terminal(monitor,reviewer,pre);g.terminal(P/'monitor_preflight001',P/'launch.py',P/'run_preflight001');q=read(pre)
 fields={'plan_sha256':P/'plan001.json','worker_preflight_sha256':P/'run_preflight001/worker_preflight.json','launch_sha256':P/'run_preflight001/launch.json','identity_sha256':P/'run_preflight001/container_identity.json','exit_sha256':P/'run_preflight001/exit.json','started_sha256':P/'monitor_preflight001/started.json','terminal_sha256':P/'monitor_preflight001/terminal.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('independent noGPU preflight')
 supplement=B/'e6/pixel_wavefront_t020_nvml_execution_integration_v4';g.terminal(supplement/'monitor_cpu001',supplement/'cpu_check.py');v=read(supplement/'cpu_proof001.json')
 if v.get('status')!='passed' or v.get('GPU_used') is not False or v.get('GT_opened') is not False or v.get('source_sha256_before')!=v.get('source_sha256_after') or any(sha(k)!=x for k,x in v['source_sha256_after'].items()):raise ValueError('three-arm integration proof')
 source.update(v['source_sha256_after'])
 g.terminal(D/'monitor_cpu003',D/'cpu_check.py');own=read(D/'cpu_proof003.json')
 if own.get('status')!='passed' or own.get('GPU_used') is not False or own.get('GT_opened') is not False or own.get('source_sha256_before')!=own.get('source_sha256_after') or any(sha(k)!=x for k,x in own['source_sha256_after'].items()):raise ValueError('plan CPU proof')
 source.update(own['source_sha256_after'])
 for x in (D/'cpu_proof003.json',D/'monitor_cpu003/started.json',D/'monitor_cpu003/terminal.json'):source[str(x.resolve())]=sha(x)
 analysis=B/'e6/pixel_wavefront_t020_nvml_execution_analysis';g.terminal(analysis/'monitor_cpu002',analysis/'cpu_check.py');av=read(analysis/'cpu_proof002.json')
 if av.get('status')!='passed' or av.get('GPU_used') is not False or av.get('GT_opened') is not False or av.get('timing_results_opened') is not False or av.get('source_sha256_before')!=av.get('source_sha256_after') or any(sha(k)!=x for k,x in av['source_sha256_after'].items()):raise ValueError('analysis CPU proof')
 source.update(av['source_sha256_after'])
 for path in (analysis/'cpu_proof002.json',analysis/'monitor_cpu002/started.json',analysis/'monitor_cpu002/terminal.json'):source[str(path.resolve())]=sha(path)
 for path in list(fields.values())+[pre,monitor/'started.json',monitor/'terminal.json',supplement/'cpu_proof001.json',supplement/'monitor_cpu001/started.json',supplement/'monitor_cpu001/terminal.json',D/'freeze.py',D/'freeze_formal.py',D/'freeze_analysis.py']:source[str(path.resolve())]=sha(path)
 prior=B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_gate.py';weight=load('_pixel_nvml_plan_weight_gate',prior,source[str(prior.resolve())]).weight_preflight();design=B/'e6/pixel_wavefront_t020_nvml_nine/design001.json';initial=B/'initial_plan.json';budget=read(initial);jobs=read(design)['jobs']
 if len(jobs)!=9 or [j.get('order') for j in jobs]!=list(range(9)) or len(weight['expected'])!=4:raise ValueError('identity inventory')
 for path in (design,initial):source[str(path.resolve())]=sha(path)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('freeze source changed')
 value={'scope':'pixel_t020_t0_nine_process_nvml','jobs':jobs,'design_sha256':sha(design),'source_sha256':source,'weight_reference_sha256':weight['worker_preflight_sha256'],'weight_files':[x['path'] for x in weight['expected']],'budget_source_sha256':sha(initial),'budget_end_unix':budget['budget_end_unix'],'job_timeout_seconds':2400,'job_grace_seconds':240,'outer_timeout_seconds':28800,'grace_seconds':600,'GPU_used':False,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'limitations':['three post-hoc DEV anchors, unequal dynamic token counts, no equal-compute claim','nine isolated model loads, MiB sampling can miss transient peaks','processor capture and synchronization perturb memory; no TTFT claim','endpoint weight stat cannot exclude transient replacement','no GPU authority before independent final prelaunch and frozen formal plan']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
