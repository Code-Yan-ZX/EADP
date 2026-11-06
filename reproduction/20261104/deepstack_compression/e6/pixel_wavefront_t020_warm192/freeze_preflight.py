"""Freeze source for a no-device-only preflight; no GPU/GT/results."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('supervisor')
def main():
 out=D/'preflight_plan001.json'
 if out.exists():raise ValueError('preflight plan exists')
 terminal(D/'monitor_design001',D/'freeze_design.py');terminal(D/'monitor_cpu001',D/'cpu_check.py');cpu=read(D/'cpu_proof001.json');design=read(D/'design001.json')
 if cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after') or any(sha(k)!=v for k,v in cpu['source_sha256_after'].items()) or design.get('selected_questions')!=192 or design.get('measured_records')!=384 or design.get('GT_opened') is not False or any(sha(k)!=v for k,v in design['source_sha256'].items()):raise ValueError('CPU/design evidence')
 prior=B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_plan001.json';source=dict(read(prior)['source_sha256']);source.update(cpu['source_sha256_after']);source.update(design['source_sha256'])
 extra=[prior,D/'design001.json',D/'rule001.json',D/'cpu_proof001.json',D/'monitor_cpu001/started.json',D/'monitor_cpu001/terminal.json',D/'monitor_design001/started.json',D/'monitor_design001/terminal.json',B/'e4_c_latency/timing_supervise.py',B/'e6/pixel_wavefront_t020_timing_accept_v5/window_overlap.py',B/'initial_plan.json',B/'e1_independent_review/pixel_wavefront_t020_timing_final_v9_prelaunch_review.json']
 for p in extra:source[str(p.resolve())]=sha(p)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed')
 result={'scope':'warm192_noGPU_processor_preflight','source_sha256':source,'design_sha256':sha(D/'design001.json'),'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed journal and prior six-arm records used only for identities','limits':['real 193-processor/4-weight preflight only; no CUDA, model load or timing','eight-process source remains GPU locked until final plans and independent reviews','one-token max differs from prior natural 2048 generation','existing cohort chosen after earlier 3-anchor results']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
