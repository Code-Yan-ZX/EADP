"""Freeze noGPU dependency preflight only; not a GPU execution plan."""
import json,os
from pathlib import Path
import gate
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def main():
 plan=D/'plan001.json'
 if plan.exists():raise ValueError('preflight plan exists')
 own=gate.read(D/'cpu_proof001.json');gate.terminal(D/'monitor_cpu001',D/'cpu_check.py')
 if own.get('status')!='passed' or own.get('GPU_used') is not False or own['source_sha256_before']!=own['source_sha256_after'] or any(gate.sha(k)!=v for k,v in own['source_sha256_after'].items()):raise ValueError('CPU proof')
 ref=B/'e6/pixel_wavefront_t020_nvml_reference';model=B/'e6/pixel_wavefront_t020_nvml_model_candidate';host=B/'e6/pixel_wavefront_t020_nvml_host_candidate';design=B/'e6/pixel_wavefront_t020_nvml_nine/design001.json'
 for path,script in [(ref/'monitor_cpu001',ref/'cpu_check.py'),(ref/'monitor_real_refs001',ref/'check_real_refs.py'),(model/'monitor_cpu001',model/'cpu_check.py')]:gate.terminal(path,script)
 cpuref=gate.read(ref/'cpu_proof001.json');cpumodel=gate.read(model/'cpu_proof001.json');actual=gate.read(ref/'real_refs001.json')
 if cpuref.get('status')!='passed' or cpumodel.get('status')!='passed' or actual.get('status')!='passed' or actual.get('references')!=9 or actual.get('existing_DEV_predictions_opened') is not True or actual.get('GT_opened') is not False or actual.get('GPU_used') is not False or actual.get('source_sha256')!=gate.sha(ref/'reference.py'):raise ValueError('reference or CPU proof')
 for proof in (cpuref,cpumodel):
  if proof['source_sha256_before']!=proof['source_sha256_after'] or any(gate.sha(k)!=v for k,v in proof['source_sha256_after'].items()):raise ValueError('candidate source drift')
 prior=B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_plan001.json';source=dict(gate.read(prior)['source_sha256']);source[str(prior.resolve())]=gate.sha(prior)
 for p in list(D.glob('*.py'))+list(ref.glob('*.py'))+list(model.glob('*.py'))+list(host.glob('*.py'))+[design,D/'cpu_proof001.json',D/'monitor_cpu001/started.json',D/'monitor_cpu001/terminal.json',ref/'cpu_proof001.json',ref/'real_refs001.json',ref/'monitor_cpu001/started.json',ref/'monitor_cpu001/terminal.json',ref/'monitor_real_refs001/started.json',ref/'monitor_real_refs001/terminal.json',model/'cpu_proof001.json',model/'monitor_cpu001/started.json',model/'monitor_cpu001/terminal.json']:
  source[str(p.resolve())]=gate.sha(p)
 for job in gate.read(design)['jobs']:
  if job['arm']=='t0_author':
   stem=job['task']+'_'+str(job['qid']);base=B/'e6/pixelprune_dev384/run_smoke001/observations'
   for p in (base/(stem+'.json'),base/(stem+'_generated.json')):source[str(p.resolve())]=gate.sha(p)
 if any(gate.sha(k)!=v for k,v in source.items()):raise ValueError('source drift')
 v={'scope':'pixel nine-reference noGPU imports only','source_sha256':source,'design_sha256':gate.sha(design),'reference_proof_sha256':gate.sha(ref/'real_refs001.json'),'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'limitations':['no model loaded, no CUDA or NVML sampling; no GPU execution authority','existing DEV outputs parsed for identity only','later real sampler/controller/worker and formal verifier require separate source freeze, same-image preflight and independent review','ordered t0 indices are archived; full hidden intermediates are not']}
 with plan.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
