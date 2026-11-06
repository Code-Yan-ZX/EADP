"""Freeze only a same-image noGPU import preflight; no model/GPU authority."""
import json,os
from pathlib import Path
import gate
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8'
def main():
 out=D/'plan001.json'
 if out.exists():raise ValueError('plan exists')
 gate.terminal(D/'monitor_cpu002',D/'cpu_check.py');cpu=gate.read(D/'cpu_proof002.json')
 if cpu.get('status')!='passed' or cpu['source_sha256_before']!=cpu['source_sha256_after'] or any(gate.sha(k)!=v for k,v in cpu['source_sha256_after'].items()):raise ValueError('CPU proof')
 prior=E/'cpu_preflight_plan001.json';p=gate.read(prior)
 if p.get('status')!='frozen' or p.get('GPU_authorized') is not False or any(gate.sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('prior source plan')
 source=dict(p['source_sha256']);source[str(prior.resolve())]=gate.sha(prior)
 for directory,proof,monitor in ((B/'e6/pixel_wavefront_t020_nvml_execution_accept','cpu_proof006.json','monitor_cpu006'),(B/'e6/pixel_wavefront_t020_nvml_execution_integration_v3','cpu_proof001.json','monitor_cpu001')):
  gate.terminal(directory/monitor,directory/'cpu_check.py');v=gate.read(directory/proof)
  if v.get('status')!='passed' or v.get('GPU_used') is not False or v.get('GT_opened') is not False or v.get('source_sha256_before')!=v.get('source_sha256_after') or any(gate.sha(k)!=z for k,z in v['source_sha256_after'].items()):raise ValueError('execution/formal CPU proof')
  source.update(v['source_sha256_after'])
  for x in (directory/proof,directory/monitor/'started.json',directory/monitor/'terminal.json'):source[str(x.resolve())]=gate.sha(x)
 for name,script in [('worker','worker.py'),('controller','controller_entry.py'),('sampler','sampler_entry.py')]:
  m=E/('monitor_'+name+'_smoke001');gate.terminal(m,E/script);started=gate.read(m/'started.json');v=gate.read(m/'stdout.log')
  if started['command'][-1]!='--no-gpu-dependency-smoke' or Path(started['cwd']).resolve()!=B or v.get('status')!='passed' or v.get('model_loaded') is not False:raise ValueError('actual host entry smoke')
  if name=='worker' and (v.get('references')!=9 or v.get('cuda_initialized') is not False):raise ValueError('worker smoke')
  if name!='worker' and v.get('gpu_queried') is not False:raise ValueError('host GPU query')
  for x in (m/'started.json',m/'terminal.json',m/'stdout.log'):source[str(x.resolve())]=gate.sha(x)
 for x in list(D.glob('*.py'))+[D/'cpu_proof002.json',D/'monitor_cpu002/started.json',D/'monitor_cpu002/terminal.json',B/'e6/pixel_wavefront_t020_nvml_cpu_plan_v4/monitor_freeze001/started.json',B/'e6/pixel_wavefront_t020_nvml_cpu_plan_v4/monitor_freeze001/terminal.json']:source[str(x.resolve())]=gate.sha(x)
 if any(gate.sha(k)!=v for k,v in source.items()):raise ValueError('source drift')
 value={'scope':'pixel nine-reference noGPU imports only','source_sha256':source,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'limits':['only noGPU same-image imports; no model or NVML query','prior accepted DEV outputs used solely for same-arm identity','final GPU/sampler admission and formal verifier still missing','t0 ordered indices archived and compared, high-dimensional original tensors not fully archived']}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
