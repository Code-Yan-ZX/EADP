"""Freeze source inventory for local noGPU entry import checks only."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v7';R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate_v2';M=B/'e6/pixel_wavefront_t020_nvml_model_candidate';REF=B/'e6/pixel_wavefront_t020_nvml_reference';P=B/'e6/pixel_wavefront_t020_nvml_preflight_candidate'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('CPU supervisor')
def main():
 out=E/'cpu_preflight_plan001.json'
 if out.exists():raise ValueError('plan exists')
 for base,proof,monitor in ((E,'cpu_proof001.json','monitor_cpu001'),(R,'cpu_proof001.json','monitor_cpu001'),(M,'cpu_proof001.json','monitor_cpu001'),(REF,'cpu_proof001.json','monitor_cpu001')):
  terminal(base/monitor,base/'cpu_check.py');q=read(base/proof)
  if q.get('status')!='passed' or q.get('GPU_used') is not False or q['source_sha256_before']!=q['source_sha256_after'] or any(sha(k)!=v for k,v in q['source_sha256_after'].items()):raise ValueError('CPU proof source')
 terminal(REF/'monitor_real_refs001',REF/'check_real_refs.py');ref=read(REF/'real_refs001.json')
 if ref.get('status')!='passed' or ref.get('references')!=9 or ref.get('existing_DEV_predictions_opened') is not True or ref.get('GT_opened') is not False:raise ValueError('reference proof')
 old=read(P/'plan001.json');source=dict(old['source_sha256'])
 for base in (E,R,M,REF):
  for p in base.glob('*.py'):source[str(p.resolve())]=sha(p)
 paths=[D/'freeze.py',P/'plan001.json',REF/'real_refs001.json',REF/'monitor_real_refs001/started.json',REF/'monitor_real_refs001/terminal.json']
 for base,proof,monitor in ((E,'cpu_proof001.json','monitor_cpu001'),(R,'cpu_proof001.json','monitor_cpu001'),(M,'cpu_proof001.json','monitor_cpu001'),(REF,'cpu_proof001.json','monitor_cpu001')):paths += [base/proof,base/monitor/'started.json',base/monitor/'terminal.json']
 for p in paths:source[str(p.resolve())]=sha(p)
 if any(sha(k)!=v for k,v in source.items()):raise ValueError('source drift')
 v={'status':'frozen','scope':'local noGPU entry dependency smoke only; no container or GPU admission','source_sha256':source,'GPU_authorized':False,'existing_DEV_predictions_opened':True,'GT_opened':False,'limitations':['the four actual CLIs still reject GPU; this plan is not a runtime admission','accepted existing DEV outputs only for identity','real GPU sampler, supervisor, source snapshot, processor and weight evidence require separate full plan and independent review']}
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
