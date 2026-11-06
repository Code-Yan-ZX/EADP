"""Post-run verifier candidate; will remain unusable without future frozen plans."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];RUN=B/'e6/pixel_wavefront_t020_timing_final_v4';DESIGN=B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if (len(c)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor proof')
 return s,t
def load(name,path,source):
 p=Path(path).resolve()
 if source.get(str(p))!=sha(p):raise ValueError('source mismatch before import')
 m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('module collision')
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out.resolve()
 if out!=(D/'verification001.json').resolve():raise ValueError('unexpected verifier output')
 formal=read(D/'verification_plan001.json');gpu_plan=read(RUN/'gpu_plan001.json');formal_sources=formal['source_sha256'];gpu_sources=gpu_plan['source_sha256']
 if formal.get('gpu_plan_sha256')!=sha(RUN/'gpu_plan001.json') or any(sha(k)!=v for k,v in formal_sources.items()) or any(sha(k)!=v for k,v in gpu_sources.items()):raise ValueError('plan or source identity')
 cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'cpu_check.py')
 if cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=formal_sources or cpu.get('source_sha256_after')!=formal_sources or cpu.get('GT_opened') is not False or cpu.get('predictions_opened') is not False:raise ValueError('formal CPU proof')
 started,ended=terminal(RUN/'monitor_run001',RUN/'launch.py');terminal(D/'monitor_snapshot001',D/'snapshot_entry.py',RUN/'run_001_sources')
 manifest=read(RUN/'run_001_sources/manifest.json')
 if manifest.get('status')!='passed' or manifest.get('gpu_plan_sha256')!=sha(RUN/'gpu_plan001.json') or set(manifest['sources'])!=set(gpu_sources) or any(v['sha256']!=gpu_sources[k] or sha(v['copy'])!=v['sha256'] for k,v in manifest['sources'].items()):raise ValueError('source snapshot')
 # GPU launch proof and independent noGPU evidence are verified before touching records.
 gate=load('_pixel_t020_formal_gpu_gate',RUN/'gpu_gate.py',gpu_sources);gate.admit()
 core=load('_pixel_t020_formal_core',D/'verify_candidate.py',formal_sources);pipeline=load('_pixel_t020_formal_pipeline',D/'pipeline.py',formal_sources);docker=load('_pixel_t020_formal_docker',D/'docker_contract.py',formal_sources);completion=load('_pixel_t020_formal_completion',D/'completion_contract.py',formal_sources);window=load('_pixel_t020_formal_window',D/'window_overlap.py',formal_sources)
 jobcheck=load('_pixel_t020_formal_jobcheck',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',gpu_sources);prefix=load('_pixel_t020_formal_prefix',RUN/'raw_prefix.py',gpu_sources);refsmod=load('_pixel_t020_formal_refs',B/'e6/pixel_wavefront_t020_model_timing_v2/reference.py',gpu_sources)
 runtime_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py';runtime=load('_pixel_t020_formal_runtime',runtime_path,gpu_sources);author=read(runtime_path.parent/'run_plan.json');references,_=refsmod.load()
 completion_report=completion.check(started,ended,formal['run_supervisor_command'],formal['run_supervisor_cwd'],gpu_sources,read(RUN/'run_001/weight_stats_before.json'),read(RUN/'run_001/weight_stats_after.json'),read(B/'e6/pixel_wavefront_model_t020/run_preflight001/worker_preflight.json')['weight_stats'])
 health=[]
 for line in (RUN/'monitor_run001/health.jsonl').read_text().splitlines():
  v=json.loads(line);window_ns=v.get('health_snapshot_window_monotonic_ns');ns=v.get('time_namespace')
  if not isinstance(window_ns,list) or len(window_ns)!=2 or not ns:raise ValueError('supervisor health window unavailable')
  health.append({'start_ns':window_ns[0],'end_ns':window_ns[1],'time_namespace':ns})
 def inspect(cid):
  p=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True);items=json.loads(p.stdout)
  if len(items)!=1:raise ValueError('Docker inspect count')
  return items[0]
 monitoring={'supervisor_health':health,'manual_health':None,'nvml':None,'external_watcher':None};design=read(DESIGN)
 value=pipeline.verify(gpu_plan,gpu_plan['jobs'],design['jobs'],references,RUN/'run_001',core.validate,jobcheck.check,prefix.check,docker.validate,inspect,author,runtime.container_command,RUN/'worker.py',monitoring,window.classify)
 value.update(independent_review=False,completion_evidence=completion_report,formal_plan_sha256=sha(D/'verification_plan001.json'),gpu_plan_sha256=sha(RUN/'gpu_plan001.json'),snapshot_manifest_sha256=sha(RUN/'run_001_sources/manifest.json'),run_health_sha256=sha(RUN/'monitor_run001/health.jsonl'),predictions_opened=True,GT_opened=False,scoring_performed=False)
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
