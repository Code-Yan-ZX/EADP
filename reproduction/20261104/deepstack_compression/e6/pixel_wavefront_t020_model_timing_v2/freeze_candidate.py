"""CPU source freeze, not a real GPU/worker/launch admission."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';RUN=B/'e6/pixel_wavefront_model_t020_run_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 out=D/'candidate_cpu_proof001.json'
 if out.exists():raise ValueError('already frozen')
 prior=read(RUN/'execution_plan001.json');sources=dict(prior['source_sha256']);paths=list(D.glob('*.py'))+[D/'design001.json',I/'pixel_wavefront_t020_run_v2_result_review.json',I/'review_pixel_wavefront_t020_run_v2_result.py',I/'monitor_pixel_wavefront_t020_run_v2_result/started.json',I/'monitor_pixel_wavefront_t020_run_v2_result/terminal.json',B/'e6/pixel_wavefront_t020_accept_v2/verification001.json',B/'e6/pixel_wavefront_t020_accept_v2/monitor_verify001/started.json',B/'e6/pixel_wavefront_t020_accept_v2/monitor_verify001/terminal.json']
 formal=read(B/'e6/pixel_wavefront_t020_accept_v2/verification001.json');review=read(I/'pixel_wavefront_t020_run_v2_result_review.json');assert formal['status']=='passed' and review['status']=='passed' and review['formal_sha256']==sha(B/'e6/pixel_wavefront_t020_accept_v2/verification001.json')
 # Include the six exact accepted records; tests read these DEV predictions only
 # to check prior model identity, never labels.
 paths.extend(Path(x) for x in formal['files_sha256'] if x.endswith('/observations/record.json'))
 for p in paths:sources[str(p.resolve())]=sha(p)
 before=dict(sources);tests=['test_contract.py','test_pipeline.py','test_selector_instrumentation.py','test_timer_worker.py','test_nvml_contract.py','test_reference.py','test_model_candidate.py'];outputs={}
 for name in tests:
  r=subprocess.run([sys.executable,str(D/name)],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);outputs[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 after={p:sha(p) for p in before};ok=all(x['returncode']==0 for x in outputs.values()) and before==after
 result={'status':'passed' if ok else 'failed','scope':'CPU candidate only; no GPU worker or container authorization','source_sha256_before':before,'source_sha256_after':after,'tests':outputs,'identity_formal_sha256':sha(B/'e6/pixel_wavefront_t020_accept_v2/verification001.json'),'identity_review_sha256':sha(I/'pixel_wavefront_t020_run_v2_result_review.json'),'GT_opened':False,'existing_DEV_predictions_opened':True,'model_loaded':False,'gpu_used':False,'GPU_authorized':False,'limitations':['No bounded model controller or actual container preflight yet.','CUDA inference tensor indices are cloned on-device for both arms in timing interval; may alter measured TTFT.','NVML validator is CPU-only and does not itself sample GPU.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
