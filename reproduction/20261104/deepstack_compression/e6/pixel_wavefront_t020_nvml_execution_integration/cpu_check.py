"""NoGPU entry-main integration with fake gate, model, and data only."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v6'
paths=list(D.glob('*.py'))+[E/'worker.py',E/'gate.py']
def hashes():return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
a=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v','test_worker_main'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60);b=hashes();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
v={'status':'passed' if r.returncode==0 and a==b else 'failed','source_sha256_before':a,'source_sha256_after':b,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'limitations':['real entry main but fake gate, input, and model continuation','not real Docker, NVML, CUDA or GPU evidence']}
with (D/'cpu_proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if v['status']!='passed':raise SystemExit(1)
