"""Bounded synthetic contract tests; no GPU/model/new prediction or GT."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def hashes():return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
a=hashes();r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-v'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120);b=hashes();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
v={'status':'passed' if r.returncode==0 and a==b else 'failed','source_sha256_before':a,'source_sha256_after':b,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'timing_results_opened':False,'limits':['fake CPU models/inputs only','source and noGPU preflight reviews do not authorize GPU']}
with (D/'cpu_proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if v['status']!='passed':raise SystemExit(1)
