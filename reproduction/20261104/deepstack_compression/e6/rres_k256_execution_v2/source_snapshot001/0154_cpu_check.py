"""Freeze execution draft CPU tests; no model, predictions, labels or GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
paths=sorted(D.glob('*.py'))
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
a=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v','test_execution'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60);b=hashes()
if r.returncode!=0 or a!=b:raise ValueError(r.stdout+r.stderr)
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','source_sha256_before':a,'source_sha256_after':b,'tests':r.stdout+r.stderr,'GT_opened':False,'predictions_opened':False,'GPU_used':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
