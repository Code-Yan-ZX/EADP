import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution_v2';paths=sorted(D.glob('*.py'))+sorted(E.glob('*.py'))
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
a=hashes();r=subprocess.run([sys.executable,'-m','unittest','-v','test_runtime'],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60);b=hashes()
if r.returncode!=0 or a!=b:raise ValueError(r.stdout+r.stderr)
with (D/'cpu_proof002.json').open('x') as f:json.dump({'status':'passed','before':a,'after':b,'tests':r.stdout+r.stderr,'GT_opened':False,'predictions_opened':False,'GPU_used':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
