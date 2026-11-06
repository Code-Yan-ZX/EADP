"""Execute real host entries with absent plan and check they fail before artifacts."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/rres_k256_execution_v2';files=sorted(D.glob('*.py'))
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();cases=[]
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)/'must_not_exist.json'
 for script,args in [('verify.py',['--out',str(out)]),('snapshot.py',[]),('launch.py',['--run-id','001'])]:
  p=subprocess.run([sys.executable,str(E/script),*args],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
  if p.returncode==0 or 'gpu_plan001.json' not in p.stderr or out.exists() or (E/'run001').exists() or (E/'source_snapshot001').exists():raise ValueError('unlocked host entry '+script+'\n'+p.stderr)
  cases.append({'script':script,'returncode':p.returncode,'rejected_missing_plan':True})
after=hashes()
if before!=after:raise ValueError('sources changed')
with (D/'cpu_proof001.json').open('x') as f:json.dump({'status':'passed','before':before,'after':after,'cases':cases,'GT_opened':False,'predictions_opened':False,'GPU_used':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
