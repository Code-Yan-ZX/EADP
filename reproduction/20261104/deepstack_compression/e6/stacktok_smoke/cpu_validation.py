"""Bounded synthetic tests and immutable source inventory, no GPU or answers."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from admission import sources,prior
from real_worker import sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();before={str(x.resolve()):sha(x) for x in sources()};identity=prior();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');cwd=Path(__file__).resolve().parent;commands=[[sys.executable,'-m','unittest','discover','-s',str(cwd),'-p','test_*.py','-v'],[sys.executable,'-m','unittest','discover','-s',str(cwd.parent/'stacktok_audit/prototype'),'-p','test_*.py','-v']];a.out.mkdir(exist_ok=False);codes=[]
for i,cmd in enumerate(commands):
 r=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=cwd,timeout=120);(a.out/('test%d.stdout'%i)).write_text(r.stdout);(a.out/('test%d.stderr'%i)).write_text(r.stderr);codes.append(r.returncode)
after={str(x.resolve()):sha(x) for x in sources()};result={'status':'passed' if codes==[0,0] and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'prior_identity':identity,'test_returncodes':codes,'python':sys.version,'interpreter':sys.executable,'GPU_used':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'limitations':['Synthetic hooks and original core only; Qwen3 runtime identity and KV need real-model smoke.','Direct original selector import bypasses unavailable author package dependency.']}
with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(0 if result['status']=='passed' else 1)
