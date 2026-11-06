"""Source-bound synthetic CPU validation; never loads model, CUDA or GT."""
import argparse,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D));import gate
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=gate.verify_plan();before=gate.source_hashes();identity=gate.prior();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=D,env=env,capture_output=True,text=True,timeout=180);a.out.mkdir(exist_ok=False);(a.out/'stdout.log').write_text(r.stdout);(a.out/'stderr.log').write_text(r.stderr);after=gate.source_hashes();result={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'prior_identity':identity,'plan_sha256':gate.sha(D/'plan001.json'),'test_returncode':r.returncode,'GT_opened':False,'GPU_used':False,'model_loaded':False,'scope':'synthetic only; no real model memory result'}
with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(0 if result['status']=='passed' else 1)
