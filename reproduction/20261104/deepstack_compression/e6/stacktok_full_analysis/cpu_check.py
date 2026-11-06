"""Frozen-source pure synthetic checks; no real predictions or GT."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import admission
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();before=admission.source_hashes();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],cwd=D,text=True,capture_output=True,timeout=120,env=env);after=admission.source_hashes();a.out.mkdir(exist_ok=False);(a.out/'tests.stdout').write_text(r.stdout);(a.out/'tests.stderr').write_text(r.stderr);result={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GT_opened':False,'real_predictions_opened':False,'test_returncode':r.returncode}
with (a.out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(0 if result['status']=='passed' else 1)
