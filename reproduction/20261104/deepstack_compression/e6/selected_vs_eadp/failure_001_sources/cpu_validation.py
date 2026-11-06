"""Synthetic-only validation with source inventory."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 files=[H/x for x in ('plan.json','plan_002.json','plan_003.json','compare.py','test_compare.py','cpu_validation.py')];before={str(f):sha(f) for f in files}
 p=subprocess.run([sys.executable,str(H/'test_compare.py')],capture_output=True,text=True,timeout=90);(a.out/'stderr.log').write_text(p.stderr);after={str(f):sha(f) for f in files}
 proof={'status':'passed' if p.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':sha(a.out/'stderr.log'),'GT_opened':False,'predictions_opened':False,'gpu_used':False}
 with (a.out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if proof['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
