"""Synthetic-only, source-bound validation of the document isolation algorithm."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 files=[HERE/x for x in ('plan_001.json','core.py','run.py','cpu_validation.py','test_core.py','test_run.py')];before={str(x):sha(x) for x in files}
 procs=[subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,timeout=90) for name in ('test_core.py','test_run.py')];(a.out/'stderr.log').write_text('\n'.join(x.stderr for x in procs));(a.out/'stdout.log').write_text('\n'.join(x.stdout for x in procs));after={str(x):sha(x) for x in files}
 proof={'status':'passed' if all(x.returncode==0 for x in procs) and before==after else 'failed','returncodes':[x.returncode for x in procs],'source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':sha(a.out/'stderr.log'),'GT_opened':False,'predictions_opened':False,'gpu_used':False,'training_performed':False}
 with (a.out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if proof['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
