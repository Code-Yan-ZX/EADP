"""Freeze independent no-GT acceptance/review scripts around synthetic tests."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 src=[HERE/x for x in ('cpu_validation.py','verify.py','container_identity.py','snapshot.py','test_verify.py','test_container_identity.py','test_review.py')]+[ROOT/'e1_independent_review/review_m3batch2_panel.py'];before={str(x):sha(x) for x in src}
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_container_identity','test_verify','test_review'],cwd=HERE,text=True,capture_output=True,timeout=120);(a.out/'stdout.log').write_text(p.stdout);(a.out/'stderr.log').write_text(p.stderr);after={str(x):sha(x) for x in src}
 ok=p.returncode==0 and before==after and 'Ran 8 tests' in p.stderr and p.stderr.rstrip().endswith('OK')
 out={'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':sha(a.out/'stderr.log'),'tests':8,'no_real_predictions_or_answers':True}
 with (a.out/'verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
