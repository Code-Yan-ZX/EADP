"""Supervised synthetic-only scoring contract, never opens real predictions/GT."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import score_selected
admission=score_selected.admission
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid test ID')
 out=HERE/('cpu_check'+a.run_id);out.mkdir(exist_ok=False)
 files=admission.sources();before=admission.source_hashes()
 proc=subprocess.run([sys.executable,'-m','unittest','-v','test_selected'],cwd=HERE,text=True,capture_output=True,timeout=120);(out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr);after=admission.source_hashes()
 if proc.returncode or 'Ran 17 tests' not in proc.stderr or before!=after:raise ValueError('scoring CPU tests failed or sources changed')
 result={'status':'passed','tests':17,'source_sha256_before':before,'source_sha256_after':after,'test_stderr_sha256':admission.sha(out/'stderr.log'),'GT_opened':False,'real_predictions_opened':False,'scope':'synthetic-only scoring and fail-closed admission'}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','tests':17}))
if __name__=='__main__':main()
