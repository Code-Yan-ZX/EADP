"""Run synthetic analysis tests only; never open full-run timing values."""
import argparse,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
import analyze
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run ID')
 out=HERE/('cpu_check'+a.run_id);out.mkdir(exist_ok=False);before=analyze.gate.source_hashes()
 proc=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis_core','test_analysis_gate','test_manual_overlap'],cwd=HERE,text=True,capture_output=True,timeout=120)
 (out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr);after=analyze.gate.source_hashes()
 if proc.returncode or 'Ran 13 tests' not in proc.stderr or before!=after:raise ValueError('synthetic tests failed or sources changed')
 result={'status':'passed','tests':13,'source_sha256_before':before,'source_sha256_after':after,'numpy_version':importlib.metadata.version('numpy'),'full_timing_values_opened':False,'GT_opened':False,'scope':'synthetic-only tests; no real timing values','stderr_sha256':analyze.gate.sha(out/'stderr.log')}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','tests':13}))
if __name__=='__main__':main()
