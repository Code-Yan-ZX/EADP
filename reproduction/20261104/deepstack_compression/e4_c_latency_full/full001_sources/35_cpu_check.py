"""Freeze sources around supervised fake CPU contract checks."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from full_admission import sources
from identity_gate import sha
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run ID')
 out=HERE/('cpu_check'+a.run_id);out.mkdir(exist_ok=False);before={str(p):sha(p) for p in sources()}
 proc=subprocess.run([sys.executable,'-m','unittest','-v','test_full','test_lifecycle'],cwd=HERE,text=True,capture_output=True,timeout=120)
 (out/'stdout.log').write_text(proc.stdout);(out/'stderr.log').write_text(proc.stderr);after={str(p):sha(p) for p in sources()}
 if proc.returncode or before!=after:raise ValueError('full CPU tests failed or source changed')
 result={'status':'passed','returncode':proc.returncode,'source_sha256_before':before,'source_sha256_after':after,'test_stderr_sha256':sha(out/'stderr.log'),'scope':'fake CPU only; no GPU, predictions or GT'}
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed'}))
if __name__=='__main__':main()
