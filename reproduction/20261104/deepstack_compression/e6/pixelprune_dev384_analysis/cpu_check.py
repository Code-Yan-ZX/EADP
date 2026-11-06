"""Run only synthetic CPU tests and bind all proposed score sources."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import admission
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();before=admission.source_hashes();run=subprocess.run([sys.executable,'-m','unittest','-v','test_score'],cwd=admission.HERE,text=True,capture_output=True);print(run.stdout,end='');print(run.stderr,end='',file=sys.stderr)
if run.returncode:raise SystemExit(run.returncode)
after=admission.source_hashes()
if before!=after:raise ValueError('source changed')
a.out.mkdir(parents=True,exist_ok=False)
with (a.out/'verification.json').open('x') as f:json.dump({'status':'passed','GT_opened':False,'real_predictions_opened':False,'source_sha256_before':before,'source_sha256_after':after,'test_stderr_sha256':__import__('hashlib').sha256(run.stderr.encode()).hexdigest()},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
