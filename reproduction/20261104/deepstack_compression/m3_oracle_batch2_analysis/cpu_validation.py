"""Source-frozen synthetic second-look scorer tests, no real data."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import analysis_gate
HERE=Path(__file__).resolve().parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);src=analysis_gate.sources();before={str(p.resolve()):analysis_gate.batch_gate.sha(p) for p in src}
 names=('test_gate','test_score_pipeline','test_native_predictions','test_paired','test_source_pinning');runs=[subprocess.run([sys.executable,'-m','unittest','-v',name],cwd=HERE,text=True,capture_output=True,timeout=120) for name in names];(a.out/'stderr.log').write_text('\n'.join(p.stderr for p in runs));(a.out/'stdout.log').write_text('\n'.join(p.stdout for p in runs));after={str(p.resolve()):analysis_gate.batch_gate.sha(p) for p in src};ok=all(p.returncode==0 for p in runs) and before==after
 with (a.out/'verification.json').open('x') as f:json.dump({'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'stderr_sha256':analysis_gate.batch_gate.sha(a.out/'stderr.log'),'GT_opened':False,'predictions_opened':False,'gpu_used':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
