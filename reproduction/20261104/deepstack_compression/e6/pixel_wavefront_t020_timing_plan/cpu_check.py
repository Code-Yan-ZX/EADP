import hashlib,json,os,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def sources():return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))}
def main():
 before=sources();e=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_freeze'],cwd=D,env=e,capture_output=True,text=True,timeout=60);after=sources();(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 v={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,'GPU_used':False,'GT_opened':False,'predictions_opened':False}
 with (D/'cpu_proof001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if v['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
