import hashlib,json,os,subprocess,sys
from pathlib import Path
import design
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 files=[D/x for x in ('design.py','test_design.py','freeze.py')];before={str(p):sha(p) for p in files}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-m','unittest','-v','test_design'],cwd=D,env=env,capture_output=True,text=True,timeout=30);(D/'cpu_test001.log').write_text(r.stdout+r.stderr)
 after={str(p):sha(p) for p in files}
 if r.returncode or before!=after:raise ValueError('CPU test or source changed')
 x=design.plan();x['source_sha256']=after;x['cpu_tests_passed']=True
 with (D/'design001.json').open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
